#!/usr/bin/env python3
"""Loss functions for segmentation tasks."""
from typing import Optional

import torch
import torch.nn.functional as F


# ── Binary losses (num_classes == 1) ──────────────────────────────────────────

def iou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3, reduction: str = "mean") -> torch.Tensor:
    """Soft IoU loss (1 - IoU) for binary segmentation.

    pred   : (B, 1, H, W) probabilities in [0, 1] — apply sigmoid before calling.
    target : (B, 1, H, W) or (B, H, W) binary float.
    """
    pred = pred.float()
    target = target.float()

    if pred.shape != target.shape:
        if pred.dim() == 4 and target.dim() == 3 and pred.size(0) == target.size(0):
            target = target.unsqueeze(1)
        else:
            raise ValueError(f"pred and target must have same shape, got {pred.shape} vs {target.shape}")

    b = pred.shape[0]
    pred_flat   = pred.view(b, -1)
    target_flat = target.view(b, -1)

    inter = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - inter
    iou   = (inter + eps) / (union + eps)
    loss  = 1.0 - iou

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def mse_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """MSE loss for binary segmentation.

    pred   : (B, 1, H, W) probabilities in [0, 1].
    target : (B, 1, H, W) or (B, H, W) binary float.
    """
    pred = pred.float()
    target = target.float()

    if pred.shape != target.shape:
        if pred.dim() == 4 and target.dim() == 3 and pred.size(0) == target.size(0):
            target = target.unsqueeze(1)
        else:
            raise ValueError(f"pred and target must have same shape, got {pred.shape} vs {target.shape}")

    loss = (pred - target) ** 2

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def combine_losses(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.1, reduction: str = "mean") -> torch.Tensor:
    """Combined binary loss: alpha * IoU + (1-alpha) * MSE.

    pred   : (B, 1, H, W) probabilities in [0, 1].
    target : (B, 1, H, W) or (B, H, W) binary float.
    """
    return alpha * iou_loss(pred, target, reduction=reduction) + (1 - alpha) * mse_loss(pred, target, reduction=reduction)


# ── Multi-class losses (num_classes > 1) ──────────────────────────────────────

def multiclass_iou_loss(logits: torch.Tensor, target: torch.Tensor,
                        eps: float = 1e-3, no_obj_weight: float = 0.1) -> torch.Tensor:
    """Soft mean-IoU loss for multi-class segmentation with no-object weighting.

    logits        : (B, C, H, W) raw logits — softmax applied internally.
    target        : (B, H, W) long tensor with class indices in [0, C-1].
    no_obj_weight : weight applied to classes absent from a sample's target
                    (0 = ignore absent classes, 1 = treat same as present).
                    Prevents absent classes from producing a spuriously large
                    loss (IoU≈0 even though the class simply isn't in the image).
    """
    B, C, H, W = logits.shape
    probs = F.softmax(logits, dim=1)                                # (B, C, H, W)

    target_onehot = F.one_hot(target, num_classes=C)                # (B, H, W, C)
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()      # (B, C, H, W)

    pred_flat   = probs.view(B, C, -1)           # (B, C, H*W)
    target_flat = target_onehot.view(B, C, -1)   # (B, C, H*W)

    inter = (pred_flat * target_flat).sum(dim=-1)                   # (B, C)
    union = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1) - inter
    iou   = (inter + eps) / (union + eps)                           # (B, C)

    # Per-sample, per-class presence flag: 1 if class appears, 0 if absent
    present = (target_flat.sum(dim=-1) > 0).float()                 # (B, C)
    # Weight: present classes → 1.0,  absent classes → no_obj_weight
    weight = present + no_obj_weight * (1.0 - present)              # (B, C)

    weighted_loss = (1.0 - iou) * weight                            # (B, C)
    # Normalise by total weight so the scale stays ~1 regardless of class balance
    return weighted_loss.sum() / weight.sum().clamp(min=1.0)


def ce_iou_loss(logits: torch.Tensor, target: torch.Tensor,
                class_weights: torch.Tensor | None = None,
                ce_weight: float = 0.5, iou_weight: float = 0.5,
                no_obj_weight: float = 0.1) -> torch.Tensor:
    """Combined CrossEntropy + soft-IoU loss for multi-class segmentation.

    logits        : (B, C, H, W) raw logits.
    target        : (B, H, W) long tensor with class indices in [0, C-1].
    class_weights : optional (C,) float tensor for inverse-frequency weighting in CE.
    no_obj_weight : passed to multiclass_iou_loss (see docstring there).
    """
    ce  = F.cross_entropy(logits, target, weight=class_weights)
    iou = multiclass_iou_loss(logits, target, no_obj_weight=no_obj_weight)
    return ce_weight * ce + iou_weight * iou


# ── Attention regulariser (class-agnostic) ────────────────────────────────────

def diversity_loss(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Penalize cosine similarity between attention channels (Gram matrix off-diagonal).

    attn: (B, C, H, W) raw attention maps from clamped softmax.
    """
    B, C, H, W = attn.shape
    a = attn.view(B, C, H * W)

    # Subsample spatial locations to save memory
    L = a.shape[-1]
    if L > 2048:
        # Generate permutation on CPU — XLA (TPU) cannot compile int64 rng ops
        idx = torch.randperm(L)[:2048].to(a.device)
        a = a[:, :, idx]

    a    = F.normalize(a.float(), dim=-1, eps=eps)                 # unit norm per channel
    gram = torch.bmm(a, a.transpose(1, 2))                         # (B, C, C)
    eye  = torch.eye(C, device=attn.device, dtype=gram.dtype)
    off_diag = gram * (1.0 - eye)
    return (off_diag ** 2).sum(dim=[1, 2]).mean() / (C * (C - 1))


if __name__ == "__main__":
    # smoke tests
    # binary
    p  = torch.sigmoid(torch.randn(2, 1, 128, 128))
    y  = (torch.rand(2, 1, 128, 128) > 0.5).float()
    print("iou_loss:      ", iou_loss(p, y).item())
    print("combine_losses:", combine_losses(p, y).item())

    # multi-class
    logits = torch.randn(2, 81, 128, 128)
    target = torch.randint(0, 81, (2, 128, 128))
    print("multiclass_iou_loss:", multiclass_iou_loss(logits, target).item())
    print("ce_iou_loss:        ", ce_iou_loss(logits, target).item())

    # attention diversity
    attn = torch.rand(2, 64, 32, 32)
    print("diversity_loss:", diversity_loss(attn).item())
