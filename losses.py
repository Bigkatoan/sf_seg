#!/usr/bin/env python3
"""Loss functions for segmentation tasks."""
from typing import Optional

import torch
import torch.nn.functional as F


def iou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3, reduction: str = "mean") -> torch.Tensor:
    """IoU loss (1 - IoU).

    Formula (per-sample):
        inter = gt * pred
        union = (gt + pred) - inter
        iou = (inter.sum + eps) / (union.sum + eps)
        loss = 1 - iou

    pred is expected to be probabilities in [0,1] (i.e., model with final sigmoid).
    target should be binary (0/1) of same shape as pred.
    Returns: scalar loss (averaged over batch when reduction='mean').
    """
    pred = pred.float()
    target = target.float()

    if pred.shape != target.shape:
        # allow target to be (B,H,W) when pred is (B,1,H,W)
        if pred.dim() == 4 and target.dim() == 3 and pred.size(0) == target.size(0):
            target = target.unsqueeze(1)
        else:
            raise ValueError(f"pred and target must have same shape, got {pred.shape} vs {target.shape}")

    b = pred.shape[0]
    pred_flat = pred.view(b, -1)
    target_flat = target.view(b, -1)

    inter = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    loss = 1.0 - iou

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss

def mse_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Mean Squared Error loss."""
    pred = pred.float()
    target = target.float()

    if pred.shape != target.shape:
        # allow target to be (B,H,W) when pred is (B,1,H,W)
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

def combine_losses(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.9, reduction: str = "mean") -> torch.Tensor:
    """Combined loss: alpha * IoU + (1-alpha) * MSE."""
    iou = iou_loss(pred, target, reduction=reduction)
    mse = mse_loss(pred, target, reduction=reduction)
    return alpha * iou + (1 - alpha) * mse


def diversity_loss(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Penalize cosine similarity between attention channels (Gram matrix off-diagonal).

    attn: (B, C, H, W) raw attention maps from clamped softmax.

    For each sample, compute the C×C Gram matrix of L2-normalised channel vectors.
    Off-diagonal entries are cosine similarities in [-1, 1]; we minimise their
    squared values so every pair of channels attends to different regions.

    Loss = 0  when all channels are perfectly orthogonal.
    Loss = 1  when all channels are identical.
    """
    B, C, H, W = attn.shape
    a = attn.view(B, C, H * W)

    # Subsample spatial locations: Gram matrix gradient signal không đổi
    # khi dùng random subset — tiết kiệm O(L) → O(max_pixels) cho bmm
    L = a.shape[-1]
    if L > 2048:
        idx = torch.randperm(L, device=a.device)[:2048]
        a = a[:, :, idx]

    a = F.normalize(a.float(), dim=-1, eps=eps)                # unit norm per channel
    gram = torch.bmm(a, a.transpose(1, 2))                     # (B, C, C) cosine sims
    eye = torch.eye(C, device=attn.device, dtype=gram.dtype)
    off_diag = gram * (1.0 - eye)
    return (off_diag ** 2).sum(dim=[1, 2]).mean() / (C * (C - 1))


if __name__ == "__main__":
    # quick smoke test
    x = torch.randn(2, 3, 128, 128)
    y = (torch.rand(2, 1, 128, 128) > 0.5).float()
    p = torch.sigmoid(torch.randn(2, 1, 128, 128))
    print("iou_loss:     ", iou_loss(p, y).item())
    print("combine_losses:", combine_losses(p, y).item())
    attn = torch.rand(2, 64, 128, 128)
    print("diversity_loss:", diversity_loss(attn).item())