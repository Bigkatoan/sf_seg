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


def focal_loss(logits: torch.Tensor, target: torch.Tensor,
               gamma: float = 2.0,
               weight: torch.Tensor | None = None) -> torch.Tensor:
    """Multi-class focal loss (Lin et al., 2017).

    Down-weights easy examples exponentially so training focuses on hard,
    rare foreground classes instead of the dominant background.

    focal(p_t) = (1 - p_t)^gamma * CE(p_t)

    gamma=0  → plain cross-entropy
    gamma=2  → standard focal (recommended for severe class imbalance)
    """
    ce = F.cross_entropy(logits, target, weight=weight, reduction='none')  # (B,H,W)
    pt = torch.exp(-ce)                                                    # model confidence
    return ((1 - pt) ** gamma * ce).mean()


def focal_iou_loss(logits: torch.Tensor, target: torch.Tensor,
                   class_weights: torch.Tensor | None = None,
                   gamma: float = 2.0,
                   focal_w: float = 0.6, iou_w: float = 0.4,
                   no_obj_weight: float = 0.1) -> torch.Tensor:
    """Focal loss + soft-IoU — best choice for heavily imbalanced datasets.

    Focal replaces CE: it ignores easy background pixels and pushes the
    model to learn rare foreground classes much more aggressively.
    """
    fl  = focal_loss(logits, target, gamma=gamma, weight=class_weights)
    iou = multiclass_iou_loss(logits, target, no_obj_weight=no_obj_weight)
    return focal_w * fl + iou_w * iou


def pure_focal_iou_loss(logits: torch.Tensor, target: torch.Tensor,
                        gamma: float = 2.0, eps: float = 1e-3,
                        absent_weight: float = 0.2) -> torch.Tensor:
    """Unified Focal-IoU with Spatial Mass Penalty for absent classes.

    Two separate objectives, each operating at its natural granularity:

      Present classes  →  (1 − IoU_c)^(γ+1)        [class-level, geometric]
      Absent  classes  →  mean_{x,y}(p_c(x,y))      [pixel-level, spatial]

    Why Spatial Mass Penalty instead of no_obj_weight on IoU:
      For an absent class c, p_c(x,y) is a spatial probability mass function
      that encodes WHERE the model thinks class c might exist (even though it
      doesn't). Scaling the IoU term (no_obj_weight) discards this spatial
      signal. The mass penalty min{mean(p_c)} instead directly constrains the
      model to allocate ZERO spatial attention to absent classes — a strictly
      stronger and more informative constraint at pixel level.

      Absent IoU ≈ ε / (Σ p_c + ε): measures overlap (always 0).
      Absent mass = mean(p_c):        measures total spatial allocation.

    Combined formula per image:
      L = (1/N_present) Σ_{c∈present} (1−IoU_c)^(γ+1)
        + absent_weight × (1/N_absent) Σ_{c∈absent} mean(p_c)

    γ=0 → plain soft-IoU + mass penalty
    γ=2 → cubic focal-IoU + mass penalty  [default]
    """
    B, C, H, W = logits.shape
    # float32 — softmax in float16 overflows when logits have large range (AMP)
    probs = F.softmax(logits.float(), dim=1)                          # (B, C, H, W)

    tgt_oh    = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()
    pred_flat = probs.view(B, C, -1)                                  # (B, C, H*W)
    tgt_flat  = tgt_oh.view(B, C, -1)

    inter = (pred_flat * tgt_flat).sum(-1)                            # (B, C)
    union = pred_flat.sum(-1) + tgt_flat.sum(-1) - inter
    iou   = ((inter + eps) / (union + eps)).clamp(0.0, 1.0)          # (B, C) ∈ [0,1]

    present = (tgt_flat.sum(-1) > 0).float()                         # (B, C)
    absent  = 1.0 - present                                           # (B, C)

    # Present: focal-IoU — (1-IoU)^(γ+1), normalised by number of present classes
    focal_iou    = (1 - iou).pow(1 + gamma) * present                # (B, C)
    n_present    = present.sum().clamp(min=1)
    present_loss = focal_iou.sum() / n_present

    # Absent: spatial mass penalty — directly minimise average predicted
    # probability across all pixels for each absent class
    spatial_mass = pred_flat.mean(-1)                                 # (B, C)
    absent_loss  = (spatial_mass * absent).sum() / absent.sum().clamp(min=1)

    return present_loss + absent_weight * absent_loss


# ── Attention guidance (IoU-based, class-specific) ───────────────────────────

def attention_guide_loss(attn: torch.Tensor, target: torch.Tensor,
                         num_classes: int, topk: int = 1,
                         guide_size: int = 14, eps: float = 1e-3) -> torch.Tensor:
    """Supervise attention maps to align with GT masks via IoU-based assignment.

    For each present class c in an image:
      1. Compute IoU between GT mask_c and every attention channel  [detached]
      2. Select top-k attention channels with highest IoU → "responsible" heads
      3. Supervise them with Dice loss to better match mask_c

    The IoU-based selection (step 1-2) is detached from the gradient graph so
    the selector is stable and gradients only flow through the supervision loss.

    Complexity:
      Downsampling to guide_size² before the bmm keeps compute tractable:
        bmm: B × num_classes × C_attn × guide_size²
        guide_size=14 → ~60M ops/batch  vs  full 112×112 → ~3.9B ops

    attn       : (B, C_attn, H', W') — attention maps from the model
    target     : (B, H, W)           — class label map (long)
    topk       : channels selected per class (1 = strict assignment)
    guide_size : spatial resolution for IoU computation
    """
    B, C_attn, _, _ = attn.shape

    # float32 — attention maps may be float16 under AMP; bmm accumulation needs precision
    attn_s = F.adaptive_avg_pool2d(attn.float(), (guide_size, guide_size))  # (B,K,g,g)
    tgt_s  = F.interpolate(
        target.float().unsqueeze(1), (guide_size, guide_size), mode='nearest'
    ).squeeze(1).long()                                               # (B, g, g)
    L = guide_size * guide_size

    gt_oh  = F.one_hot(tgt_s, num_classes).permute(0, 3, 1, 2).float()
    gt_f   = gt_oh.view(B, num_classes, L)                           # (B, C, L)
    atn_f  = attn_s.view(B, C_attn, L)                              # (B, K, L)

    # IoU matrix (B, num_classes, C_attn) — detached, selection is non-differentiable
    with torch.no_grad():
        inter   = torch.bmm(gt_f, atn_f.transpose(1, 2))            # (B, C, K)
        union   = gt_f.sum(-1, keepdim=True) + atn_f.sum(-1).unsqueeze(1) - inter
        iou_m   = (inter + eps) / (union + eps)                      # (B, C, K)
        _, topk_idx = iou_m.topk(topk, dim=2)                       # (B, C, topk)

    # Gather top-k attention maps for each class: (B, num_classes, topk, L)
    idx       = topk_idx.unsqueeze(-1).expand(-1, -1, -1, L)
    atn_exp   = atn_f.unsqueeze(1).expand(-1, num_classes, -1, -1)
    topk_maps = torch.gather(atn_exp, 2, idx).mean(2)               # (B, C, L)

    # Normalise to [0,1] for comparison with binary GT mask
    mx        = topk_maps.amax(-1, keepdim=True).clamp(min=eps)
    topk_norm = (topk_maps / mx).clamp(0, 1)                        # (B, C, L)

    # Dice loss — only for present foreground classes (skip background idx=0)
    present = (gt_f.sum(-1) > 0)                                     # (B, C)
    present[:, 0] = False
    if not present.any():
        return torch.zeros(1, device=attn.device).squeeze()

    p    = topk_norm[present]                                         # (N, L)
    t    = gt_f[present]                                              # (N, L)
    dice = 1 - (2 * (p * t).sum(-1) + eps) / (p.sum(-1) + t.sum(-1) + eps)
    return dice.mean()


def attention_exclusivity_loss(attn: torch.Tensor, target: torch.Tensor,
                                num_classes: int,
                                guide_size: int = 14,
                                eps: float = 1e-3) -> torch.Tensor:
    """Winner-takes-all: each attention channel specialises for at most one class.

    For every channel k, its "winner" is the class with the highest IoU.
    We then penalise that channel for overlapping with every OTHER class.

    Non-winner IoU → 0  means the channel only responds to one class concept.

    Paired with attention_guide_loss this creates a stable competition:
      guide_loss     : winner   SHOULD match its class mask  (pull toward GT)
      exclusivity_loss: non-winners SHOULD NOT match any mask (push away from others)

    After training the assignment matrix  channel_k → class_c  can be read off
    by averaging per-channel winner classes on the val set, allowing attention
    maps to be used as standalone class-specific encoders.

    Gradient design:
      Winner selection uses torch.no_grad() → the selector is a constant mask.
      Gradients flow only through the non-winner IoU terms, pushing those
      channel-class IoUs toward 0 without touching the winner channel.
    """
    B, K, _, _ = attn.shape

    # float32 — attn may be float16 under AMP
    attn_s = F.adaptive_avg_pool2d(attn.float(), (guide_size, guide_size))  # (B,K,g,g)
    tgt_s  = F.interpolate(
        target.float().unsqueeze(1), (guide_size, guide_size), mode='nearest'
    ).squeeze(1).long()                                               # (B, g, g)
    L = guide_size * guide_size

    gt_f  = F.one_hot(tgt_s, num_classes).permute(0, 3, 1, 2).float().view(B, num_classes, L)
    atn_f = attn_s.view(B, K, L)

    present = gt_f.sum(-1) > 0   # (B, C)
    present[:, 0] = False         # skip background
    if not present.any():
        return torch.zeros(1, device=attn.device).squeeze()

    # IoU(channel k, class c) — full gradient for the loss term
    inter      = torch.bmm(gt_f, atn_f.transpose(1, 2))              # (B, C, K)
    union      = gt_f.sum(-1, keepdim=True) + atn_f.sum(-1).unsqueeze(1) - inter
    iou_m      = ((inter + eps) / (union + eps)).clamp(0.0, 1.0)     # (B, C, K)
    iou_m      = iou_m * present.float().unsqueeze(-1)               # mask absent
    iou_per_ch = iou_m.permute(0, 2, 1)                              # (B, K, C)

    # Winner mask — detached so selection carries no gradient
    with torch.no_grad():
        best      = iou_per_ch.max(dim=-1, keepdim=True).values      # (B, K, 1)
        winner    = (iou_per_ch == best).float()                      # (B, K, C)

    # Non-winner overlap: what each channel incorrectly covers
    non_winner_iou = iou_per_ch * (1.0 - winner)                     # (B, K, C)
    denom = (1.0 - winner).sum().clamp(min=1)
    return non_winner_iou.sum() / denom


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
        idx = torch.randperm(L, device=a.device)[:2048]
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
