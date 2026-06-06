"""sf_loss — unified training loss for sf_seg.

Shared tensors (computed once per step):
    probs       softmax(logits)          used by: seg IoU
    ce_raw      CE per pixel (no_reduce) used by: seg focal + boundary weighting
    attn_s      pool(attn, guide_size²)  used by: guide, excl
    gt_f        one_hot(target_14)       used by: guide, excl
    iou_m_ng    no-grad BMM(gt_f, atn_f) used by: guide winner + excl winner mask
"""
from __future__ import annotations
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SFLossConfig:
    num_classes:      int   = 151
    # Term weights
    boundary_weight:  float = 0.3
    diversity_weight: float = 0.1
    guide_weight:     float = 0.3
    excl_weight:      float = 0.2
    # Seg — matches focal_iou_loss defaults
    focal_gamma:      float = 2.0
    focal_w:          float = 0.6   # weight for focal CE in seg term
    iou_w:            float = 0.4   # weight for soft-IoU in seg term
    no_obj_weight:    float = 0.1   # weight for absent classes in IoU
    # Boundary
    edge_weight:      float = 4.0
    corner_weight:    float = 6.0
    boundary_dilate:  int   = 2
    # Attention
    guide_size:       int   = 14

    @classmethod
    def from_args(cls, args) -> "SFLossConfig":
        return cls(
            num_classes=getattr(args, "num_classes", 151),
            boundary_weight=getattr(args, "boundary_weight", 0.3),
            diversity_weight=getattr(args, "diversity_weight", 0.1),
            guide_weight=getattr(args, "attn_guide_weight", 0.3),
            excl_weight=getattr(args, "attn_exclusive_weight", 0.2),
            focal_gamma=2.0,
            no_obj_weight=getattr(args, "no_obj_weight", 0.1),
            edge_weight=getattr(args, "edge_weight", 4.0),
            corner_weight=getattr(args, "corner_weight", 6.0),
        )


# ── Term implementations ──────────────────────────────────────────────────────

def _seg_term(target: torch.Tensor, probs: torch.Tensor,
              ce_raw: torch.Tensor, cfg: SFLossConfig,
              eps: float = 1e-3) -> torch.Tensor:
    """Replicate focal_iou_loss exactly, reusing shared probs and ce_raw.

    focal_w * focal_CE  +  iou_w * soft_IoU(no_obj_weight)
    — identical gradient signal to the pre-sf_loss focal_iou_loss call.
    """
    B, C = probs.shape[:2]

    # Focal CE — reuses shared ce_raw (no second forward through CE)
    pt    = torch.exp(-ce_raw.clamp(max=100.0))
    f_ce  = ((1.0 - pt) ** cfg.focal_gamma * ce_raw).mean()

    # Soft IoU — reuses shared probs (no second softmax)
    tgt_oh    = F.one_hot(target.clamp(0, C - 1), C).permute(0, 3, 1, 2).float()
    pred_flat = probs.view(B, C, -1)
    tgt_flat  = tgt_oh.view(B, C, -1)

    inter   = (pred_flat * tgt_flat).sum(-1)                       # (B,C)
    union   = pred_flat.sum(-1) + tgt_flat.sum(-1) - inter
    iou     = ((inter + eps) / (union + eps).clamp(min=eps)).clamp(0., 1.)

    present = (tgt_flat.sum(-1) > 0).float()
    weight  = present + cfg.no_obj_weight * (1.0 - present)        # no_obj for absent
    iou_l   = ((1.0 - iou) * weight).sum() / weight.sum().clamp(min=1.)

    return cfg.focal_w * f_ce + cfg.iou_w * iou_l


def _boundary_term(ce_raw: torch.Tensor, target: torch.Tensor,
                   cfg: SFLossConfig) -> torch.Tensor:
    """Focal CE weighted by edge + corner importance map, reusing shared ce_raw."""
    with torch.no_grad():
        t   = target.float().unsqueeze(1)

        lap  = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                             dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
        edge = (F.conv2d(t, lap, padding=1).abs() > 0).float()

        kx     = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
        ky     = kx.transpose(2, 3).contiguous()
        c      = F.conv2d(t, kx, padding=1).abs() * F.conv2d(t, ky, padding=1).abs()
        corner = c / c.amax(dim=[2, 3], keepdim=True).clamp(min=1e-8)

        if cfg.boundary_dilate > 0:
            k      = 2 * cfg.boundary_dilate + 1
            edge   = F.max_pool2d(edge,   k, stride=1, padding=cfg.boundary_dilate)
            corner = F.max_pool2d(corner, k, stride=1, padding=cfg.boundary_dilate)

        w = (1.0
             + cfg.edge_weight   * edge.squeeze(1)
             + cfg.corner_weight * corner.squeeze(1)).clamp(max=20.0)

    # Reuse shared ce_raw — no second cross_entropy call
    pt    = torch.exp(-ce_raw.clamp(max=100.0))
    focal = (1.0 - pt).pow(cfg.focal_gamma) * ce_raw
    focal = torch.nan_to_num(focal, nan=0.0, posinf=100.0, neginf=0.0)

    return torch.nan_to_num((focal * w).sum() / w.sum().clamp(min=1.0), nan=0.0)


def _attn_term(attn_s: torch.Tensor, gt_f: torch.Tensor,
               atn_f: torch.Tensor, iou_m_ng: torch.Tensor,
               cfg: SFLossConfig, eps: float = 1e-3):
    """Guide + exclusivity losses sharing the pre-computed no-grad IoU matrix."""
    B, num_classes, L = gt_f.shape
    B, K, _           = atn_f.shape

    present = (gt_f.sum(-1) > 0)
    present[:, 0] = False

    if not present.any():
        zero = atn_f.new_tensor(0.0)
        return zero, zero

    # ── Guide: winner channel → Dice with its GT class mask ──────────────────
    _, topk_idx = iou_m_ng.topk(1, dim=2)                              # (B,C,1)
    idx         = topk_idx.unsqueeze(-1).expand(-1, -1, -1, L)
    atn_exp     = atn_f.unsqueeze(1).expand(-1, num_classes, -1, -1)   # (B,C,K,L)
    w_maps      = torch.gather(atn_exp, 2, idx).squeeze(2)             # (B,C,L)
    mx          = w_maps.amax(-1, keepdim=True).clamp(min=1e-6)
    w_norm      = (w_maps / mx).clamp(0.0, 1.0)

    p    = w_norm[present]
    t    = gt_f[present]
    dice = 1.0 - (2.0 * (p * t).sum(-1) + eps) / (p.sum(-1) + t.sum(-1) + eps)
    guide = dice.mean()

    # ── Excl: non-winner channels must not overlap other classes ─────────────
    inter_g = torch.bmm(gt_f, atn_f.transpose(1, 2))                  # (B,C,K)
    union_g = gt_f.sum(-1, keepdim=True) + atn_f.sum(-1).unsqueeze(1) - inter_g
    iou_g   = ((inter_g + eps) / (union_g + eps)).clamp(0.0, 1.0)
    iou_g   = iou_g * present.float().unsqueeze(-1)
    iou_pc  = iou_g.permute(0, 2, 1)                                   # (B,K,C)

    with torch.no_grad():
        iou_pc_ng = iou_m_ng.permute(0, 2, 1) * present.float().unsqueeze(-2)
        best      = iou_pc_ng.max(dim=-1, keepdim=True).values
        winner    = (iou_pc_ng == best).float()                        # (B,K,C)

    non_win = iou_pc * (1.0 - winner)
    excl    = non_win.sum() / (1.0 - winner).sum().clamp(min=1.0)

    return guide, excl


def _diversity_term(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Gram-matrix off-diagonal cosine similarity penalty across channels."""
    B, C, H, W = attn.shape
    a = attn.view(B, C, H * W).float()
    if a.shape[-1] > 1024:
        # subsample spatially — avoid randperm (CPU-GPU sync); strided slice is faster
        a = a[:, :, ::max(1, a.shape[-1] // 1024)]
    a    = F.normalize(a, dim=-1, eps=eps)
    gram = torch.bmm(a, a.transpose(1, 2))                            # (B,C,C)
    eye  = torch.eye(C, device=attn.device, dtype=gram.dtype)
    off  = gram * (1.0 - eye)
    return (off ** 2).sum(dim=[1, 2]).mean() / max(1.0, C * (C - 1))


# ── Public API ────────────────────────────────────────────────────────────────

def sf_loss(logits: torch.Tensor, attn: torch.Tensor,
            target: torch.Tensor, cfg: SFLossConfig):
    """Unified sf_seg training loss.

    Shared computations vs. calling individual functions separately:
        softmax(logits)          1×  instead of 2×  (saves 1 softmax at 224×224×151)
        cross_entropy (no_red)   1×  instead of 2×  (shared by seg focal + boundary)
        pool(attn, guide_size²)  1×  instead of 2×
        interpolate(target)      1×  instead of 2×
        one_hot(tgt_14)          1×  instead of 2×
        no-grad BMM(IoU matrix)  1×  shared for guide + excl winner selection

    Args:
        logits : (B, C, H, W)    raw model logits
        attn   : (B, K, H', W')  attention maps from head_large
        target : (B, H, W)       integer class labels in [0, C-1]
        cfg    : SFLossConfig

    Returns:
        total  : scalar loss (differentiable)
        parts  : dict {"seg","boundary","guide","excl","div"} — detached scalars
    """
    zero = logits.new_tensor(0.0)

    # ── Shared: softmax + per-pixel CE ────────────────────────────────────────
    probs  = F.softmax(logits.float(), dim=1)              # reused by seg IoU
    ce_raw = F.cross_entropy(logits.float(), target,
                             reduction="none")              # reused by seg focal + boundary

    # ── Shared: attn + target downsampled to guide_size × guide_size ──────────
    g      = cfg.guide_size
    attn_s = F.adaptive_avg_pool2d(attn.float(), (g, g))
    tgt_s  = F.interpolate(target.float().unsqueeze(1), (g, g),
                           mode="nearest").squeeze(1).long()
    L    = g * g
    B, K = attn_s.shape[:2]
    gt_f = (F.one_hot(tgt_s.clamp(0, cfg.num_classes - 1), cfg.num_classes)
              .permute(0, 3, 1, 2).float()
              .view(B, cfg.num_classes, L))
    atn_f = attn_s.view(B, K, L)

    # ── Shared: no-grad IoU matrix ─────────────────────────────────────────────
    with torch.no_grad():
        inter_ng = torch.bmm(gt_f, atn_f.transpose(1, 2))
        union_ng = (gt_f.sum(-1, keepdim=True)
                    + atn_f.sum(-1).unsqueeze(1) - inter_ng)
        iou_m_ng = ((inter_ng + 1e-3) / (union_ng + 1e-3)).clamp(0.0, 1.0)

    # ── Loss terms ─────────────────────────────────────────────────────────────
    s  = _seg_term(target, probs, ce_raw, cfg)

    bd = (cfg.boundary_weight * _boundary_term(ce_raw, target, cfg)
          if cfg.boundary_weight > 0 else zero)

    if cfg.guide_weight > 0 or cfg.excl_weight > 0:
        g_raw, e_raw = _attn_term(attn_s, gt_f, atn_f, iou_m_ng, cfg)
        g_ = cfg.guide_weight * g_raw
        e_ = cfg.excl_weight  * e_raw
    else:
        g_ = e_ = zero

    d  = (cfg.diversity_weight * _diversity_term(attn)
          if cfg.diversity_weight > 0 else zero)

    total = s + bd + g_ + e_ + d

    parts = {
        "seg":      s.detach(),
        "boundary": bd.detach(),
        "guide":    g_.detach(),
        "excl":     e_.detach(),
        "div":      d.detach(),
    }
    return total, parts


if __name__ == "__main__":
    import time

    cfg = SFLossConfig(num_classes=151, boundary_weight=0.3,
                       diversity_weight=0.1, guide_weight=0.3, excl_weight=0.2)
    B      = 4
    logits = torch.randn(B, 151, 224, 224)
    attn   = torch.rand(B, 128, 56, 56)
    target = torch.randint(0, 151, (B, 224, 224))

    for _ in range(2):
        loss, parts = sf_loss(logits, attn, target, cfg)

    t0 = time.perf_counter()
    N  = 10
    for _ in range(N):
        loss, parts = sf_loss(logits, attn, target, cfg)
    elapsed = (time.perf_counter() - t0) / N * 1000

    print(f"sf_loss total  : {loss.item():.4f}")
    for k, v in parts.items():
        print(f"  {k:<10}: {v.item():.4f}")
    print(f"\nAvg time/call  : {elapsed:.1f} ms  (B={B}, CPU)")
