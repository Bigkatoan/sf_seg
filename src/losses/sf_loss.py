"""sf_loss — unified training loss for sf_seg.

Replaces the individual attention_guide_loss / attention_exclusivity_loss /
diversity_loss / edge_corner_loss / focal_iou_loss calls in the trainer with
a single function that shares intermediate tensors across terms.

Shared tensors (computed once per step):
    probs       softmax(logits)          used by: seg IoU, spatial mass penalty
    attn_s      pool(attn, guide_size²)  used by: guide, excl
    gt_f        one_hot(target_14)       used by: guide, excl
    iou_m_ng    no-grad BMM(gt_f, atn_f) used by: guide winner selection + excl winner mask
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SFLossConfig:
    num_classes:     int   = 151
    # Term weights
    boundary_weight: float = 0.3
    diversity_weight: float = 0.1
    guide_weight:    float = 0.3
    excl_weight:     float = 0.2
    # Seg
    focal_gamma:     float = 2.0
    absent_weight:   float = 0.2
    seg_ce_w:        float = 0.5   # blend: focal_ce vs focal_iou
    seg_iou_w:       float = 0.5
    # Boundary
    edge_weight:     float = 4.0
    corner_weight:   float = 6.0
    boundary_dilate: int   = 2
    # Attention
    guide_size:      int   = 14

    @classmethod
    def from_args(cls, args) -> "SFLossConfig":
        """Build from trainer args namespace (maps config.json field names)."""
        return cls(
            num_classes=getattr(args, "num_classes", 151),
            boundary_weight=getattr(args, "boundary_weight", 0.3),
            diversity_weight=getattr(args, "diversity_weight", 0.1),
            guide_weight=getattr(args, "attn_guide_weight", 0.3),
            excl_weight=getattr(args, "attn_exclusive_weight", 0.2),
            focal_gamma=2.0,
            absent_weight=getattr(args, "absent_weight", 0.2),
            edge_weight=getattr(args, "edge_weight", 4.0),
            corner_weight=getattr(args, "corner_weight", 6.0),
        )


# ── Term implementations ──────────────────────────────────────────────────────

def _seg_term(logits: torch.Tensor, target: torch.Tensor,
              probs: torch.Tensor, cfg: SFLossConfig,
              eps: float = 1e-3) -> torch.Tensor:
    """Focal CE + focal soft-IoU + spatial mass penalty.

    Uses pre-computed `probs` (softmax of logits) so the caller's softmax is
    reused instead of being recomputed inside multiclass_iou_loss.
    """
    B, C = probs.shape[:2]

    # Focal CE — logits directly (more stable than log(softmax))
    ce     = F.cross_entropy(logits.float(), target, reduction="none")  # (B,H,W)
    pt     = torch.exp(-ce.clamp(max=100.0))
    f_ce   = ((1.0 - pt) ** cfg.focal_gamma * ce).mean()

    # Soft IoU from shared probs
    tgt_oh    = F.one_hot(target.clamp(0, C - 1), C).permute(0, 3, 1, 2).float()
    pred_flat = probs.view(B, C, -1)
    tgt_flat  = tgt_oh.view(B, C, -1)

    inter   = (pred_flat * tgt_flat).sum(-1)                          # (B,C)
    union   = pred_flat.sum(-1) + tgt_flat.sum(-1) - inter
    iou     = ((inter + eps) / (union + eps).clamp(min=eps)).clamp(0.0, 1.0)

    present = (tgt_flat.sum(-1) > 0).float()                         # (B,C)
    absent  = 1.0 - present

    # Focal IoU for present classes: (1−IoU)^(γ+1)
    f_iou     = ((1.0 - iou).pow(1.0 + cfg.focal_gamma) * present).sum()
    f_iou     = f_iou / present.sum().clamp(min=1.0)

    # Spatial mass penalty for absent classes
    mass      = (pred_flat.mean(-1) * absent).sum() / absent.sum().clamp(min=1.0)

    return cfg.seg_ce_w * f_ce + cfg.seg_iou_w * f_iou + cfg.absent_weight * mass


def _boundary_term(logits: torch.Tensor, target: torch.Tensor,
                   cfg: SFLossConfig) -> torch.Tensor:
    """Focal CE weighted by edge + corner importance map."""
    with torch.no_grad():
        t   = target.float().unsqueeze(1)

        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                           dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
        edge = (F.conv2d(t, lap, padding=1).abs() > 0).float()

        kx   = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                             dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
        ky   = kx.transpose(2, 3).contiguous()
        c    = F.conv2d(t, kx, padding=1).abs() * F.conv2d(t, ky, padding=1).abs()
        corner = c / c.amax(dim=[2, 3], keepdim=True).clamp(min=1e-8)

        if cfg.boundary_dilate > 0:
            k      = 2 * cfg.boundary_dilate + 1
            edge   = F.max_pool2d(edge,   k, stride=1, padding=cfg.boundary_dilate)
            corner = F.max_pool2d(corner, k, stride=1, padding=cfg.boundary_dilate)

        w = (1.0
             + cfg.edge_weight   * edge.squeeze(1)
             + cfg.corner_weight * corner.squeeze(1)).clamp(max=20.0)  # (B,H,W)

    ce    = F.cross_entropy(logits.float(), target, reduction="none")
    ce    = torch.nan_to_num(ce, nan=0.0, posinf=100.0, neginf=0.0)
    pt    = torch.exp(-ce.clamp(max=100.0))
    focal = (1.0 - pt).pow(cfg.focal_gamma) * ce

    return torch.nan_to_num((focal * w).sum() / w.sum().clamp(min=1.0), nan=0.0)


def _attn_term(attn_s: torch.Tensor, gt_f: torch.Tensor,
               atn_f: torch.Tensor, iou_m_ng: torch.Tensor,
               cfg: SFLossConfig, eps: float = 1e-3):
    """Guide + exclusivity losses sharing the pre-computed no-grad IoU matrix.

    Args:
        attn_s   : (B, K, g, g)      pooled attention  (float32)
        gt_f     : (B, C, L)          GT one-hot flat   (float32)
        atn_f    : (B, K, L)          attention flat    (float32)
        iou_m_ng : (B, C, K) no-grad  IoU(class, channel)
    Returns:
        guide_loss, excl_loss — both scalar tensors
    """
    B, num_classes, L = gt_f.shape
    B, K, _           = atn_f.shape

    present = (gt_f.sum(-1) > 0)   # (B, C)
    present[:, 0] = False           # skip background

    if not present.any():
        zero = atn_f.new_tensor(0.0)
        return zero, zero

    # ── Guide: winner channel → Dice with its GT class mask ──────────────────
    # topk from no-grad IoU — winner selection carries no gradient
    _, topk_idx = iou_m_ng.topk(1, dim=2)                              # (B,C,1)
    idx         = topk_idx.unsqueeze(-1).expand(-1, -1, -1, L)
    atn_exp     = atn_f.unsqueeze(1).expand(-1, num_classes, -1, -1)   # (B,C,K,L)
    w_maps      = torch.gather(atn_exp, 2, idx).squeeze(2)             # (B,C,L)
    mx          = w_maps.amax(-1, keepdim=True).clamp(min=1e-6)
    w_norm      = (w_maps / mx).clamp(0.0, 1.0)

    p    = w_norm[present]                                              # (N,L)
    t    = gt_f[present]                                                # (N,L)
    dice = 1.0 - (2.0*(p*t).sum(-1) + eps) / (p.sum(-1) + t.sum(-1) + eps)
    guide = dice.mean()

    # ── Excl: non-winner channels must not overlap other classes ─────────────
    # Recompute IoU WITH gradient (needed for excl gradient flow)
    inter_g = torch.bmm(gt_f, atn_f.transpose(1, 2))                  # (B,C,K)
    union_g = gt_f.sum(-1, keepdim=True) + atn_f.sum(-1).unsqueeze(1) - inter_g
    iou_g   = ((inter_g + eps) / (union_g + eps)).clamp(0.0, 1.0)
    iou_g   = iou_g * present.float().unsqueeze(-1)
    iou_pc  = iou_g.permute(0, 2, 1)                                   # (B,K,C)

    # Winner mask from no-grad IoU — consistent with guide assignment
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
    a = attn.view(B, C, H * W)
    if a.shape[-1] > 2048:
        idx = torch.randperm(a.shape[-1], device=a.device)[:2048]
        a   = a[:, :, idx]
    a    = F.normalize(a.float(), dim=-1, eps=eps)
    gram = torch.bmm(a, a.transpose(1, 2))                            # (B,C,C)
    eye  = torch.eye(C, device=attn.device, dtype=gram.dtype)
    off  = gram * (1.0 - eye)
    return (off ** 2).sum(dim=[1, 2]).mean() / max(1.0, C * (C - 1))


# ── Public API ────────────────────────────────────────────────────────────────

def sf_loss(logits: torch.Tensor, attn: torch.Tensor,
            target: torch.Tensor, cfg: SFLossConfig):
    """Unified sf_seg training loss.

    Shared computation per step vs. calling individual functions:
        softmax(logits)          1×  instead of 2×  (saves 1 softmax at 224×224)
        pool(attn, guide_size²)  1×  instead of 2×
        interpolate(target)      1×  instead of 2×
        one_hot(tgt_14)          1×  instead of 2×
        no-grad BMM(IoU matrix)  1×  (shared for guide + excl winner selection)

    Args:
        logits : (B, C, H, W)    raw model logits
        attn   : (B, K, H', W')  attention maps from head_large
        target : (B, H, W)       integer class labels in [0, C-1]
        cfg    : SFLossConfig

    Returns:
        total  : scalar loss (differentiable)
        parts  : dict {"seg","boundary","guide","excl","div"} — detached scalars for logging
    """
    zero = logits.new_tensor(0.0)

    # ── Shared: softmax ────────────────────────────────────────────────────────
    probs = F.softmax(logits.float(), dim=1)   # (B,C,H,W) — reused by seg + spatial mass

    # ── Shared: attn + target downsampled to guide_size × guide_size ──────────
    g      = cfg.guide_size
    attn_s = F.adaptive_avg_pool2d(attn.float(), (g, g))          # (B,K,g,g)
    tgt_s  = F.interpolate(target.float().unsqueeze(1), (g, g),
                           mode="nearest").squeeze(1).long()       # (B,g,g)
    L    = g * g
    B, K = attn_s.shape[:2]
    gt_f = (F.one_hot(tgt_s.clamp(0, cfg.num_classes - 1), cfg.num_classes)
              .permute(0, 3, 1, 2).float()
              .view(B, cfg.num_classes, L))                        # (B,C,L)
    atn_f = attn_s.view(B, K, L)                                  # (B,K,L)

    # ── Shared: no-grad IoU matrix ─────────────────────────────────────────────
    with torch.no_grad():
        inter_ng = torch.bmm(gt_f, atn_f.transpose(1, 2))         # (B,C,K)
        union_ng = (gt_f.sum(-1, keepdim=True)
                    + atn_f.sum(-1).unsqueeze(1) - inter_ng)
        iou_m_ng = ((inter_ng + 1e-3) / (union_ng + 1e-3)).clamp(0.0, 1.0)

    # ── Loss terms ─────────────────────────────────────────────────────────────
    s  = _seg_term(logits, target, probs, cfg)

    bd = (cfg.boundary_weight * _boundary_term(logits, target, cfg)
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
    B   = 4
    logits = torch.randn(B, 151, 224, 224)
    attn   = torch.rand(B, 128, 56, 56)
    target = torch.randint(0, 151, (B, 224, 224))

    # Warmup
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
