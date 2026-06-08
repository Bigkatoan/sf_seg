"""sf_loss — unified training loss for sf_seg.

Shared tensors (computed once per step):
    ce_raw      CE per pixel (no_reduce) used by: seg focal CE
    attn_s      pool(attn, guide_size²)  used by: guide, excl
    gt_f        one_hot(target_14)       used by: guide, excl
    iou_m_ng    no-grad BMM(gt_f, atn_f) used by: guide winner + excl winner mask

Note: probs (softmax) is NOT computed at full resolution — _seg_term downsamples
logits to 56×56 before softmax, reducing peak memory from ~2 GB to ~120 MB
(B=32, C=151). Focal CE still uses full-resolution ce_raw.

Note: boundary/edge term removed — it showed near-zero contribution (bd≈0)
while adding instability. Attention guide+excl losses already supervise
the model to produce spatially precise, class-specific maps.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class SFLossConfig:
    num_classes:      int   = 151
    # Term weights
    diversity_weight: float = 0.1
    guide_weight:     float = 0.3
    excl_weight:      float = 0.2
    # Seg — matches focal_iou_loss defaults exactly
    focal_gamma:      float = 2.0
    focal_w:          float = 1.0
    iou_w:            float = 0.5
    no_obj_weight:    float = 0.1
    # Attention
    guide_size:       int   = 14

    @classmethod
    def from_args(cls, args) -> "SFLossConfig":
        return cls(
            num_classes=getattr(args, "num_classes", 151),
            diversity_weight=getattr(args, "diversity_weight", 0.1),
            guide_weight=getattr(args, "attn_guide_weight", 0.3),
            excl_weight=getattr(args, "attn_exclusive_weight", 0.2),
            focal_gamma=2.0,
            iou_w=getattr(args, "iou_w", 0.5),
            no_obj_weight=getattr(args, "no_obj_weight", 0.1),
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe(t: torch.Tensor) -> torch.Tensor:
    """Zero out any NaN / ±Inf in a tensor — last-resort guard on term outputs."""
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


# ── Term implementations ──────────────────────────────────────────────────────

def _seg_term(target: torch.Tensor, logits: torch.Tensor,
              ce_raw: torch.Tensor, cfg: SFLossConfig,
              iou_size: int = 56, eps: float = 1e-3) -> torch.Tensor:
    """focal_w * focal_CE  +  iou_w * soft_IoU(no_obj_weight).

    Focal CE uses full-resolution ce_raw (cheap, BxHxW).
    Soft IoU downsamples logits to iou_size before softmax — avoids allocating
    a (B, C, H, W) fp32 probs tensor which is ~2 GB for B=32, C=151, H=W=224.
    """
    B, C, H, W = logits.shape

    # Focal CE — full resolution (ce_raw is BxHxW, cheap)
    ce_c  = ce_raw.clamp(max=100.0)
    pt    = torch.exp(-ce_c)
    f_ce  = _safe(((1.0 - pt) ** cfg.focal_gamma * ce_c).mean())

    # Soft IoU — downsampled to iou_size×iou_size (16× smaller memory for 224→56)
    # pool logits first then softmax avoids large fp32 intermediate
    s = min(iou_size, H, W)
    if H != s or W != s:
        logits_s = F.adaptive_avg_pool2d(logits.float(), (s, s))
        tgt_s    = F.interpolate(target.float().unsqueeze(1), (s, s),
                                 mode="nearest").squeeze(1).long().clamp(0, C - 1)
    else:
        logits_s = logits.float()
        tgt_s    = target.clamp(0, C - 1)

    probs_s   = F.softmax(logits_s, dim=1)
    tgt_oh    = F.one_hot(tgt_s, C).permute(0, 3, 1, 2).float()
    pred_flat = probs_s.view(B, C, -1)
    tgt_flat  = tgt_oh.view(B, C, -1)

    inter   = (pred_flat * tgt_flat).sum(-1)
    union   = pred_flat.sum(-1) + tgt_flat.sum(-1) - inter
    iou     = ((inter + eps) / (union + eps).clamp(min=eps)).clamp(0., 1.)

    present = (tgt_flat.sum(-1) > 0).float()
    weight  = present + cfg.no_obj_weight * (1.0 - present)
    iou_l   = _safe(((1.0 - iou) * weight).sum() / weight.sum().clamp(min=eps))

    return _safe(cfg.focal_w * f_ce + cfg.iou_w * iou_l)


def _attn_term(attn_s: torch.Tensor, gt_f: torch.Tensor,
               atn_f: torch.Tensor, iou_m_ng: torch.Tensor,
               cfg: SFLossConfig, eps: float = 1e-3):
    """Guide + exclusivity losses sharing the pre-computed no-grad IoU matrix.

    All BMM ops use explicit .float() to stay out of the autocast float16 path
    (torch.bmm is in the autocast eligible list and would otherwise run fp16).
    """
    B, num_classes, L = gt_f.shape
    B, K, _           = atn_f.shape

    present = (gt_f.sum(-1) > 0)
    present[:, 0] = False   # skip background class

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

    p     = w_norm[present]
    t     = gt_f[present]
    denom = (p.sum(-1) + t.sum(-1) + eps).clamp(min=eps)
    dice  = 1.0 - (2.0 * (p * t).sum(-1) + eps) / denom
    guide = _safe(dice.mean())

    # ── Excl: non-winner channels must not overlap other classes ─────────────
    gt_f_f32  = gt_f.float()
    atn_f_f32 = atn_f.float()

    inter_g = torch.bmm(gt_f_f32, atn_f_f32.transpose(1, 2))          # (B,C,K) f32
    union_g = (gt_f_f32.sum(-1, keepdim=True)
               + atn_f_f32.sum(-1).unsqueeze(1) - inter_g)
    iou_g   = ((inter_g + eps) / (union_g + eps).clamp(min=eps)).clamp(0.0, 1.0)
    iou_g   = iou_g * present.float().unsqueeze(-1)
    iou_pc  = iou_g.permute(0, 2, 1)                                   # (B,K,C)

    with torch.no_grad():
        iou_pc_ng = iou_m_ng.permute(0, 2, 1) * present.float().unsqueeze(-2)
        best      = iou_pc_ng.max(dim=-1, keepdim=True).values
        winner    = (iou_pc_ng == best).float()

    non_win = iou_pc * (1.0 - winner)
    excl    = _safe(non_win.sum() / (1.0 - winner).sum().clamp(min=1.0))

    return guide, excl


def _diversity_term(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Gram-matrix off-diagonal cosine similarity penalty across channels."""
    B, C, H, W = attn.shape
    a = attn.view(B, C, H * W).float()
    if a.shape[-1] > 1024:
        a = a[:, :, ::max(1, a.shape[-1] // 1024)]
    a    = F.normalize(a, dim=-1, eps=eps)
    gram = torch.bmm(a.float(), a.float().transpose(1, 2))             # explicit f32
    eye  = torch.eye(C, device=attn.device, dtype=gram.dtype)
    off  = gram * (1.0 - eye)
    return _safe((off ** 2).sum(dim=[1, 2]).mean() / max(1.0, C * (C - 1)))


# ── Public API ────────────────────────────────────────────────────────────────

def sf_loss(logits: torch.Tensor, attn: torch.Tensor,
            target: torch.Tensor, cfg: SFLossConfig):
    """Unified sf_seg training loss.

    Terms: seg (focal_iou) + guide (Dice) + excl (IoU suppression) + div (Gram)
    Boundary/edge term removed — showed near-zero contribution with instability.

    Args:
        logits : (B, C, H, W)    raw model logits (fp16 or fp32)
        attn   : (B, K, H', W')  attention maps from head_large
        target : (B, H, W)       integer class labels — values outside [0,C-1]
                                  are clamped (handles ADE20K ignore pixels 255)
        cfg    : SFLossConfig

    Returns:
        total  : scalar loss (differentiable)
        parts  : dict {"seg","guide","excl","div"} — detached scalars for logging
    """
    C    = cfg.num_classes
    zero = logits.new_tensor(0.0)

    # Safety: clamp target before ALL indexing/CE — ADE20K has ignore pixels (255)
    tgt = target.clamp(0, C - 1)

    # ── Shared: per-pixel CE (full resolution, always float32) ───────────────
    # probs NOT computed here — _seg_term computes downsampled probs for IoU
    # to avoid allocating a (B,C,H,W) fp32 tensor (~2 GB for B=32,C=151,HW=224)
    ce_raw = F.cross_entropy(logits.float(), tgt, reduction="none")

    # ── Shared: attn + target downsampled to guide_size × guide_size ──────────
    g      = cfg.guide_size
    attn_s = F.adaptive_avg_pool2d(attn.float(), (g, g))
    tgt_s  = F.interpolate(tgt.float().unsqueeze(1), (g, g),
                           mode="nearest").squeeze(1).long()
    L    = g * g
    B, K = attn_s.shape[:2]
    gt_f = (F.one_hot(tgt_s, C)
              .permute(0, 3, 1, 2).float()
              .view(B, C, L))
    atn_f = attn_s.view(B, K, L)

    # ── Shared: no-grad IoU matrix (explicit float32 — bmm is autocast-eligible)
    with torch.no_grad():
        gt_f32  = gt_f.float()
        atn_f32 = atn_f.float()
        inter_ng = torch.bmm(gt_f32, atn_f32.transpose(1, 2))
        union_ng = (gt_f32.sum(-1, keepdim=True)
                    + atn_f32.sum(-1).unsqueeze(1) - inter_ng)
        iou_m_ng = ((inter_ng + 1e-3)
                    / (union_ng + 1e-3).clamp(min=1e-3)).clamp(0.0, 1.0)

    # ── Loss terms ─────────────────────────────────────────────────────────────
    s = _seg_term(tgt, logits, ce_raw, cfg)

    if cfg.guide_weight > 0 or cfg.excl_weight > 0:
        g_raw, e_raw = _attn_term(attn_s, gt_f, atn_f, iou_m_ng, cfg)
        g_ = cfg.guide_weight * g_raw
        e_ = cfg.excl_weight  * e_raw
    else:
        g_ = e_ = zero

    d = (cfg.diversity_weight * _diversity_term(attn)
         if cfg.diversity_weight > 0 else zero)

    total = s + g_ + e_ + d

    parts = {
        "seg":   s.detach(),
        "guide": g_.detach(),
        "excl":  e_.detach(),
        "div":   d.detach(),
    }
    return total, parts


if __name__ == "__main__":
    import time

    cfg = SFLossConfig(num_classes=151, diversity_weight=0.1,
                       guide_weight=0.3, excl_weight=0.2)
    B      = 4
    logits = torch.randn(B, 151, 224, 224)
    attn   = torch.rand(B, 128, 56, 56)
    target = torch.randint(0, 151, (B, 224, 224))
    target[0, 0, 0] = 255   # ADE20K ignore pixel — should be handled safely

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
    print(f"\nAll finite: {all(torch.isfinite(v) for v in parts.values())}")
    print(f"Avg time/call  : {elapsed:.1f} ms  (B={B}, CPU)")
