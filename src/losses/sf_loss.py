"""sf_loss — unified training loss for sf_seg.

All primary loss terms are computed at full image resolution (512×512):
    ce_raw / focal CE  — per-pixel cross-entropy,  (B, H, W)  — always full res
    soft IoU           — per-class gather trick,    no (B,C,H,W) one_hot needed
    Sobel edge         — logsumexp confidence proxy,(B, 1, H, W) — full res

Attention-only terms (guide, excl, div) operate at their natural resolutions:
    attn_s / gt_f  — pooled to guide_size×guide_size (14×14)
    diversity      — on head_large attn map (H/4)

IoU peak memory with iou_downsample=4: (B, C, H/4, W/4) bfloat16 ≈ 78 MB for
B=16, C=151, 512². Full-res (iou_downsample=1) would be 1.25 GB.
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
    edge_weight:      float = 0.3
    # Seg — matches focal_iou_loss defaults exactly
    focal_gamma:      float = 2.0
    focal_w:          float = 1.0
    iou_w:            float = 0.5
    iou_downsample:   int   = 4     # spatial downsample for IoU: 4 = H/4 (16× faster)
    iou_form:         str   = 'linear'  # 'linear' = 1-IoU [1→0] | 'log' = -ln(IoU) [∞→0]
    no_obj_weight:    float = 0.1
    # Boundary — upweight pixels at class boundaries in focal CE
    boundary_weight:  float = 3.0   # 1.0 = disabled
    # Attention
    guide_size:       int   = 14

    @classmethod
    def from_args(cls, args) -> "SFLossConfig":
        return cls(
            num_classes=getattr(args, "num_classes", 151),
            diversity_weight=getattr(args, "diversity_weight", 0.1),
            guide_weight=getattr(args, "attn_guide_weight", 0.3),
            excl_weight=getattr(args, "attn_exclusive_weight", 0.2),
            edge_weight=getattr(args, "edge_weight", 0.3),
            focal_gamma=2.0,
            focal_w=getattr(args, "focal_w", 1.0),
            iou_w=getattr(args, "iou_w", 0.5),
            iou_downsample=getattr(args, "iou_downsample", 4),
            iou_form=getattr(args, "iou_form", "linear"),
            no_obj_weight=getattr(args, "no_obj_weight", 0.1),
            boundary_weight=getattr(args, "boundary_weight", 3.0),
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _safe(t: torch.Tensor) -> torch.Tensor:
    """Zero out any NaN / ±Inf in a tensor — last-resort guard on term outputs."""
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def _boundary_mask(tgt: torch.Tensor) -> torch.Tensor:
    """Morphological gradient on integer label map → binary boundary pixel mask.
    tgt: (B, H, W) int64.  Returns (B, H, W) float32, 1.0 at boundaries.
    Dilation − erosion > 0 ↔ the 3×3 neighbourhood contains ≥ 2 distinct classes.
    """
    t   = tgt.float().unsqueeze(1)
    dil = F.max_pool2d(t, kernel_size=3, stride=1, padding=1)
    ero = -F.max_pool2d(-t, kernel_size=3, stride=1, padding=1)
    return (dil - ero > 0.5).squeeze(1).float()


# ── Term implementations ──────────────────────────────────────────────────────

def _seg_term(target: torch.Tensor, logits: torch.Tensor,
              ce_raw: torch.Tensor, cfg: SFLossConfig,
              valid: torch.Tensor,
              eps: float = 1e-3) -> torch.Tensor:
    """focal_w * focal_CE (sqrt-frequency per-class normalized)  +  iou_w * soft_IoU.

    CE is averaged within each present class then sqrt(n_c)-weighted across
    classes — rare classes get boosted ~1/sqrt(freq) without fully muting
    dominant ones.

    target  : tgt_safe — already clamped to [0, C-1], 255 replaced with 0 for indexing
    ce_raw  : 0.0 at ignore pixels (computed with ignore_index=255)
    valid   : (B,H,W) bool — True where target != 255
    """
    B, C, H, W = logits.shape

    # Focal CE — bỏ qua hoàn toàn khi focal_w=0 (IoU-only) để khỏi tính thừa
    if cfg.focal_w > 0:
        ce_c  = ce_raw.clamp(max=100.0)
        pt    = torch.exp(-ce_c)
        focal = (1.0 - pt) ** cfg.focal_gamma * ce_c

        if cfg.boundary_weight > 1.0:
            bw    = 1.0 + (cfg.boundary_weight - 1.0) * _boundary_mask(target)
            focal = focal * bw

        # Per-class CE normalization (sqrt-frequency): average focal loss within
        # each class, then weighted-average across classes with weight sqrt(n_c).
        tgt_flat   = target.view(-1)                                   # (B*H*W)
        focal_flat = focal.view(-1)
        valid_flat = valid.float().view(-1)
        cls_sum    = torch.zeros(C, device=logits.device).scatter_add(0, tgt_flat, focal_flat * valid_flat)
        cls_cnt    = torch.zeros(C, device=logits.device).scatter_add(0, tgt_flat, valid_flat)
        cls_w      = cls_cnt.sqrt()                                    # 0 for absent classes
        cls_mean   = cls_sum / cls_cnt.clamp(min=1.0)
        f_ce       = _safe((cls_mean * cls_w).sum() / cls_w.sum().clamp(min=1.0))
    else:
        f_ce = logits.new_tensor(0.0)

    if cfg.iou_w <= 0:
        return _safe(cfg.focal_w * f_ce)

    # Optionally downsample logits+target for IoU (class signal preserved)
    ds = max(1, cfg.iou_downsample)
    if ds > 1:
        iou_logits = F.avg_pool2d(logits.float(), kernel_size=ds, stride=ds)
        iou_target = F.interpolate(
            target.unsqueeze(1).float(), (H // ds, W // ds), mode='nearest'
        ).squeeze(1).long().clamp(0, C - 1)
        valid_iou  = F.interpolate(
            valid.float().unsqueeze(1), (H // ds, W // ds), mode='nearest'
        ).squeeze(1)
    else:
        iou_logits = logits
        iou_target = target
        valid_iou  = valid.float()

    Bi, Ci, Hi, Wi = iou_logits.shape

    # Soft IoU — bfloat16 probs (fp32 exponent range, no overflow on spatial sums)
    probs      = F.softmax(iou_logits.bfloat16(), dim=1)              # (B, C, Hi, Wi)
    tgt_flat   = iou_target.view(Bi, -1)                              # (B, N)
    valid_flat = valid_iou.view(Bi, -1)                               # (B, N) float

    pred_sum = probs.sum(dim=[2, 3]).float()                          # (B, C)

    inter_diag = (probs.view(Bi, Ci, -1)
                  .gather(1, tgt_flat.unsqueeze(1))
                  .squeeze(1))                                        # (B, N) bf16
    # Weight by valid_flat: ignore pixels do not contribute to intersection
    inter = (torch.zeros(Bi, Ci, device=logits.device, dtype=torch.float32)
             .scatter_add(1, tgt_flat, inter_diag.float() * valid_flat))  # (B, C)

    with torch.no_grad():
        # gt_sum: count only valid pixels per class
        gt_sum = (torch.zeros(Bi, Ci, device=logits.device, dtype=torch.float32)
                  .scatter_add(1, tgt_flat, valid_flat))              # (B, C)

    iou     = ((inter + eps)
               / (pred_sum + gt_sum - inter + eps).clamp(min=eps)).clamp(0., 1.)
    present = (gt_sum > 0).float()
    weight  = present + cfg.no_obj_weight * (1.0 - present)
    # Dạng loss: 'linear' = (1-IoU) ∈[0,1] (gradient bị chặn); 'log' = -ln(IoU)
    # ∈[0,∞) — gradient -1/IoU rất mạnh khi IoU thấp (init) → hội tụ nhanh hơn.
    if cfg.iou_form == 'log':
        iou_term = -torch.log(iou.clamp(min=1e-3))   # ∞→0, clamp tránh nổ tại IoU=0
    else:
        iou_term = 1.0 - iou
    # Chuẩn hóa theo SỐ CLASS TỒN TẠI, KHÔNG phải tổng weight: class vắng (140
    # cái × no_obj 0.1 ≈ 58% tổng weight) sẽ pha loãng tín hiệu class tồn tại
    # nếu nằm trong mẫu số. Giờ: mean sạch trên class tồn tại + no-obj penalty.
    iou_l   = _safe((iou_term * weight).sum() / present.sum().clamp(min=1.0))

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

    present = (gt_f.sum(-1) > 0)   # all 150 classes are real (label 0 = wall)

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


# ── Sobel kernels (cached per device) ─────────────────────────────────────────

def _sobel_kernels(device: torch.device):
    """Return (kx, ky) Sobel kernels as (1,1,3,3) float32 tensors."""
    kx = torch.tensor([[1., 0., -1.],
                        [2., 0., -2.],
                        [1., 0., -1.]], device=device, dtype=torch.float32).view(1, 1, 3, 3) / 8.
    return kx, kx.transpose(2, 3).contiguous()


def _sobel_mag(x: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor,
               eps: float = 1e-6) -> torch.Tensor:
    """Edge magnitude map: sqrt(Gx² + Gy²) for (B,1,H,W) input."""
    return (F.conv2d(x, kx, padding=1) ** 2
            + F.conv2d(x, ky, padding=1) ** 2
            + eps).sqrt()


def _sobel_term(logits_f32: torch.Tensor, target: torch.Tensor,
                cfg: SFLossConfig,
                eps: float = 1e-6) -> torch.Tensor:
    """Sobel gradient-matching loss at full resolution.

    Prediction proxy — max(softmax) via the logsumexp identity (no (B,C,H,W) tensor):
        conf = exp( max_c logits_c  −  logsumexp_c logits_c )   → (B, 1, H, W)

    logits_f32 must already be float32 (callers pass the shared fp32 cast to
    avoid creating a second (B,C,H,W) fp32 tensor alongside the one used for CE).

    Loss is normalised by GT edge energy so raw value ≈ 1.0 at initialisation.
    """
    B, C, H, W = logits_f32.shape

    # ── Prediction: max(softmax) without materialising (B,C,H,W) ───────────────
    conf = torch.exp(logits_f32.max(dim=1, keepdim=True).values
                     - torch.logsumexp(logits_f32, dim=1, keepdim=True))    # (B,1,H,W)

    # ── GT: normalise class index → [0,1] ──────────────────────────────────────
    tgt_norm = target.float().unsqueeze(1) / max(C - 1, 1)                  # (B,1,H,W)

    # ── Sobel edge magnitude maps ───────────────────────────────────────────────
    kx, ky = _sobel_kernels(logits_f32.device)
    edge_pred = _sobel_mag(conf,     kx, ky, eps)
    with torch.no_grad():
        edge_gt   = _sobel_mag(tgt_norm, kx, ky, eps)
        weight    = 1.0 + edge_gt
        # GT edge energy — normalises the loss so raw ≈ 1.0 when edge_pred ≈ 0
        gt_energy = (edge_gt.pow(2) * weight).mean().clamp(min=1e-4)

    diff = edge_pred - edge_gt
    return _safe((diff.pow(2) * weight).mean() / gt_energy)


# ── Public API ────────────────────────────────────────────────────────────────

def sf_loss(logits: torch.Tensor, attn: torch.Tensor,
            target: torch.Tensor, cfg: SFLossConfig):
    """Unified sf_seg training loss.

    Terms:
      seg  — focal CE + soft IoU (main segmentation objective)
      guide — Dice between top-k attention channels and GT masks
      excl  — non-winner attention channels must not overlap other classes
      div   — Gram-matrix cosine penalty to diversify attention channels
      edge  — Sobel gradient matching: aligns predicted and GT edge maps
               so the model correctly predicts class boundaries and corners

    Args:
        logits : (B, C, H, W)    raw model logits (fp16 or fp32)
        attn   : (B, K, H', W')  attention maps from head_large
        target : (B, H, W)       integer class labels — values outside [0,C-1]
                                  are clamped (handles ADE20K ignore pixels 255)
        cfg    : SFLossConfig

    Returns:
        total  : scalar loss (differentiable)
        parts  : dict {"seg","guide","excl","div","edge"} — detached scalars for logging
    """
    C    = cfg.num_classes
    zero = logits.new_tensor(0.0)

    # Ignore pixels: ADE20K uses 255 for "no region" — must not be trained on.
    valid    = (target != 255)                  # (B, H, W) bool
    tgt_safe = target.clamp(0, C - 1)          # safe for indexing; 255→0 at ignore positions

    # ── Shared fp32 logits — created ONCE, reused by CE and edge terms ────────
    logits_f32 = logits.float()

    # ── Shared: per-pixel CE — ce_raw=0 at ignore pixels (ignore_index=255) ──
    ce_raw = F.cross_entropy(logits_f32, target.long(), ignore_index=255, reduction="none")

    # ── Loss terms ─────────────────────────────────────────────────────────────
    s = _seg_term(tgt_safe, logits, ce_raw, cfg, valid)

    # Guide/excl: only compute expensive attn↔GT precompute when needed
    if cfg.guide_weight > 0 or cfg.excl_weight > 0:
        g      = cfg.guide_size
        attn_s = F.adaptive_avg_pool2d(attn.float(), (g, g))
        tgt_s  = F.interpolate(tgt_safe.float().unsqueeze(1), (g, g),
                               mode="nearest").squeeze(1).long()
        L    = g * g
        B, K = attn_s.shape[:2]
        gt_f  = (F.one_hot(tgt_s, C)
                   .permute(0, 3, 1, 2).float()
                   .view(B, C, L))
        atn_f = attn_s.view(B, K, L)
        with torch.no_grad():
            gt_f32   = gt_f.float()
            atn_f32  = atn_f.float()
            inter_ng = torch.bmm(gt_f32, atn_f32.transpose(1, 2))
            union_ng = (gt_f32.sum(-1, keepdim=True)
                        + atn_f32.sum(-1).unsqueeze(1) - inter_ng)
            iou_m_ng = ((inter_ng + 1e-3)
                        / (union_ng + 1e-3).clamp(min=1e-3)).clamp(0.0, 1.0)
        g_raw, e_raw = _attn_term(attn_s, gt_f, atn_f, iou_m_ng, cfg)
        g_ = cfg.guide_weight * g_raw
        e_ = cfg.excl_weight  * e_raw
    else:
        g_ = e_ = zero

    d = (cfg.diversity_weight * _diversity_term(attn)
         if cfg.diversity_weight > 0 else zero)

    ed = (cfg.edge_weight * _sobel_term(logits_f32, tgt_safe, cfg)
          if cfg.edge_weight > 0 else zero)

    total = s + g_ + e_ + d + ed

    parts = {
        "seg":   s.detach(),
        "guide": g_.detach(),
        "excl":  e_.detach(),
        "div":   d.detach(),
        "edge":  ed.detach(),
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
