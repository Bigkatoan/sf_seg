"""SF-Seg V2 — Custom ConvNeXt-style backbone thay ResNet-18.

So với V1 (sf_seg_r18.py):
  - Backbone tự thiết kế (SFBackbone), không dùng pretrained ImageNet
  - Channel pyramid [32, 64, 128, 256] với num_channels=32 (V1: [64,128,256,512])
  - Không cần adapters — backbone output channels khớp trực tiếp với head sizes
  - Decoder nhẹ hơn: single 3×3 conv mỗi tầng (V1: 2-conv fuse blocks)
  - LayerScale trong mỗi ConvNeXtBlock → stable training từ đầu

Variants (backbone depth [s1, s2, s3, s4]):
  'nano'  : [2, 3, 6, 2] → ~2.9M total
  'micro' : [2, 3, 9, 2] → ~3.4M total  ← default (ngang SegFormer-B0 3.7M)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.models.common import _gn
from src.models.sf_seg_r18 import SparseAttnHead, AttentionHead


# ── Backbone blocks ───────────────────────────────────────────────────────────

class ConvNeXtBlock(nn.Module):
    """
    x → DWConv(7×7) → GN → pw_expand(4×) → GELU → pw_project → LayerScale → + x
    """
    def __init__(self, C: int, expand: int = 4, kernel: int = 7,
                 layer_scale: float = 1e-6):
        super().__init__()
        mid = C * expand
        self.dw    = nn.Conv2d(C, C, kernel, padding=kernel // 2, groups=C, bias=False)
        self.norm  = _gn(C)
        self.pw1   = nn.Conv2d(C, mid, 1)
        self.act   = nn.GELU()
        self.pw2   = nn.Conv2d(mid, C, 1)
        self.gamma = nn.Parameter(torch.full((C, 1, 1), layer_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))


class DownsampleBlock(nn.Sequential):
    """GN + Conv2d(stride=2) — 2× spatial downsample với channel projection."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__(
            _gn(in_c),
            nn.Conv2d(in_c, out_c, 2, stride=2, bias=False),
        )


# ── Backbone ──────────────────────────────────────────────────────────────────

class SFBackbone(nn.Module):
    """
    4-stage ConvNeXt-style backbone thiết kế cho multi-scale dense prediction.

    Returns (f_detail, f1, f2, f3, f4):
      f_detail : (B,  32, H/2,  W/2)  — H/2 detail path cho sharp boundaries
      f1       : (B,  C1, H/4,  W/4)  → head_large
      f2       : (B,  C2, H/8,  W/8)  → head_medium
      f3       : (B,  C3, H/16, W/16) → head_small
      f4       : (B,  C4, H/32, W/32) → head_tiny
    """
    DEPTHS = {
        'nano':  [2, 3, 6, 2],
        'micro': [2, 3, 9, 2],
    }

    def __init__(self, channels: tuple[int, int, int, int], variant: str = 'micro',
                 dw_kernel: int = 3):
        super().__init__()
        C1, C2, C3, C4 = channels
        d1, d2, d3, d4  = self.DEPTHS[variant]

        # Stem: RGB → 32ch at H/2  (preserved for detail branch)
        self.stem   = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            _gn(32),
            nn.GELU(),
        )

        # Stage 1: H/2 → H/4,  32 → C1
        self.down1  = DownsampleBlock(32, C1)
        self.stage1 = nn.Sequential(*[ConvNeXtBlock(C1, kernel=dw_kernel) for _ in range(d1)])

        # Stage 2: H/4 → H/8,  C1 → C2
        self.down2  = DownsampleBlock(C1, C2)
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(C2, kernel=dw_kernel) for _ in range(d2)])

        # Stage 3: H/8 → H/16, C2 → C3
        self.down3  = DownsampleBlock(C2, C3)
        self.stage3 = nn.Sequential(*[ConvNeXtBlock(C3, kernel=dw_kernel) for _ in range(d3)])

        # Stage 4: H/16 → H/32, C3 → C4
        self.down4  = DownsampleBlock(C3, C4)
        self.stage4 = nn.Sequential(*[ConvNeXtBlock(C4, kernel=dw_kernel) for _ in range(d4)])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        f_detail = self.stem(x)
        # Gradient checkpointing on the 4 stages: recompute activations in
        # backward instead of storing them, cutting peak VRAM by ~35-40%
        # and enabling batch_size to increase from 16 → 32 on a 24 GB GPU.
        _ck = checkpoint
        f1 = _ck(self.stage1, self.down1(f_detail), use_reentrant=False)
        f2 = _ck(self.stage2, self.down2(f1),       use_reentrant=False)
        f3 = _ck(self.stage3, self.down3(f2),       use_reentrant=False)
        f4 = _ck(self.stage4, self.down4(f3),       use_reentrant=False)
        return f_detail, f1, f2, f3, f4


# ── Model ─────────────────────────────────────────────────────────────────────

class sf_seg_v2(nn.Module):
    """
    SF-Seg V2: SFBackbone + 4 attention heads + lightweight decoder.

    Interface giống sf_seg (V1):
      forward(x) → (logits, attn_guide, attn_large)
      model._attns  — dict với saliency maps của 4 heads
      model._aux    — auxiliary logits (training only)
    """

    def __init__(self, num_channels: int = 32, focus_size: int = 64,
                 num_classes: int = 151, backbone_variant: str = 'micro',
                 dw_kernel: int = 3):
        super().__init__()
        self.num_classes = num_classes
        C = num_channels
        C1, C2, C3, C4 = C, 2 * C, 4 * C, 8 * C   # 32, 64, 128, 256 (với C=32)
        D = 2 * C                                    # decoder width = 64

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = SFBackbone((C1, C2, C3, C4), variant=backbone_variant,
                                   dw_kernel=dw_kernel)

        # ── Attention heads (giống V1, channels nhỏ hơn) ─────────────────────
        self.head_tiny   = SparseAttnHead(C4, max(4, focus_size // 8), num_heads=8)
        self.head_small  = SparseAttnHead(C3, max(4, focus_size // 4), num_heads=4)
        self.head_medium = SparseAttnHead(C2, max(4, focus_size // 8), num_heads=4,
                                          cross_kv_feat_dim=C4)
        self.head_large  = AttentionHead(C1, focus_size, guided=False)

        # ── Decoder: lightweight single-conv fuse ────────────────────────────
        # Project C4→D và C3→D (C2=D và C1=D//2 đã match, không cần project)
        def _proj(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, bias=False), _gn(out_c), nn.ReLU(inplace=True))

        def _fuse(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                _gn(out_c), nn.ReLU(inplace=True))

        self.proj_tiny  = _proj(C4, D)          # 256 → 64
        self.proj_small = _proj(C3, D)          # 128 → 64

        self.fuse_ts   = _fuse(D + D,     D)       # tiny+small   : 128 → 64
        self.fuse_tsm  = _fuse(D + C2,   D)        # +medium      : 128 → 64  (C2=D=64)
        self.fuse_tsml = _fuse(D + C1,   D // 2)   # +large       :  96 → 32  (C1=32)
        self.pre_masks = _fuse(D // 2,   D // 2)   # refine       :  32 → 32

        # Classifier
        _cls_h = max(256, num_classes)
        self.masks = nn.Sequential(
            nn.Conv2d(D // 2, _cls_h, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(_cls_h, num_classes, 1),
        )

        # ── Detail branch: H/2 skip cho sharp boundaries ──────────────────────
        _hr = max(16, D // 4)    # = 16
        self.hr_adapt = nn.Sequential(
            nn.Conv2d(32, _hr, 1, bias=False), _gn(_hr), nn.ReLU(inplace=True))
        self.hr_fuse  = nn.Sequential(
            nn.Conv2d(D // 2 + _hr, D // 2, 3, padding=1, bias=False),
            _gn(D // 2), nn.ReLU(inplace=True))

        # ── Auxiliary classifiers (deep supervision) ──────────────────────────
        self.aux_cls_tiny   = nn.Conv2d(C4, num_classes, 1)
        self.aux_cls_small  = nn.Conv2d(C3, num_classes, 1)
        self.aux_cls_medium = nn.Conv2d(C2, num_classes, 1)
        self._aux: tuple | None   = None
        self._attns: dict         = {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor):
        return self.backbone(x)   # (f_detail, f1, f2, f3, f4)

    def extract_features(self, x: torch.Tensor, full_res: bool = True):
        f_detail, f1, f2, f3, f4 = self._encode(x)

        a_tiny,   sal_tiny   = self.head_tiny(f4)
        a_small,  sal_small  = self.head_small(f3)
        a_medium, sal_medium = self.head_medium(f2, global_feat=a_tiny)
        a_large,  attn_l     = self.head_large(f1)

        self._attns = {
            'tiny':   sal_tiny.detach(),    # (B, 8,  H/32, W/32) — received attn
            'small':  sal_small.detach(),   # (B, 4,  H/16, W/16)
            'medium': sal_medium.detach(),  # (B, 4,  H/32, W/32) — global token map
            'large':  attn_l.detach(),      # (B, C1, H/4,  W/4)
        }

        if self.training:
            self._aux = (
                self.aux_cls_tiny(a_tiny),
                self.aux_cls_small(a_small),
                self.aux_cls_medium(a_medium),
            )
        else:
            self._aux = None

        # Bottom-up fusion
        t_up  = F.interpolate(self.proj_tiny(a_tiny),
                              a_small.shape[2:], mode='bilinear', align_corners=False)
        s_p   = self.proj_small(a_small)
        d1    = self.fuse_ts(torch.cat([t_up, s_p], dim=1))          # (B, D, H/16)

        d1_up = F.interpolate(d1, a_medium.shape[2:], mode='bilinear', align_corners=False)
        d2    = self.fuse_tsm(torch.cat([d1_up, a_medium], dim=1))   # (B, D, H/8)

        d2_up = F.interpolate(d2, a_large.shape[2:], mode='bilinear', align_corners=False)
        d3    = self.fuse_tsml(torch.cat([d2_up, a_large], dim=1))   # (B, D//2, H/4)

        if not full_res:
            return d3, attn_l, f_detail
        return self.pre_masks(d3), attn_l, f_detail

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        H, W                    = x.shape[2], x.shape[3]
        d_up, attn_l, f_detail  = self.extract_features(x)   # (B, D//2, H/4)

        # Detail refinement at H/2
        d_half = F.interpolate(d_up, (H // 2, W // 2), mode='bilinear', align_corners=False)
        hr     = self.hr_adapt(f_detail)
        d_half = self.hr_fuse(torch.cat([d_half, hr], dim=1))

        # Classify at H/2 → upsample to full res
        logits     = F.interpolate(
            self.masks(d_half), (H, W), mode='bilinear', align_corners=False)
        attn_guide = F.interpolate(
            attn_l.amax(dim=1, keepdim=True), (H, W), mode='bilinear', align_corners=False)
        return logits, attn_guide, attn_l

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # Stub methods dùng chung với V1 (compatibility)
    def routing_sparsity_loss(self):
        return None

    def routing_weight_stats(self):
        return {}
