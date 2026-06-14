"""SF-Seg V2 — Custom ConvNeXt-style backbone thay ResNet-18.

So với V1 (sf_seg_r18.py):
  - Backbone tự thiết kế (SFBackbone), không dùng pretrained ImageNet
  - Channel pyramid [32, 64, 128, 256] với num_channels=32 (V1: [64,128,256,512])
  - Không cần adapters — backbone output channels khớp trực tiếp với head sizes
  - Decoder nhẹ hơn: single 3×3 conv mỗi tầng (V1: 2-conv fuse blocks)
  - LayerScale trong mỗi ConvNeXtBlock → stable training từ đầu

Variants (backbone depth [s1, s2, s3, s4]):
  'nano'  : [2, 3, 6, 2] → ~3.3M total
  'micro' : [2, 3, 9, 2] → ~3.85M total ← default (ngang SegFormer-B0 3.7M)

Decoder cuối mở rộng (2026-06-11): fuse_tsml ra decoder_dim=256 @H/4 +
ConvNeXt refine block, project về hr_dim=96 trước khi lên H/2. Bottleneck
32-dim cũ là nguyên nhân confusion giữa class ngữ nghĩa gần nhau
(door→wall 30%, plant→tree 30% trên val).
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
        'small': [3, 4, 12, 3],   # với num_channels=64 → ~17M total
    }

    def __init__(self, channels: tuple[int, int, int, int], variant: str = 'micro',
                 dw_kernel: int = 3, grad_checkpoint: bool = True):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
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
        if self.grad_checkpoint and self.training:
            # Recompute stage activations in backward: −35-40% peak VRAM,
            # nhưng backbone bị forward 2 lần → chậm ~25-35%/step.
            _ck = checkpoint
            f1 = _ck(self.stage1, self.down1(f_detail), use_reentrant=False)
            f2 = _ck(self.stage2, self.down2(f1),       use_reentrant=False)
            f3 = _ck(self.stage3, self.down3(f2),       use_reentrant=False)
            f4 = _ck(self.stage4, self.down4(f3),       use_reentrant=False)
        else:
            f1 = self.stage1(self.down1(f_detail))
            f2 = self.stage2(self.down2(f1))
            f3 = self.stage3(self.down3(f2))
            f4 = self.stage4(self.down4(f3))
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
                 dw_kernel: int = 3, decoder_dim: int = 256, hr_dim: int = 96,
                 grad_checkpoint: bool = True,
                 attn_masks: tuple[int, int, int] | None = None,
                 budget_ladder: bool = False, pos_encode: bool = False,
                 dropout: float = 0.0, enable_ensemble: bool = False,
                 attn_temperature: float = 1.0, attn_op: str = 'topk'):
        super().__init__()
        self.num_classes = num_classes
        self.drop_p = dropout      # Dropout2d trước classifier (functional → key names ổn định)
        self.enable_ensemble = enable_ensemble
        C = num_channels
        C1, C2, C3, C4 = C, 2 * C, 4 * C, 8 * C   # 32, 64, 128, 256 (với C=32)
        D = 2 * C                                    # decoder width = 64
        Dd = decoder_dim                             # wide final decoder = 256

        # ── Backbone ──────────────────────────────────────────────────────────
        self.backbone = SFBackbone((C1, C2, C3, C4), variant=backbone_variant,
                                   dw_kernel=dw_kernel, grad_checkpoint=grad_checkpoint)

        # ── Attention heads ───────────────────────────────────────────────────
        # attn_masks: số masks (tiny, small, medium) — mặc định 2× quy tắc cũ
        # vì qk_dim=32 đã tách khỏi phép chia channel (nhiều masks không làm
        # similarity dim co lại; chỉ value dim chia nhỏ).
        # budget_ladder: budgets log-spaced k_max→4 — mask to làm stuff-detector,
        # mask nhỏ cho object nhỏ sở hữu trọn mass (k là SÀN diện tích focus).
        if attn_masks is None:
            attn_masks = (max(16, C4 // 16), max(8, C3 // 16), max(8, C2 // 8))
        nh_tiny, nh_small, nh_medium = attn_masks
        _qk = 32
        self.head_tiny   = SparseAttnHead(C4, max(4, focus_size // 8), num_heads=nh_tiny,
                                          qk_dim=_qk, budget_ladder=budget_ladder, pos_encode=pos_encode,
                                          temperature=attn_temperature, attn_op=attn_op)
        self.head_small  = SparseAttnHead(C3, max(4, focus_size // 4), num_heads=nh_small,
                                          qk_dim=_qk, budget_ladder=budget_ladder, pos_encode=pos_encode,
                                          temperature=attn_temperature, attn_op=attn_op)
        self.head_medium = SparseAttnHead(C2, max(4, focus_size // 8), num_heads=nh_medium,
                                          cross_kv_feat_dim=C4, qk_dim=_qk, budget_ladder=budget_ladder,
                                          pos_encode=pos_encode, temperature=attn_temperature, attn_op=attn_op)
        # head_large: SparseAttnHead cross-attn (Q từ f1 @H/4, K/V từ f4 global).
        # Trước là spatial-gating (clamp k≥N → dense 100% "đỏ hết", ablation +0.0006
        # vô dụng). Giờ sparse + global context ở high-res → có cơ hội hữu ích.
        self.head_large  = SparseAttnHead(C1, focus_size, num_heads=max(4, nh_medium // 2),
                                          cross_kv_feat_dim=C4, qk_dim=_qk, budget_ladder=budget_ladder,
                                          pos_encode=pos_encode, temperature=attn_temperature, attn_op=attn_op)

        # ── Sparse Ensemble branch (region-gated weak predictors) ─────────────
        # Mỗi mask → classifier (dv→C) trên feature value-weighted, gated bởi
        # vùng attend → một weak predictor. Gộp Σ (pred·gate) + correction head.
        if enable_ensemble:
            dv_t, dv_s, dv_m = C4 // nh_tiny, C3 // nh_small, C2 // nh_medium
            def _mask_clf(dv):   # dv→hidden→C cho mask mỏng
                h = max(32, dv)
                return nn.Sequential(nn.Conv2d(dv, h, 1), nn.GELU(),
                                     nn.Conv2d(h, num_classes, 1))
            self.ens_clf_tiny   = _mask_clf(dv_t)
            self.ens_clf_small  = _mask_clf(dv_s)
            self.ens_clf_medium = _mask_clf(dv_m)
            # Correction: [decoder_logit (C) + ensemble_logit (C)] → refine → final
            _ec = max(256, num_classes)
            self.ens_correct = nn.Sequential(
                nn.Conv2d(2 * num_classes, _ec, 3, padding=1),
                _gn(_ec), nn.GELU(),
                nn.Conv2d(_ec, num_classes, 1))
            nn.init.zeros_(self.ens_correct[-1].weight)   # zero-init → final≈decoder lúc đầu
            nn.init.zeros_(self.ens_correct[-1].bias)
        self._per_mask_preds: list | None = None
        self._ensemble_logit: torch.Tensor | None = None

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
        # Wide final decoder @ H/4: 32-dim bottleneck cũ gộp các class ngữ
        # nghĩa gần nhau (door→wall 30%, plant→tree 30% trên val) — 150-way
        # classification cần nhiều chiều phân biệt hơn.
        self.fuse_tsml = _fuse(D + C1,   Dd)       # +large       :  96 → 256 (C1=32)
        self.pre_masks = ConvNeXtBlock(Dd, expand=2, kernel=7)   # refine @ H/4

        # Project xuống hr_dim TRƯỚC khi upsample H/2 — giữ VRAM thấp
        self.proj_hr = _proj(Dd, hr_dim)           # 256 → 96, 1×1

        # Classifier
        _cls_h = max(256, num_classes)
        self.masks = nn.Sequential(
            nn.Conv2d(hr_dim, _cls_h, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(_cls_h, num_classes, 1),
        )

        # ── Detail branch: H/2 skip cho sharp boundaries ──────────────────────
        _hr = max(16, D // 4)    # = 16
        self.hr_adapt = nn.Sequential(
            nn.Conv2d(32, _hr, 1, bias=False), _gn(_hr), nn.ReLU(inplace=True))
        self.hr_fuse  = nn.Sequential(
            nn.Conv2d(hr_dim + _hr, hr_dim, 3, padding=1, bias=False),
            _gn(hr_dim), nn.ReLU(inplace=True))

        # ── Class-presence-guided global context ─────────────────────────────
        # g = GAP(f4): tóm tắt toàn ảnh. presence_head được supervise bằng BCE
        # multi-label "class nào có mặt trong ảnh" (label free từ mask) → g buộc
        # phải mang thông tin tổng quan. ctx_proj inject g (additive) vào decoder
        # @H/4 — mỗi pixel thấy scene context khi phân loại (fix door↔wall,
        # house↔building 46%: texture cục bộ giống hệt, chỉ scene phân biệt được).
        self.presence_head = nn.Linear(C4, num_classes)
        self.ctx_proj      = nn.Linear(C4, Dd)
        # Zero-init: tại bước 0 decoder y hệt model không có context → warm-start
        # từ checkpoint cũ không bị xáo trộn, context "mọc" dần theo gradient.
        nn.init.zeros_(self.ctx_proj.weight)
        nn.init.zeros_(self.ctx_proj.bias)

        # Late fusion (logit prior): S_final = S + late_ctx(p) — presence logits
        # chỉnh trực tiếp logits cuối, đồng nhất không gian. Phân rã Bayes:
        # log P(c|pixel,ảnh) ≈ log P(c|pixel) + log P(c|ảnh). Bù cho mid-fusion:
        # (a) ma trận 150×150 đọc được ("có bed → đè building"), (b) seg loss
        # có đường gradient trực tiếp về presence_head. Zero-init như ctx_proj.
        self.late_ctx = nn.Linear(num_classes, num_classes, bias=False)
        nn.init.zeros_(self.late_ctx.weight)

        # ── Auxiliary classifiers (deep supervision) ──────────────────────────
        self.aux_cls_tiny   = nn.Conv2d(C4, num_classes, 1)
        self.aux_cls_small  = nn.Conv2d(C3, num_classes, 1)
        self.aux_cls_medium = nn.Conv2d(C2, num_classes, 1)
        self._aux: tuple | None   = None
        self._attns: dict         = {}
        self._presence: torch.Tensor | None = None
        self._attn_div: torch.Tensor | None = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor):
        return self.backbone(x)   # (f_detail, f1, f2, f3, f4)

    def _build_ensemble(self, scales: list):
        """Mỗi mask → classifier(dv→C) gated bởi vùng attend → weak predictor.
        Gộp Σ(pred·gate)/Σgate ở res chung. Lưu per-mask preds cho loss.
        scales: list (classifier, per_mask_feat (B,M,dv,h,w), gate (B,M,h,w))."""
        preds, num, den = [], None, None
        ref_hw = scales[-1][1].shape[-2:]          # res chung = medium (H/8)
        for clf, feat, gate in scales:
            B, M, dv, h, w = feat.shape
            p = clf(feat.reshape(B * M, dv, h, w)).view(B, M, -1, h, w)  # (B,M,C,h,w)
            g = gate.unsqueeze(2)                                         # (B,M,1,h,w)
            g = g / (g.amax(dim=(-1, -2), keepdim=True) + 1e-6)           # norm [0,1] mỗi mask
            preds.append((p, gate))                                       # cho per-mask loss
            pg = (p * g).flatten(1, 2)                                    # (B,M*C,h,w)
            pg = F.interpolate(pg, ref_hw, mode='bilinear', align_corners=False)
            gg = F.interpolate(g.flatten(1, 2), ref_hw, mode='bilinear', align_corners=False)
            pg = pg.view(B, M, -1, *ref_hw); gg = gg.view(B, M, 1, *ref_hw)
            num = pg.sum(1) if num is None else num + pg.sum(1)           # (B,C,h,w)
            den = gg.sum(1) if den is None else den + gg.sum(1)
        self._ensemble_logit = num / (den + 1e-6)
        self._per_mask_preds = preds if self.training else None

    def extract_features(self, x: torch.Tensor, full_res: bool = True):
        f_detail, f1, f2, f3, f4 = self._encode(x)

        ens = self.enable_ensemble
        if ens:
            a_tiny,   sal_tiny,   pf_t, pg_t = self.head_tiny(f4, return_per_mask=True)
            a_small,  sal_small,  pf_s, pg_s = self.head_small(f3, return_per_mask=True)
            a_medium, sal_medium, pf_m, pg_m = self.head_medium(f2, global_feat=a_tiny,
                                                                return_per_mask=True)
            self._build_ensemble([(self.ens_clf_tiny,   pf_t, pg_t),
                                  (self.ens_clf_small,  pf_s, pg_s),
                                  (self.ens_clf_medium, pf_m, pg_m)])
        else:
            a_tiny,   sal_tiny   = self.head_tiny(f4)
            a_small,  sal_small  = self.head_small(f3)
            a_medium, sal_medium = self.head_medium(f2, global_feat=a_tiny)
            self._per_mask_preds = None
            self._ensemble_logit = None
        a_large,  attn_l     = self.head_large(f1, global_feat=a_tiny)

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
            # Gom anti-collapse diversity loss từ 3 sparse heads (mean)
            divs = [h._div_loss for h in (self.head_tiny, self.head_small, self.head_medium)
                    if getattr(h, '_div_loss', None) is not None]
            self._attn_div = sum(divs) / len(divs) if divs else None
        else:
            self._aux = None
            self._attn_div = None

        # Bottom-up fusion
        t_up  = F.interpolate(self.proj_tiny(a_tiny),
                              a_small.shape[2:], mode='bilinear', align_corners=False)
        s_p   = self.proj_small(a_small)
        d1    = self.fuse_ts(torch.cat([t_up, s_p], dim=1))          # (B, D, H/16)

        d1_up = F.interpolate(d1, a_medium.shape[2:], mode='bilinear', align_corners=False)
        d2    = self.fuse_tsm(torch.cat([d1_up, a_medium], dim=1))   # (B, D, H/8)

        d2_up = F.interpolate(d2, a_large.shape[2:], mode='bilinear', align_corners=False)
        d3    = self.fuse_tsml(torch.cat([d2_up, a_large], dim=1))   # (B, Dd, H/4)

        # Global context: GAP + GMP. Mean-pool một mình rửa trôi object nhỏ
        # (class chiếm 1/256 token → 0.4% của g; đo được: recall presence 4%
        # với class <1% diện tích). Max-pool cho phép MỘT token mạnh kích hoạt
        # class — sum giữ nguyên 256-d nên mọi layer phía sau không đổi shape.
        g = f4.mean(dim=(2, 3)) + f4.amax(dim=(2, 3))                # (B, C4)
        self._presence = self.presence_head(g)                       # (B, num_classes)
        d3 = d3 + self.ctx_proj(g).unsqueeze(-1).unsqueeze(-1)       # broadcast add

        if not full_res:
            return d3, attn_l, f_detail
        # Wide refine @ H/4, then project to hr_dim before H/2 upsample
        return self.proj_hr(self.pre_masks(d3)), attn_l, f_detail

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, upsample: bool = True):
        """upsample=False: trả logits tại H/2 (training loss tính tại H/2 với
        target nearest-downsample — nhanh 1.5×, VRAM −7.7GB; upsample cuối là
        bilinear không tham số nên không mất tín hiệu học). Val dùng full res."""
        H, W                    = x.shape[2], x.shape[3]
        d_up, attn_l, f_detail  = self.extract_features(x)   # (B, hr_dim, H/4)

        # Detail refinement at H/2
        d_half = F.interpolate(d_up, (H // 2, W // 2), mode='bilinear', align_corners=False)
        hr     = self.hr_adapt(f_detail)
        d_half = self.hr_fuse(torch.cat([d_half, hr], dim=1))

        # Channel dropout ngay trước classifier — chống overfit (train-val gap nở)
        if self.training and self.drop_p > 0:
            d_half = F.dropout2d(d_half, self.drop_p)
        logits = self.masks(d_half)                           # (B, C, H/2, W/2)

        # Late fusion: cộng prior per-class từ presence (broadcast mọi pixel).
        # KHÔNG detach p — seg loss sửa trực tiếp presence_head qua đường này.
        logits = logits + self.late_ctx(self._presence).unsqueeze(-1).unsqueeze(-1)

        # Ensemble correction: final = decoder_logit + correct([decoder, ensemble]).
        # Zero-init correct → lúc đầu final≈decoder (warm-start không xáo). Ensemble
        # (pre-predict từ các weak mask) sửa lại vùng sai.
        if self.enable_ensemble and self._ensemble_logit is not None:
            ens = F.interpolate(self._ensemble_logit, logits.shape[2:],
                                mode='bilinear', align_corners=False)
            logits = logits + self.ens_correct(torch.cat([logits, ens], dim=1))

        if not upsample:
            return logits, attn_l.amax(dim=1, keepdim=True), attn_l

        # Classify at H/2 → upsample to full res
        logits     = F.interpolate(logits, (H, W), mode='bilinear', align_corners=False)
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
