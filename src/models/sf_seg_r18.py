"""sf_seg với ResNet-18 backbone (ImageNet pretrained).

4-head multi-scale design — channels (shallow→deep, default num_channels=64):
  layer1 (64ch,  H/4)  → adapt_large  → head_large  (C   =  64)
  layer2 (128ch, H/8)  → adapt_medium → head_medium (2C  = 128)
  layer3 (256ch, H/16) → adapt_small  → head_small  (4C  = 256)
  layer4 (512ch, H/32) → adapt_tiny   → head_tiny   (8C  = 512)

Channel pyramid matches ResNet's natural width progression — fewer channels at
high resolution (H/4), more at low resolution (H/32). This avoids expensive
8× channel expansion at H/4 that dominated VRAM usage in the previous design.

Activations: ReLU throughout (no GELU).
Normalization: BatchNorm in ResNet backbone, GroupNorm in adapters/decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ────────────────────────────────────────────────────────────────────

def _gn(C: int) -> nn.GroupNorm:
    for g in [8, 4, 2, 1]:
        if C % g == 0:
            return nn.GroupNorm(g, C)


try:
    from src.ops import clamped_softmax as _clamped_softmax
except Exception:
    def _clamped_softmax(score: torch.Tensor, k: float) -> torch.Tensor:
        """PyTorch fallback — used when CUDA extension is unavailable."""
        orig_dtype = score.dtype
        score      = score.float()
        L, int_k   = score.shape[-1], int(k)
        p          = F.softmax(score, dim=-1) * k
        top_vals   = torch.topk(p, k=int_k, dim=-1).values
        top_sorted = torch.sort(top_vals, dim=-1, descending=True).values
        cumsum     = top_sorted.cumsum(dim=-1)
        j          = torch.arange(1, int_k + 1, device=score.device, dtype=torch.float32)
        lam_j      = (j - cumsum) / (L - j).clamp(min=1e-6)
        valid      = top_sorted - 1.0 >= lam_j
        j_sat      = valid.long().sum(dim=-1, keepdim=True)
        lam_star   = torch.gather(lam_j, -1, (j_sat - 1).clamp(min=0))
        lam_star   = lam_star * (j_sat > 0).float()
        return (p - lam_star).clamp(0.0, 1.0).to(orig_dtype)


class AttentionHead(nn.Module):
    """Spatial gating head (CBAM-style). Used for head_large (boundary detail)."""
    def __init__(self, num_channels: int, focus_size: int, guided: bool = False):
        super().__init__()
        C = num_channels
        self.focus_size = focus_size
        self.proj = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),
            nn.Conv2d(C, C * 2, 1, bias=False),
            _gn(C * 2),
        )
        self.channel_mix = nn.Sequential(
            nn.Conv2d(C, C, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.guidance_weight = nn.Parameter(torch.zeros(1)) if guided else None

    def forward(self, x: torch.Tensor,
                guidance: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        score, features = self.proj(x).chunk(2, dim=1)
        B, C, H, W = score.shape

        if guidance is not None and self.guidance_weight is not None:
            score = score + self.guidance_weight * guidance

        k    = min(self.focus_size ** 2, H * W - 1)
        attn = _clamped_softmax(score.view(B, C, H * W), float(k)).view(B, C, H, W)
        return self.channel_mix(attn * features), attn


class SparseAttnHead(nn.Module):
    """True sparse attention head using clamped_softmax.

    Self-attention (global_feat=None):
        out[n] = Σ_m α(n,m) · V[m],  α = clamped_softmax(Q·Kᵀ / √d)
        → information flows between ALL positions (global receptive field)

    Cross-attention (global_feat≠None):
        K, V are projected from global_feat (coarser scale features);
        each local query attends to the full set of global tokens.
    """

    def __init__(self, C: int, focus_size: int, num_heads: int = 4,
                 cross_kv_feat_dim: int | None = None):
        super().__init__()
        assert C % num_heads == 0, f"C={C} not divisible by num_heads={num_heads}"
        self.C         = C
        self.num_heads = num_heads
        self.d         = C // num_heads
        self.scale     = self.d ** -0.5
        self.focus_size = focus_size

        self.qkv = nn.Conv2d(C, 3 * C, 1, bias=False)

        if cross_kv_feat_dim is not None:
            self.cross_proj_k = nn.Conv2d(cross_kv_feat_dim, C, 1, bias=False)
            self.cross_proj_v = nn.Conv2d(cross_kv_feat_dim, C, 1, bias=False)
        else:
            self.cross_proj_k = None
            self.cross_proj_v = None

        self.out_proj = nn.Sequential(
            nn.Conv2d(C, C, 1, bias=False),
            _gn(C),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor,
                global_feat: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        N  = H * W
        H_ = self.num_heads
        d  = self.d

        qkv = self.qkv(x).view(B, 3, H_, d, N)
        Q, K_self, V_self = qkv.unbind(dim=1)   # each: (B, H_, d, N)
        Q      = Q.permute(0, 1, 3, 2)           # (B, H_, N, d)
        K_self = K_self.permute(0, 1, 3, 2)      # (B, H_, N, d)
        V_self = V_self.permute(0, 1, 3, 2)      # (B, H_, N, d)

        if global_feat is not None and self.cross_proj_k is not None:
            K_ext = self.cross_proj_k(global_feat)  # (B, C, H_g, W_g)
            V_ext = self.cross_proj_v(global_feat)
            N_g   = K_ext.shape[2] * K_ext.shape[3]
            K_use = K_ext.view(B, H_, d, N_g).permute(0, 1, 3, 2)  # (B, H_, N_g, d)
            V_use = V_ext.view(B, H_, d, N_g).permute(0, 1, 3, 2)
        else:
            K_use, V_use = K_self, V_self

        N_k = K_use.shape[2]
        sim = torch.matmul(Q, K_use.transpose(-2, -1)) * self.scale  # (B, H_, N, N_k)

        # Budget: cap at 25% of N_k so sparsity is resolution-independent.
        # focus_size² sets the upper bound (coarse scale = small focus).
        k_budget = min(self.focus_size ** 2, max(1, N_k // 4))
        attn     = _clamped_softmax(sim.reshape(B * H_ * N, N_k),
                                    float(k_budget)).view(B, H_, N, N_k)

        out = torch.matmul(attn, V_use)                     # (B, H_, N, d)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)  # (B, C, H, W)
        out = self.out_proj(out)

        # Spatial saliency: received attention per KEY position (column mean).
        # This answers "which positions are being looked at?" rather than
        # "how much does each query send?" (row mean — always constant = k/N_k).
        # For self-attn: shape (B, H_, N_k) → view as (B, H_, H, W).
        # For cross-attn: shape (B, H_, N_k=global) → view as (B, H_, H_k, W_k).
        with torch.no_grad():
            H_k = round(N_k ** 0.5)
            W_k = N_k // H_k
            saliency = attn.detach().mean(dim=-2).view(B, H_, H_k, W_k)
        return out, saliency


# ── ResNet-18 loader ───────────────────────────────────────────────────────────

def _build_resnet18(pretrained: bool):
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        return resnet18(weights=weights)
    except ImportError:
        raise ImportError("pip install torchvision")


# ── Decoder helpers ────────────────────────────────────────────────────────────

def _fuse_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), _gn(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False), _gn(out_c), nn.ReLU(inplace=True),
    )


def _blend_block(c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c, c, 3, padding=1, bias=False), _gn(c), nn.ReLU(inplace=True),
    )


# ── Model ──────────────────────────────────────────────────────────────────────

class sf_seg(nn.Module):
    """
    Multi-scale Sparse-Focus Segmentation với ResNet-18 backbone.

    Channel pyramid (shallow → deep, default num_channels=64):
      head_large  :  C =  64  at H/4   (layer1 adapted, fine spatial detail)
      head_medium : 2C = 128  at H/8   (layer2 adapted)
      head_small  : 4C = 256  at H/16  (layer3 adapted)
      head_tiny   : 8C = 512  at H/32  (layer4 adapted, global semantic context)

    Adapters are now identity-like (no channel expansion) — matches ResNet's
    natural width progression so no expensive 8× expansion at H/4.

    ResNet-18 feature map sizes (input H×W):
    ┌──────────────┬──────────┬────────┬───────────────────────────────────┐
    │ ResNet layer │ Channels │ Scale  │ Adapter → Head                    │
    ├──────────────┼──────────┼────────┼───────────────────────────────────┤
    │ layer1       │ 64       │ H/4    │ adapt_large  (64→C=64)  head_large  │
    │ layer2       │ 128      │ H/8    │ adapt_medium (128→2C=128) head_medium│
    │ layer3       │ 256      │ H/16   │ adapt_small  (256→4C=256) head_small │
    │ layer4       │ 512      │ H/32   │ adapt_tiny   (512→8C=512) head_tiny  │
    └──────────────┴──────────┴────────┴───────────────────────────────────┘

    Decoder: bottom-up UNet (tiny → small → medium → large → full_res).
    Activations: ReLU. Normalization: BN (backbone), GN (adapters/decoder).
    """

    _R18_CHANNELS = (64, 128, 256, 512)

    def __init__(self, num_channels: int = 64, focus_size: int = 64,
                 encoder_stride: int = 1, num_classes: int = 151,
                 decoder_type: str = "dense",
                 encoder_pretrained: str | None = None,
                 pretrained_backbone: bool = True):
        super().__init__()
        self.num_classes    = num_classes
        self.encoder_stride = encoder_stride
        self.decoder_type   = decoder_type
        C = num_channels
        C1, C2, C3, C4 = C, 2 * C, 4 * C, 8 * C   # 64, 128, 256, 512

        # ── ResNet-18 backbone ─────────────────────────────────────────────────
        r18 = _build_resnet18(pretrained=pretrained_backbone)
        # Split stem to expose H/2 intermediate features for the detail branch.
        # conv1(7×7,s=2)+bn1+relu → (64, H/2);  maxpool(s=2) → (64, H/4)
        self.r18_stem_conv = nn.Sequential(r18.conv1, r18.bn1, r18.relu)  # → H/2
        self.r18_stem_pool = r18.maxpool                                    # → H/4
        self.r18_layer1 = r18.layer1   # (64,  H/4)
        self.r18_layer2 = r18.layer2   # (128, H/8)
        self.r18_layer3 = r18.layer3   # (256, H/16)
        self.r18_layer4 = r18.layer4   # (512, H/32)

        # ── Adapters ───────────────────────────────────────────────────────────
        ch_l, ch_m, ch_s, ch_t = self._R18_CHANNELS
        self.adapt_large  = nn.Sequential(
            nn.Conv2d(ch_l, C1, 1, bias=False), _gn(C1), nn.ReLU(inplace=True))
        self.adapt_medium = nn.Sequential(
            nn.Conv2d(ch_m, C2, 1, bias=False), _gn(C2), nn.ReLU(inplace=True))
        self.adapt_small  = nn.Sequential(
            nn.Conv2d(ch_s, C3, 1, bias=False), _gn(C3), nn.ReLU(inplace=True))
        self.adapt_tiny   = nn.Sequential(
            nn.Conv2d(ch_t, C4, 1, bias=False), _gn(C4), nn.ReLU(inplace=True))

        # ── Attention heads ────────────────────────────────────────────────────
        # tiny/small: true sparse self-attention (global receptive field)
        # medium: cross-attention to tiny's global tokens (4×fewer tokens to attend)
        # large: spatial gating (CBAM-style) — kept for sharp boundary detail
        self.head_tiny   = SparseAttnHead(C4, max(4, focus_size // 8), num_heads=8)
        self.head_small  = SparseAttnHead(C3, max(4, focus_size // 4), num_heads=4)
        # focus_size//8: cross-attention N_k = (H/32)² = 256 at 512px
        # using focus_size//2 would give k=1024 >> N_k=256 → always 100% budget (not sparse)
        # focus_size//8 = 8 → k=64 out of 256 global tokens = 25% budget
        self.head_medium = SparseAttnHead(C2, max(4, focus_size // 8), num_heads=4,
                                          cross_kv_feat_dim=C4)
        self.head_large  = AttentionHead(C1, focus_size, guided=False)

        # ── Decoder ─────────────────────────────────────────────────────────────
        D = 2 * C   # decoder width (default 128)
        self.blend_up_tiny = _blend_block(C4)
        self.fuse_tiny_sm  = _fuse_block(C4 + C3, D)       # 512+256=768 → 128
        self.blend_up_sm   = _blend_block(D)
        self.fuse_sm_med   = _fuse_block(D + C2, D)        # 128+128=256 → 128
        self.blend_up_med  = _blend_block(D)
        self.fuse_med_lg   = _fuse_block(D + C1, D // 2)   # 128+64=192  → 64
        self.pre_masks     = _blend_block(D // 2)
        out_ch = num_classes if num_classes > 1 else 1
        _cls_hidden = max(256, out_ch)
        self.masks = nn.Sequential(
            nn.Conv2d(D // 2, _cls_hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(_cls_hidden, out_ch, 1),
        )

        # ── Detail branch: H/2 skip from R18 stem → sharp boundary prediction ──
        # r18_stem_conv outputs 64ch at H/2; decoder outputs D//2 ch at H/4.
        # Fusing them at H/2 lets the classifier see 2× finer spatial resolution,
        # halving the minimum resolvable boundary width (4px → 2px at 512 input).
        _hr = max(32, D // 4)
        self.hr_adapt = nn.Sequential(
            nn.Conv2d(64, _hr, 1, bias=False), _gn(_hr), nn.ReLU(inplace=True))
        self.hr_fuse  = nn.Sequential(
            nn.Conv2d(D // 2 + _hr, D // 2, 3, padding=1, bias=False),
            _gn(D // 2), nn.ReLU(inplace=True))

        # ── Auxiliary classifiers for deep supervision ─────────────────────────
        self.aux_cls_tiny   = nn.Conv2d(C4, out_ch, 1)
        self.aux_cls_small  = nn.Conv2d(C3, out_ch, 1)
        self.aux_cls_medium = nn.Conv2d(C2, out_ch, 1)
        self._aux: tuple | None = None

        if encoder_pretrained:
            print("[r18] encoder_pretrained ignored — backbone uses ImageNet weights")

    # ── Encode ────────────────────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor):
        """Return adapted features at H/4, H/8, H/16, H/32."""
        f_hr   = self.r18_stem_conv(x)     # (64,  H/2)
        self._stem_hr = f_hr               # cache H/2 skip for hr_fuse in forward()
        f      = self.r18_stem_pool(f_hr)  # (64,  H/4)
        raw_l  = self.r18_layer1(f)        # (64,  H/4)
        raw_m  = self.r18_layer2(raw_l)    # (128, H/8)
        raw_s  = self.r18_layer3(raw_m)    # (256, H/16)
        raw_t  = self.r18_layer4(raw_s)    # (512, H/32)
        return (self.adapt_large(raw_l),   # (C1, H/4)
                self.adapt_medium(raw_m),  # (C2, H/8)
                self.adapt_small(raw_s),   # (C3, H/16)
                self.adapt_tiny(raw_t))    # (C4, H/32)

    # ── Forward ───────────────────────────────────────────────────────────────────

    def extract_features(self, x: torch.Tensor, full_res: bool = True):
        H, W = x.shape[2], x.shape[3]
        f1, f2, f3, f4 = self._encode(x)

        # head_tiny: sparse self-attention — global semantic context at H/32
        a_tiny,   sal_tiny   = self.head_tiny(f4)

        # head_small: sparse self-attention — fine detail at H/16
        a_small,  sal_small  = self.head_small(f3)

        # head_medium: cross-attention to head_tiny's output as global context
        a_medium, sal_medium = self.head_medium(f2, global_feat=a_tiny)

        # head_large: spatial gating for boundary-sharp detail
        a_large, attn_l = self.head_large(f1)

        # Cache all attention maps (saliency per head/channel, detached)
        # sal_*: (B, num_heads, H_scale, W_scale) from SparseAttnHead
        # attn_l: (B, C, H/4, W/4) from AttentionHead
        self._attns = {
            'tiny':   sal_tiny.detach(),    # (B, 8,  H/32, W/32)
            'small':  sal_small.detach(),   # (B, 4,  H/16, W/16)
            'medium': sal_medium.detach(),  # (B, 4,  H/8,  W/8)
            'large':  attn_l.detach(),      # (B, 64, H/4,  W/4)
        }

        if self.training:
            self._aux = (
                self.aux_cls_tiny(a_tiny),
                self.aux_cls_small(a_small),
                self.aux_cls_medium(a_medium),
            )
        else:
            self._aux = None

        t_up  = self.blend_up_tiny(
            F.interpolate(a_tiny,  a_small.shape[2:],   mode='bilinear', align_corners=False))
        d_sm  = self.fuse_tiny_sm(torch.cat([t_up, a_small], dim=1))

        s_up  = self.blend_up_sm(
            F.interpolate(d_sm,   a_medium.shape[2:],  mode='bilinear', align_corners=False))
        d_med = self.fuse_sm_med(torch.cat([s_up, a_medium], dim=1))

        m_up  = self.blend_up_med(
            F.interpolate(d_med,  a_large.shape[2:],   mode='bilinear', align_corners=False))
        d_lg  = self.fuse_med_lg(torch.cat([m_up, a_large], dim=1))

        if not full_res:
            return d_lg, attn_l

        # pre_masks at H/4 — classification happens at low res, upsample in forward()
        return self.pre_masks(d_lg), attn_l

    def forward(self, x: torch.Tensor):
        H, W = x.shape[2], x.shape[3]
        d_up, attn_l = self.extract_features(x)   # d_up = pre_masks(d_lg) at H/4

        # Detail refinement at H/2: fuse decoder with stem skip
        d_half = F.interpolate(d_up, (H // 2, W // 2), mode='bilinear', align_corners=False)
        hr     = self.hr_adapt(self._stem_hr)                   # (B, _hr, H/2)
        d_half = self.hr_fuse(torch.cat([d_half, hr], dim=1))  # (B, D//2, H/2)

        # Classify at H/2, then 2× upsample — halves minimum boundary width vs H/4
        logits = F.interpolate(
            self.masks(d_half), (H, W), mode='bilinear', align_corners=False)
        attn_guide   = attn_l.amax(dim=1, keepdim=True)
        attn_guide   = F.interpolate(attn_guide, (H, W), mode='bilinear', align_corners=False)
        return logits, attn_guide, attn_l

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def routing_sparsity_loss(self):
        return None

    def routing_weight_stats(self):
        return {}


def main():
    print("sf_seg_r18 — ResNet-18 backbone (ImageNet pretrained, 4-head pyramid)\n")
    model = sf_seg(num_channels=64, focus_size=64, num_classes=151,
                   pretrained_backbone=True)
    x = torch.randn(2, 3, 512, 512)
    logits, attn_guide, attn = model(x)

    total = model.get_num_parameters()
    r18   = sum(p.numel() for m in [model.r18_stem_conv, model.r18_stem_pool,
                                     model.r18_layer1, model.r18_layer2,
                                     model.r18_layer3, model.r18_layer4]
                for p in m.parameters())
    adpt  = sum(p.numel() for m in [model.adapt_large, model.adapt_medium,
                                     model.adapt_small, model.adapt_tiny]
                for p in m.parameters())
    heads = sum(p.numel() for m in [model.head_large,  model.head_medium,
                                     model.head_small, model.head_tiny]
                for p in m.parameters())
    print(f"Total params      : {total:,}")
    print(f"  ResNet-18 (1-4) : {r18:,}")
    print(f"  adapters (4)    : {adpt:,}")
    print(f"  attn heads (4)  : {heads:,}")
    print(f"  decoder+cls     : {total - r18 - adpt - heads:,}")
    print(f"Logits            : {tuple(logits.shape)}")
    print(f"Attn (large)      : {tuple(attn.shape)}")


if __name__ == "__main__":
    main()
