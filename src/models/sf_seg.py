import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gn(C: int) -> nn.GroupNorm:
    for g in [8, 4, 2, 1]:
        if C % g == 0:
            return nn.GroupNorm(g, C)


# ── ResNet-style basic block ──────────────────────────────────────────────────

class BasicBlock(nn.Module):
    """
    ResNet BasicBlock: Conv-GN-GELU → Conv-GN + residual shortcut.

    stride=2  : downsample spatially, shortcut uses 1×1 projection.
    in_c != out_c : shortcut uses 1×1 projection to match channels.
    """
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            _gn(out_c),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            _gn(out_c),
        )
        need_proj = (stride != 1 or in_c != out_c)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
            _gn(out_c),
        ) if need_proj else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.shortcut(x) + self.conv2(self.conv1(x)))


# ── Sparse attention (clamped softmax) ───────────────────────────────────────

def _clamped_softmax(score: torch.Tensor, k: float) -> torch.Tensor:
    """Budget-constrained softmax: each value in [0,1], sum = k per channel.

    Always computed in float32 — 1e-9 underflows to 0 in float16, causing
    division-by-zero NaN that propagates through the whole forward pass.
    """
    orig_dtype = score.dtype
    score = score.float()                                  # float32 regardless of AMP

    L     = score.shape[-1]
    int_k = int(k)
    p     = F.softmax(score, dim=-1) * k

    top_vals   = torch.topk(p, k=int_k, dim=-1).values
    top_sorted = torch.sort(top_vals, dim=-1, descending=True).values
    cumsum     = top_sorted.cumsum(dim=-1)

    j        = torch.arange(1, int_k + 1, device=score.device, dtype=torch.float32)
    lam_j    = (j - cumsum) / (L - j).clamp(min=1e-6)    # 1e-6 safe for float32
    valid    = top_sorted - 1.0 >= lam_j
    j_sat    = valid.long().sum(dim=-1, keepdim=True)
    lam_star = torch.gather(lam_j, dim=-1, index=(j_sat - 1).clamp(min=0))
    lam_star = lam_star * (j_sat > 0).float()

    return (p - lam_star).clamp(0.0, 1.0).to(orig_dtype)


# ── Attention head (on backbone features) ────────────────────────────────────

class attention_head(nn.Module):
    """
    Sparse-focus attention operating on shared backbone features.

    Input  : (B, C, H, W) — feature map from one backbone stage (not raw RGB).
    Score  : DWConv(3×3) captures local spatial context before scoring,
             then 1×1 conv projects to 2C and splits into score | features.
    Attn   : clamped_softmax → sparse weights, budget = focus_size².
    Output : channel_mix(attn × features),  attn
    """

    def __init__(self, num_channels: int, focus_size: int):
        super().__init__()
        C = num_channels
        self.focus_size = focus_size
        self.proj = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False),  # depthwise 3×3
            nn.Conv2d(C, C * 2, 1, bias=False),                    # pointwise → score|feats
            _gn(C * 2),
        )
        self.channel_mix = nn.Sequential(
            nn.Conv2d(C, C, 1, bias=False),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor):
        score, features = self.proj(x).chunk(2, dim=1)    # each (B, C, H, W)
        B, C, H, W = score.shape
        k    = min(self.focus_size ** 2, H * W - 1)
        attn = _clamped_softmax(score.view(B, C, H * W), float(k)).view(B, C, H, W)
        return self.channel_mix(attn * features), attn


# ── Multi-scale segmentation model ───────────────────────────────────────────

class sf_seg(nn.Module):
    """
    Multi-scale Sparse-Focus Segmentation with ResNet backbone.

    Backbone (shared, ResNet-style BasicBlocks with GN + GELU):
    ┌────────┬──────────────────────────────────┬────────────────┐
    │ Layer  │ Blocks                           │ Output         │
    ├────────┼──────────────────────────────────┼────────────────┤
    │ stem   │ Conv(3→C//2, 3×3, s=2) + GN+GELU│ (C//2, H/2)   │
    │ stage1 │ BasicBlock(C//2→C, s=2)          │                │
    │        │ BasicBlock(C→C)                  │ (C,    H/4)  ←─ head_large  │
    │ stage2 │ BasicBlock(C→C, s=2)             │                │
    │        │ BasicBlock(C→C)                  │ (C,    H/8)  ←─ head_medium │
    │ stage3 │ BasicBlock(C→C, s=2)             │                │
    │        │ BasicBlock(C→C)                  │ (C,    H/16) ←─ head_small  │
    └────────┴──────────────────────────────────┴────────────────┘

    Attention heads: lightweight DW+PW projection on backbone features
    → clamped softmax → attn × features → channel_mix.

    Decoder (same bottom-up UNet structure as 8749cd1, GELU instead of ReLU):
    a_small → upsample → blend_up_sm → cat(a_medium) → fuse_sm_med → d_med
    d_med   → upsample → blend_up_med → cat(a_large) → fuse_med_lg → d_lg
    d_lg    → upsample → pre_masks → masks → logits
    """

    def __init__(self, num_channels: int = 32, focus_size: int = 32,
                 encoder_stride: int = 2, num_classes: int = 1,
                 decoder_type: str = "dense",
                 encoder_pretrained: str | None = None):
        super().__init__()
        self.num_classes    = num_classes
        self.encoder_stride = encoder_stride
        self.decoder_type   = decoder_type
        C = num_channels

        # ── Shared ResNet backbone ─────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, C // 2, 3, stride=2, padding=1, bias=False),
            _gn(C // 2),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(               # H/4  → head_large
            BasicBlock(C // 2, C, stride=2),
            BasicBlock(C, C),
        )
        self.stage2 = nn.Sequential(               # H/8  → head_medium
            BasicBlock(C, C, stride=2),
            BasicBlock(C, C),
        )
        self.stage3 = nn.Sequential(               # H/16 → head_small
            BasicBlock(C, C, stride=2),
            BasicBlock(C, C),
        )

        # ── Attention heads ────────────────────────────────────────────────────
        self.head_small  = attention_head(C, focus_size=max(2, focus_size // 8))
        self.head_medium = attention_head(C, focus_size=max(4, focus_size // 2))
        self.head_large  = attention_head(C, focus_size=focus_size)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.blend_up_sm  = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, bias=False), _gn(C), nn.GELU(),
        )
        self.blend_up_med = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, bias=False), _gn(C), nn.GELU(),
        )
        self.fuse_sm_med = nn.Sequential(
            nn.Conv2d(C + C, C, 3, padding=1, bias=False), _gn(C), nn.GELU(),
            nn.Conv2d(C, C, 3, padding=1, bias=False),     _gn(C), nn.GELU(),
        )
        self.fuse_med_lg = nn.Sequential(
            nn.Conv2d(C + C, C // 2, 3, padding=1, bias=False), _gn(C // 2), nn.GELU(),
            nn.Conv2d(C // 2, C // 2, 3, padding=1, bias=False), _gn(C // 2), nn.GELU(),
        )
        self.pre_masks = nn.Sequential(
            nn.Conv2d(C // 2, C // 2, 3, padding=1, bias=False), _gn(C // 2), nn.GELU(),
        )
        out_ch = num_classes if num_classes > 1 else 1
        self.masks = nn.Conv2d(C // 2, out_ch, kernel_size=1)

        if encoder_pretrained:
            self._load_pretrained_encoder(encoder_pretrained)

    # ── Backbone ──────────────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor):
        """Return backbone features at H/4, H/8, H/16."""
        f  = self.stem(x)
        f_l = self.stage1(f)
        f_m = self.stage2(f_l)
        f_s = self.stage3(f_m)
        return f_l, f_m, f_s

    # ── Forward ───────────────────────────────────────────────────────────────

    def extract_features(self, x: torch.Tensor, full_res: bool = True):
        """Run backbone + attention + decoder up to pre_masks.

        full_res=False : stop at d_lg (H/4) — used by pretrain_encoder.py.
        """
        H, W = x.shape[2], x.shape[3]

        f_l, f_m, f_s = self._encode(x)
        a_large,  attn_l = self.head_large(f_l)    # (B, C, H/4,  W/4)
        a_medium, _      = self.head_medium(f_m)   # (B, C, H/8,  W/8)
        a_small,  _      = self.head_small(f_s)    # (B, C, H/16, W/16)

        # Bottom-up decoder
        a_s_up = self.blend_up_sm(
            F.interpolate(a_small, size=a_medium.shape[2:],
                          mode='bilinear', align_corners=False))
        d_med  = self.fuse_sm_med(torch.cat([a_s_up, a_medium], dim=1))

        d_m_up = self.blend_up_med(
            F.interpolate(d_med, size=a_large.shape[2:],
                          mode='bilinear', align_corners=False))
        d_lg   = self.fuse_med_lg(torch.cat([d_m_up, a_large], dim=1))

        if not full_res:
            return d_lg, attn_l   # (B, C//2, H/4, W/4)

        d_up = self.pre_masks(
            F.interpolate(d_lg, size=(H, W), mode='bilinear', align_corners=False))
        return d_up, attn_l

    def forward(self, x: torch.Tensor):
        H, W = x.shape[2], x.shape[3]
        d_up, attn_l = self.extract_features(x)
        logits = self.masks(d_up)

        attn_guide = attn_l.amax(dim=1, keepdim=True)
        attn_guide = F.interpolate(attn_guide, size=(H, W),
                                   mode='bilinear', align_corners=False)
        return logits, attn_guide, attn_l

    # ── Utilities ─────────────────────────────────────────────────────────────

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def routing_sparsity_loss(self):
        return None

    def routing_weight_stats(self):
        return {}

    def _load_pretrained_encoder(self, path: str) -> None:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        nc   = ckpt.get('num_channels')
        C    = self.stem[0].out_channels * 2
        if nc and nc != C:
            raise ValueError(f"Pretrained num_channels={nc} != model num_channels={C}")
        backbone_sd = ckpt.get('backbone')
        if backbone_sd is None:
            print("[warn] no 'backbone' key in checkpoint, skipping")
            return
        missing, _ = self.load_state_dict(backbone_sd, strict=False)
        missing = [k for k in missing if not k.startswith('masks.')]
        if missing:
            print(f"[warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        print(f"Loaded backbone from {path}  "
              f"(epoch={ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc',0):.4f})")


def main():
    print("sf_seg — ResNet backbone + GELU\n")
    model = sf_seg(num_channels=128, focus_size=28, num_classes=151)
    x     = torch.randn(2, 3, 224, 224)
    logits, attn_guide, attn = model(x)

    total = model.get_num_parameters()
    bb    = sum(p.numel() for m in [model.stem, model.stage1, model.stage2, model.stage3]
                for p in m.parameters())
    heads = sum(p.numel() for m in [model.head_small, model.head_medium, model.head_large]
                for p in m.parameters())
    print(f"Total params  : {total:,}")
    print(f"  backbone    : {bb:,}  (ResNet BasicBlocks)")
    print(f"  attn heads  : {heads:,}")
    print(f"  decoder+cls : {total - bb - heads:,}")
    print(f"Logits        : {tuple(logits.shape)}")

    print(f"\n{'Head':<14} {'Feature level':>14} {'focus_size':>12} {'coverage':>10}")
    print("-" * 54)
    for name, head, hw in [
        ("head_large",  model.head_large,  (224//4)**2),
        ("head_medium", model.head_medium, (224//8)**2),
        ("head_small",  model.head_small,  (224//16)**2),
    ]:
        fs = head.focus_size
        k  = min(fs**2, hw - 1)
        print(f"{name:<14} {'H/' + str(224 // (hw**0.5).__int__()):>14} "
              f"{fs:>12} {k/hw:>9.1%}")


if __name__ == "__main__":
    main()
