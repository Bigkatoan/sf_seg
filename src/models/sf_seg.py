import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sparse attention (clamped softmax) ───────────────────────────────────────

def _clamped_softmax(score: torch.Tensor, k: float) -> torch.Tensor:
    """
    Budget-constrained softmax: each value in [0,1], sum = k per channel.

    Closed-form solution via Lagrangian duality.
    Only topk(k) needed instead of full sort: O(k log k) vs O(L log L).
    All CUDA ops, no Python loop.
    """
    L     = score.shape[-1]
    int_k = int(k)
    p     = F.softmax(score, dim=-1) * k

    top_vals   = torch.topk(p, k=int_k, dim=-1).values
    top_sorted = torch.sort(top_vals, dim=-1, descending=True).values
    cumsum     = top_sorted.cumsum(dim=-1)

    j     = torch.arange(1, int_k + 1, device=score.device, dtype=p.dtype)
    lam_j = (j - cumsum) / (L - j).clamp(min=1e-9)

    valid    = top_sorted - 1.0 >= lam_j
    j_sat    = valid.long().sum(dim=-1, keepdim=True)
    lam_star = torch.gather(lam_j, dim=-1, index=(j_sat - 1).clamp(min=0))
    lam_star = lam_star * (j_sat > 0).to(p.dtype)

    return (p - lam_star).clamp(0.0, 1.0)


# ── Single-scale attention head ───────────────────────────────────────────────

class attention_head(nn.Module):
    """
    One attention head operating at a fixed spatial scale.

    Input  : x (B, 3, H, W) — already resized to target scale by sf_seg.
    Encoder: Conv(3→C, stride=2) → Conv(C→2C) → split score | features.
    Attention: clamped_softmax, budget k = min(focus_size², H'×W'-1).
               k is clamped per-forward so the same focus_size is safe across
               all three scales without overflow.
    Blend  : 1×1 conv after attn×features — mixes channels that were
             independently weighted by attention, enabling cross-channel
             interaction before entering the decoder.
    Output : attended (B, C, H/2, W/2),  attn (B, C, H/2, W/2).
    """

    def __init__(self, num_channels: int, focus_size: int):
        super().__init__()
        C = num_channels
        self.focus_size = focus_size
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )
        self.enc2        = nn.Conv2d(C, C * 2, 3, padding=1)
        self.channel_mix = nn.Sequential(        # cross-channel blend after attention
            nn.Conv2d(C, C, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        enc2_out        = self.enc2(self.enc1(x))          # (B, 2C, H/2, W/2)
        score, features = enc2_out.chunk(2, dim=1)

        B, C, Hh, Ww = score.shape
        k    = min(self.focus_size * self.focus_size, Hh * Ww - 1)
        attn = _clamped_softmax(score.view(B, C, Hh * Ww), float(k))
        attn = attn.view(B, C, Hh, Ww)

        attended = self.channel_mix(attn * features)       # (B, C, H/2, W/2)
        return attended, attn


# ── Multi-scale model ─────────────────────────────────────────────────────────

class sf_seg(nn.Module):
    """
    Multi-scale Sparse-Focus Segmentation.

    Three independent attention heads receive the image at three resolutions.
    Outputs are fused bottom-up (coarse→fine) in a UNet-style decoder.
    No shared weights between heads — each scale learns its own attention
    pattern without interference.

    Scales (default image_size=224, encoder_stride=2 inside each head):
    ┌──────────────┬─────────────┬──────────────────┬──────────────┐
    │ Head         │ Input       │ Attention space   │ Coverage k/L │
    ├──────────────┼─────────────┼──────────────────┼──────────────┤
    │ head_small   │ H/16 (14²)  │ H/32 × W/32 (7²) │ ~33%  global │
    │ head_medium  │ H/4  (56²)  │ H/8  × W/8 (28²) │ ~33%  mid    │
    │ head_large   │ H    (224²) │ H/2  × W/2 (112²)│  ~8%  local  │
    └──────────────┴─────────────┴──────────────────┴──────────────┘

    Decoder (bottom-up):
      a_small  (C, 7,   7)
        ↓ upsample ×4 → blend_up_sm (C→C, 3×3)          adapt to 28×28 space
        cat[↑, a_medium] → fuse_sm_med (2×conv)  →  d_med (C, 28,  28)
        ↓ upsample ×4 → blend_up_med (C→C, 3×3)          adapt to 112×112 space
        cat[↑, a_large]  → fuse_med_lg (2×conv)  →  d_lg  (C//2, 112, 112)
        ↓ upsample ×2 → pre_masks (C//2→C//2, 3×3)       final blend
        → masks (1×1)  →  logits (num_classes, H, W)

    blend_up_sm / blend_up_med: learned projections applied after bilinear
    upsample — they adapt stretched features to the target resolution space
    before concatenation (same principle as FPN lateral connections).
    """

    def __init__(self, num_channels: int = 32, focus_size: int = 32,
                 encoder_stride: int = 2, num_classes: int = 1,
                 decoder_type: str = "dense",
                 encoder_pretrained: str | None = None):
        super().__init__()
        self.num_classes    = num_classes
        self.encoder_stride = encoder_stride   # kept for API compat
        self.decoder_type   = decoder_type
        C = num_channels

        # ── Three independent attention heads ─────────────────────────────────
        self.head_small  = attention_head(C, focus_size=max(2, focus_size // 8))
        self.head_medium = attention_head(C, focus_size=max(4, focus_size // 2))
        self.head_large  = attention_head(C, focus_size=focus_size)

        if encoder_pretrained:
            self._load_pretrained_encoder(encoder_pretrained)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.blend_up_sm  = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.blend_up_med = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.fuse_sm_med = nn.Sequential(
            nn.Conv2d(C + C, C, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, padding=1),     nn.ReLU(inplace=True),
        )

        # ── fuse_med_lg: dense vs sparse ─────────────────────────────────────
        if decoder_type == "dense":
            # All 2C input channels mix freely into C//2 output channels.
            self.fuse_med_lg = nn.Sequential(
                nn.Conv2d(C + C, C // 2, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(C // 2, C // 2, 3, padding=1), nn.ReLU(inplace=True),
            )
            self._routing = None

        else:  # "sparse"
            # Factored: depthwise spatial + sparse pointwise routing.
            #
            # Dense:  Conv2d(2C→C//2, 3×3)  — every input channel talks to every output
            # Sparse: Depthwise(2C, 3×3)    — spatial per-channel  (no cross-channel mix)
            #       + Conv1×1(2C→C//2)      — channel routing       (L1 → sparsity)
            #       + Conv2d(C//2, 3×3)     — local spatial refine
            #
            # The 1×1 routing weight W ∈ R^(C//2 × 2C) is the object of interest:
            # after training, W[i, j] ≈ 0 means output feature i ignores input channel j.
            # Visualising W reveals which attention channels own which output features.
            self._dw     = nn.Sequential(
                nn.Conv2d(C + C, C + C, 3, padding=1, groups=C + C),
                nn.ReLU(inplace=True),
            )
            self._routing = nn.Conv2d(C + C, C // 2, 1, bias=False)
            self._refine  = nn.Sequential(
                nn.Conv2d(C // 2, C // 2, 3, padding=1), nn.ReLU(inplace=True),
            )

        # ── Final prediction ──────────────────────────────────────────────────
        self.pre_masks = nn.Sequential(
            nn.Conv2d(C // 2, C // 2, 3, padding=1), nn.ReLU(inplace=True),
        )
        out_ch     = num_classes if num_classes > 1 else 1
        self.masks = nn.Conv2d(C // 2, out_ch, kernel_size=1)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        H, W = x.shape[2], x.shape[3]

        # Resize to three scales
        x_small  = F.interpolate(x, size=(max(H // 16, 2), max(W // 16, 2)),
                                  mode='bilinear', align_corners=False)
        x_medium = F.interpolate(x, size=(H // 4, W // 4),
                                  mode='bilinear', align_corners=False)

        # Attention at each scale (independent)
        a_small,  _      = self.head_small(x_small)           # (B, C, H/32, W/32)
        a_medium, _      = self.head_medium(x_medium)         # (B, C, H/8,  W/8)
        a_large,  attn_l = self.head_large(x)                 # (B, C, H/2,  W/2)

        # Decoder: bottom-up
        a_s_up = self.blend_up_sm(
            F.interpolate(a_small, size=a_medium.shape[2:],
                          mode='bilinear', align_corners=False))
        d_med  = self.fuse_sm_med(torch.cat([a_s_up, a_medium], dim=1))

        d_m_up = self.blend_up_med(
            F.interpolate(d_med, size=a_large.shape[2:],
                          mode='bilinear', align_corners=False))
        fused  = torch.cat([d_m_up, a_large], dim=1)

        if self.decoder_type == "dense":
            d_lg = self.fuse_med_lg(fused)
        else:
            d_lg = self._refine(F.relu(self._routing(self._dw(fused))))

        d_up   = self.pre_masks(
            F.interpolate(d_lg, size=(H, W), mode='bilinear', align_corners=False))
        logits = self.masks(d_up)

        attn_guide = attn_l.amax(dim=1, keepdim=True)
        attn_guide = F.interpolate(attn_guide, size=(H, W),
                                   mode='bilinear', align_corners=False)
        return logits, attn_guide, attn_l

    def _load_pretrained_encoder(self, path: str) -> None:
        ckpt = torch.load(path, map_location='cpu')
        nc   = ckpt.get('num_channels')
        C    = self.head_large.enc1[0].out_channels
        if nc and nc != C:
            raise ValueError(
                f"Pretrained encoder num_channels={nc} != model num_channels={C}")
        for head in (self.head_small, self.head_medium, self.head_large):
            head.enc1.load_state_dict(ckpt['enc1'])
            head.enc2.load_state_dict(ckpt['enc2'])
        epoch = ckpt.get('epoch', '?')
        acc   = ckpt.get('val_acc', 0.0)
        print(f"Loaded pretrained encoder from {path}  (epoch={epoch}, val_acc={acc:.4f})")

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def routing_sparsity_loss(self) -> torch.Tensor | None:
        """L1 norm of routing weights (sparse variant only).
        Add to total loss with a small weight to encourage channel specialisation.
        Returns None for dense variant.
        """
        if self._routing is None:
            return None
        return self._routing.weight.abs().mean()

    def routing_weight_stats(self) -> dict:
        """Sparsity diagnostics for the routing matrix (sparse variant only).
        Useful for monitoring how many channels are truly specialised.
        Returns empty dict for dense variant.
        """
        if self._routing is None:
            return {}
        W = self._routing.weight.detach().abs()  # (C//2, 2C, 1, 1) → squeeze
        W = W.squeeze(-1).squeeze(-1)            # (C//2, 2C)
        return {
            "routing_mean":     W.mean().item(),
            "routing_sparsity": (W < 1e-3).float().mean().item(),  # fraction near-zero
            "routing_max":      W.max().item(),
        }


def main():
    print("sf_seg — Multi-scale Sparse-Focus Segmentation\n")
    model = sf_seg(num_channels=64, focus_size=32, encoder_stride=2, num_classes=81)
    x     = torch.randn(2, 3, 224, 224)
    logits, attn_guide, attn = model(x)

    print(f"Parameters  : {model.get_num_parameters():,}")
    print(f"Output      : logits {tuple(logits.shape)}")
    print(f"Attn guide  : {tuple(attn_guide.shape)}")
    print()

    rows = [
        ("head_small",  max(224//16,2), max(224//16,2)//2, model.head_small.focus_size),
        ("head_medium", 224//4,         224//8,             model.head_medium.focus_size),
        ("head_large",  224,            224//2,             model.head_large.focus_size),
    ]
    print(f"{'Head':<14} {'Input':>9} {'Attn space':>12} {'k':>6} {'L':>8} {'k/L':>7}")
    print("-" * 58)
    for name, inp, attn_size, fs in rows:
        L = attn_size * attn_size
        k = min(fs * fs, L - 1)
        print(f"{name:<14} {inp:>4}×{inp:<4} {attn_size:>4}×{attn_size:<4}  "
              f"{k:>6} {L:>8} {100*k/L:>6.1f}%")


if __name__ == "__main__":
    main()
