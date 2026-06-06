"""sf_seg với ResNet-18 backbone (ImageNet pretrained).

Khác với sf_seg.py (custom backbone):
  - Backbone: torchvision ResNet-18, load weights ImageNet-1K sẵn
  - Feature channels: 64 / 128 / 256 → adapter 1×1 conv về C
  - Stem: Conv(7×7, s=2) + BN + ReLU + MaxPool → H/4
  - BatchNorm trong ResNet gốc, GroupNorm + GELU trong adapter/head/decoder
  - Backbone params ~7M (layer1-3); tổng ~10M
  - Không cần chạy pretrain_encoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers (tự chứa để không phụ thuộc sf_seg.py) ───────────────────────────

def _gn(C: int) -> nn.GroupNorm:
    for g in [8, 4, 2, 1]:
        if C % g == 0:
            return nn.GroupNorm(g, C)


def _clamped_softmax(score: torch.Tensor, k: float) -> torch.Tensor:
    L     = score.shape[-1]
    int_k = int(k)
    p     = F.softmax(score, dim=-1) * k
    top_vals   = torch.topk(p, k=int_k, dim=-1).values
    top_sorted = torch.sort(top_vals, dim=-1, descending=True).values
    cumsum     = top_sorted.cumsum(dim=-1)
    j      = torch.arange(1, int_k + 1, device=score.device, dtype=p.dtype)
    lam_j  = (j - cumsum) / (L - j).clamp(min=1e-9)
    valid  = top_sorted - 1.0 >= lam_j
    j_sat  = valid.long().sum(dim=-1, keepdim=True)
    lam_star = torch.gather(lam_j, dim=-1, index=(j_sat - 1).clamp(min=0))
    lam_star = lam_star * (j_sat > 0).to(p.dtype)
    return (p - lam_star).clamp(0.0, 1.0)


class attention_head(nn.Module):
    def __init__(self, num_channels: int, focus_size: int):
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
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor):
        score, features = self.proj(x).chunk(2, dim=1)
        B, C, H, W = score.shape
        k    = min(self.focus_size ** 2, H * W - 1)
        attn = _clamped_softmax(score.view(B, C, H * W), float(k)).view(B, C, H, W)
        return self.channel_mix(attn * features), attn


# ── ResNet-18 backbone wrapper ────────────────────────────────────────────────

def _build_resnet18(pretrained: bool):
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        return resnet18(weights=weights)
    except ImportError:
        raise ImportError("pip install torchvision")


# ── Model ─────────────────────────────────────────────────────────────────────

class sf_seg(nn.Module):
    """
    Multi-scale Sparse-Focus Segmentation với ResNet-18 backbone.

    ResNet-18 feature map sizes (input 224×224):
    ┌──────────────┬──────────┬────────┬──────────────────────┐
    │ ResNet layer │ Channels │ Size   │ Head                 │
    ├──────────────┼──────────┼────────┼──────────────────────┤
    │ layer1       │ 64       │ H/4    │ adapt_large→head_large  │
    │ layer2       │ 128      │ H/8    │ adapt_medium→head_medium│
    │ layer3       │ 256      │ H/16   │ adapt_small→head_small  │
    └──────────────┴──────────┴────────┴──────────────────────┘

    Adapter: Conv(in→C, 1×1) + GN + GELU — unify channel count trước khi vào head.
    Decoder: giống sf_seg.py (GN + GELU).
    """

    # ResNet-18 output channels tại layer1/2/3
    _R18_CHANNELS = (64, 128, 256)

    def __init__(self, num_channels: int = 32, focus_size: int = 32,
                 encoder_stride: int = 2, num_classes: int = 1,
                 decoder_type: str = "dense",
                 encoder_pretrained: str | None = None,
                 pretrained_backbone: bool = True):
        super().__init__()
        self.num_classes    = num_classes
        self.encoder_stride = encoder_stride
        self.decoder_type   = decoder_type
        C = num_channels

        # ── ResNet-18 backbone ─────────────────────────────────────────────────
        r18 = _build_resnet18(pretrained=pretrained_backbone)
        # Stem: conv1(7×7,s=2) + bn1 + relu + maxpool(s=2) → (64, H/4)
        self.r18_stem   = nn.Sequential(r18.conv1, r18.bn1, r18.relu, r18.maxpool)
        self.r18_layer1 = r18.layer1   # (64,  H/4)
        self.r18_layer2 = r18.layer2   # (128, H/8)
        self.r18_layer3 = r18.layer3   # (256, H/16)
        # layer4 không dùng — bỏ để tiết kiệm memory

        # ── Adapter: project về C channels ────────────────────────────────────
        ch_l, ch_m, ch_s = self._R18_CHANNELS
        self.adapt_large  = nn.Sequential(
            nn.Conv2d(ch_l, C, 1, bias=False), _gn(C), nn.GELU(),
        )
        self.adapt_medium = nn.Sequential(
            nn.Conv2d(ch_m, C, 1, bias=False), _gn(C), nn.GELU(),
        )
        self.adapt_small  = nn.Sequential(
            nn.Conv2d(ch_s, C, 1, bias=False), _gn(C), nn.GELU(),
        )

        # ── Attention heads ────────────────────────────────────────────────────
        self.head_small  = attention_head(C, focus_size=max(2, focus_size // 8))
        self.head_medium = attention_head(C, focus_size=max(4, focus_size // 2))
        self.head_large  = attention_head(C, focus_size=focus_size)

        # ── Decoder (giống sf_seg.py) ──────────────────────────────────────────
        self.blend_up_sm  = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, bias=False), _gn(C), nn.GELU(),
        )
        self.blend_up_med = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, bias=False), _gn(C), nn.GELU(),
        )
        self.fuse_sm_med = nn.Sequential(
            nn.Conv2d(C + C, C, 3, padding=1, bias=False), _gn(C), nn.GELU(),
            nn.Conv2d(C, C, 3, padding=1, bias=False),     _gn(C), nn.GELU(),
            nn.Dropout2d(0.1),
        )
        self.fuse_med_lg = nn.Sequential(
            nn.Conv2d(C + C, C // 2, 3, padding=1, bias=False), _gn(C // 2), nn.GELU(),
            nn.Conv2d(C // 2, C // 2, 3, padding=1, bias=False), _gn(C // 2), nn.GELU(),
            nn.Dropout2d(0.1),
        )
        self.pre_masks = nn.Sequential(
            nn.Conv2d(C // 2, C // 2, 3, padding=1, bias=False), _gn(C // 2), nn.GELU(),
        )
        out_ch = num_classes if num_classes > 1 else 1
        self.masks = nn.Conv2d(C // 2, out_ch, kernel_size=1)

        if encoder_pretrained:
            print(f"[r18] encoder_pretrained ignored — backbone uses ImageNet weights")

    # ── Encode ────────────────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor):
        """Return adapted features at H/4, H/8, H/16."""
        f       = self.r18_stem(x)           # (B,  64, H/4)
        raw_l   = self.r18_layer1(f)         # (B,  64, H/4)
        raw_m   = self.r18_layer2(raw_l)     # (B, 128, H/8)
        raw_s   = self.r18_layer3(raw_m)     # (B, 256, H/16)
        return (self.adapt_large(raw_l),
                self.adapt_medium(raw_m),
                self.adapt_small(raw_s))

    # ── Forward ───────────────────────────────────────────────────────────────

    def extract_features(self, x: torch.Tensor, full_res: bool = True):
        H, W = x.shape[2], x.shape[3]
        f_l, f_m, f_s = self._encode(x)
        a_large,  attn_l = self.head_large(f_l)
        a_medium, _      = self.head_medium(f_m)
        a_small,  _      = self.head_small(f_s)

        a_s_up = self.blend_up_sm(
            F.interpolate(a_small, size=a_medium.shape[2:],
                          mode='bilinear', align_corners=False))
        d_med  = self.fuse_sm_med(torch.cat([a_s_up, a_medium], dim=1))

        d_m_up = self.blend_up_med(
            F.interpolate(d_med, size=a_large.shape[2:],
                          mode='bilinear', align_corners=False))
        d_lg   = self.fuse_med_lg(torch.cat([d_m_up, a_large], dim=1))

        if not full_res:
            return d_lg, attn_l

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

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def routing_sparsity_loss(self):
        return None

    def routing_weight_stats(self):
        return {}


def main():
    print("sf_seg_r18 — ResNet-18 backbone (ImageNet pretrained)\n")
    model = sf_seg(num_channels=128, focus_size=28, num_classes=151,
                   pretrained_backbone=True)
    x = torch.randn(2, 3, 224, 224)
    logits, attn_guide, attn = model(x)

    total = model.get_num_parameters()
    r18   = sum(p.numel() for m in [model.r18_stem, model.r18_layer1,
                                     model.r18_layer2, model.r18_layer3]
                for p in m.parameters())
    adpt  = sum(p.numel() for m in [model.adapt_large, model.adapt_medium,
                                     model.adapt_small]
                for p in m.parameters())
    heads = sum(p.numel() for m in [model.head_small, model.head_medium,
                                     model.head_large]
                for p in m.parameters())
    print(f"Total params      : {total:,}")
    print(f"  ResNet-18 (1-3) : {r18:,}")
    print(f"  adapters        : {adpt:,}")
    print(f"  attn heads      : {heads:,}")
    print(f"  decoder+cls     : {total - r18 - adpt - heads:,}")
    print(f"Logits            : {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
