import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, num_channels=32, focus_size=16, encoder_stride=1):
        super(AttentionBlock, self).__init__()
        self.focus_k = focus_size * focus_size
        # encoder_stride=2: first conv halves H×W, subsequent convs work on smaller maps
        # padding='same' không hỗ trợ stride>1 → dùng padding=1 tường minh (equiv cho 3×3, stride=1)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding=1, stride=encoder_stride),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1),
        )

    @staticmethod
    def _clamped_softmax(score: torch.Tensor, k: float) -> torch.Tensor:
        """
        Closed-form: mỗi giá trị trong [0,1], tổng = k.

        Proof: j* ≤ k (nếu j* > k thì sum ≥ j* > k, mâu thuẫn).
        → Chỉ cần topk(k) thay vì sort(L): k log k << L log L.
        Toàn bộ là CUDA ops đơn lẻ, không có Python for loop.
        """
        L = score.shape[-1]
        int_k = int(k)
        p = F.softmax(score, dim=-1) * k                          # (B, N, L), sum = k

        top_vals = torch.topk(p, k=int_k, dim=-1).values          # (B, N, k)
        top_sorted = torch.sort(top_vals, dim=-1, descending=True).values
        cumsum = top_sorted.cumsum(dim=-1)                         # (B, N, k)

        j = torch.arange(1, int_k + 1, device=score.device, dtype=p.dtype)
        lam_j = (j - cumsum) / (L - j).clamp(min=1e-9)

        valid = top_sorted - 1.0 >= lam_j
        j_sat = valid.long().sum(dim=-1, keepdim=True)            # (B, N, 1)

        lam_star = torch.gather(lam_j, dim=-1, index=(j_sat - 1).clamp(min=0))
        lam_star = lam_star * (j_sat > 0).to(p.dtype)

        return (p - lam_star).clamp(0.0, 1.0)

    def forward(self, x):
        out = self.encoder(x)                                      # (B, 2N, H', W')
        score, features = out.chunk(2, dim=1)
        B, N, H, W = score.shape
        attn = self._clamped_softmax(score.view(B, N, H * W), float(self.focus_k))
        attn = attn.view(B, N, H, W)
        return attn * features, attn

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class sf_seg(nn.Module):
    def __init__(self, num_channels=32, focus_size=16, encoder_stride=1):
        """
        encoder_stride=2: encoder works at H/2 × W/2, logits upsampled to input
        resolution BEFORE sigmoid — gradient flows through unbounded logits, not
        through the saturating sigmoid nonlinearity at low resolution.

        focus_size=16 recommended with encoder_stride=2:
          k = 16² = 256 pixel coverage per channel (same as stride=1)
          k/L = 256/4096 = 6.25%  (denser than stride=1, but absolute k preserved)
        """
        super(sf_seg, self).__init__()
        self.encoder_stride = encoder_stride
        self.attention_block = AttentionBlock(
            num_channels=num_channels,
            focus_size=focus_size,
            encoder_stride=encoder_stride,
        )
        self.masks = nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        H, W = x.shape[2], x.shape[3]
        attended_features, attn = self.attention_block(x)   # (B, N, H', W')

        logits = self.masks(attended_features)               # (B, 1, H', W') — raw, unbounded

        # Upsample logits trước sigmoid: gradient qua logit (unbounded) mượt hơn
        # qua sigmoid value (bounded [0,1]) đã bị squash ở resolution thấp
        if self.encoder_stride > 1:
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        masks = torch.sigmoid(logits)                        # sigmoid ở full resolution

        N = attn.shape[1]
        attn_guide = attn.sum(dim=1, keepdim=True) / N
        if self.encoder_stride > 1:
            attn_guide = F.interpolate(attn_guide, size=(H, W), mode='bilinear', align_corners=False)

        return masks, attn_guide, attn

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def main():
    for stride, fs in [(1, 16), (2, 16)]:
        model = sf_seg(num_channels=64, focus_size=fs, encoder_stride=stride)
        x = torch.randn(2, 3, 128, 128)
        masks, attn_guide, attn = model(x)
        L = attn.shape[2] * attn.shape[3]
        k = fs * fs
        print(f"stride={stride} focus_size={fs}: params={model.get_num_parameters():,} "
              f"attn={attn.shape} masks={masks.shape} k={k} k/L={k/L:.2%}")

if __name__ == "__main__":
    main()
