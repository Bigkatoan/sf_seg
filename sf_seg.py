import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, in_channels=3, num_channels=32, focus_size=16):
        super(AttentionBlock, self).__init__()
        self.focus_k = focus_size * focus_size
        # Encoder: width C trong suốt, chỉ expand → 2C ở layer cuối
        # Layer giữa tốn C² thay vì (2C)²=4C² → tiết kiệm ~2× compute
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding='same'),
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

        # Chỉ lấy top-k phần tử — j* không thể lớn hơn k
        top_vals = torch.topk(p, k=int_k, dim=-1).values          # (B, N, k)
        top_sorted = torch.sort(top_vals, dim=-1, descending=True).values  # sort k << L phần tử
        cumsum = top_sorted.cumsum(dim=-1)                         # (B, N, k)

        j = torch.arange(1, int_k + 1, device=score.device, dtype=p.dtype)
        lam_j = (j - cumsum) / (L - j).clamp(min=1e-9)           # λ ứng với j pixel bão hoà

        valid = top_sorted - 1.0 >= lam_j
        j_sat = valid.long().sum(dim=-1, keepdim=True)            # (B, N, 1)

        lam_star = torch.gather(lam_j, dim=-1, index=(j_sat - 1).clamp(min=0))
        lam_star = lam_star * (j_sat > 0).to(p.dtype)

        return (p - lam_star).clamp(0.0, 1.0)

    def forward(self, x):
        out = self.encoder(x)                                      # (B, 2N, H, W)
        score, features = out.chunk(2, dim=1)                      # mỗi (B, N, H, W)
        B, N, H, W = score.shape
        attn = self._clamped_softmax(score.view(B, N, H * W), float(self.focus_k))
        attn = attn.view(B, N, H, W)
        return attn * features, attn

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class sf_seg(nn.Module):
    def __init__(self, num_channels=32, focus_size=16, extra_channels=None):
        """
        num_channels:   số channels của block đầu (nhận RGB)
        focus_size:     budget k = focus_size² cho tất cả các block
        extra_channels: list channels cho các block tiếp theo, ví dụ [16, 16]
                        mỗi block nhận attended_features của block trước làm input
        """
        super(sf_seg, self).__init__()
        extra_channels = list(extra_channels) if extra_channels else []

        channel_list = [num_channels] + extra_channels   # [64, 16, 16]
        in_ch_list   = [3] + channel_list[:-1]           # [3, 64, 16]

        self.blocks = nn.ModuleList([
            AttentionBlock(in_ch, out_ch, focus_size)
            for in_ch, out_ch in zip(in_ch_list, channel_list)
        ])
        self.masks = nn.Conv2d(channel_list[-1], 1, kernel_size=3, padding='same')

    def forward(self, x: torch.Tensor):
        feat = x
        all_attn = []
        for block in self.blocks:
            feat, attn = block(feat)
            all_attn.append(attn)

        masks = torch.sigmoid(self.masks(feat))

        # Gộp attn của tất cả block: (B, sum_C, H, W)
        all_attn_cat = torch.cat(all_attn, dim=1)
        N = all_attn_cat.shape[1]
        attn_guide = all_attn_cat.sum(dim=1, keepdim=True) / N   # (B, 1, H, W) ∈ [0,1]

        return masks, attn_guide, all_attn_cat

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def main():
    print("=== 1 block (baseline) ===")
    m1 = sf_seg(num_channels=64)
    print(f"  params: {m1.get_num_parameters():,}")

    print("=== 3 blocks: 64 → 16 → 16 ===")
    m3 = sf_seg(num_channels=64, extra_channels=[16, 16])
    print(f"  params: {m3.get_num_parameters():,}")

    x = torch.randn(2, 3, 128, 128)
    for name, model in [("1-block", m1), ("3-block", m3)]:
        masks, attn_guide, attn = model(x)
        print(f"  [{name}] masks={masks.shape}  attn_guide={attn_guide.shape}  attn={attn.shape}")

if __name__ == "__main__":
    main()
