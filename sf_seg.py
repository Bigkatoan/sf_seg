import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, num_channels=32, focus_size=16):
        super(AttentionBlock, self).__init__()
        self.focus_k = focus_size * focus_size
        # Encoder: width C trong suốt, chỉ expand → 2C ở layer cuối
        # Layer giữa tốn C² thay vì (2C)²=4C² → tiết kiệm ~2× compute
        self.encoder = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=3, padding='same'),
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
    def __init__(self, num_channels=32, focus_size=16):
        super(sf_seg, self).__init__()
        self.attention_block = AttentionBlock(num_channels=num_channels, focus_size=focus_size)
        self.masks = nn.Conv2d(in_channels=num_channels, out_channels=1, kernel_size=3, padding='same')

    def forward(self, x: torch.Tensor):
        attended_features, attn = self.attention_block(x)   # attn: (B, N, H, W)
        masks = torch.sigmoid(self.masks(attended_features))
        # attn_guide: normalize về [0,1] bằng cách chia N channels
        # → pixel nào được tất cả channel attend (weight=1) sẽ đạt 1.0
        N = attn.shape[1]
        attn_guide = attn.sum(dim=1, keepdim=True) / N      # (B, 1, H, W) ∈ [0,1]
        return masks, attn_guide, attn

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
def main():
    model = sf_seg(num_channels=128)
    print(f"Number of parameters: {model.get_num_parameters():,}")
    x = torch.randn(1, 3, 128, 128)
    masks, attn_guide, attn = model(x)
    print(f"Mask shape:      {masks.shape}")
    print(f"Attn guide shape:{attn_guide.shape}")
    print(f"Attn shape:      {attn.shape}")

if __name__ == "__main__":
    main()