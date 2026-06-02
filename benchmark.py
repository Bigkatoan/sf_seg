#!/usr/bin/env python3
"""Benchmark từng phần của sf_seg để tìm bottleneck."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sf_seg import sf_seg, AttentionBlock

B, C, H, W = 64, 128, 128, 128
L = H * W
K = 16 * 16  # focus_k
WARMUP = 5
RUNS = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")


def bench(name, fn, *args):
    # warmup
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(RUNS):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / RUNS
    print(f"  {name:<40s} {ms:7.2f} ms/iter")
    return ms


print("=" * 55)
print("Input shape: (B=%d, C=%d, H=%d, W=%d)" % (B, C, H, W))
print("=" * 55)

x      = torch.randn(B, 3, H, W, device=device)
score  = torch.randn(B, C, L, device=device)
p      = F.softmax(score, dim=-1) * K

# ── 1. Các CUDA op riêng lẻ trong _clamped_softmax ──────────────────────────
print("\n[1] Từng op trong _clamped_softmax  (shape B,C,L = %d,%d,%d)" % (B, C, L))

bench("softmax × k",
      lambda: F.softmax(score, dim=-1) * K)

bench("topk(k=%d) từ L=%d" % (K, L),
      lambda: torch.topk(p, k=K, dim=-1))

top_vals = torch.topk(p, k=K, dim=-1).values
bench("sort(%d phần tử)" % K,
      lambda: torch.sort(top_vals, dim=-1, descending=True))

top_sorted = torch.sort(top_vals, dim=-1, descending=True).values
cumsum     = top_sorted.cumsum(dim=-1)
j_idx      = torch.arange(1, K + 1, device=device, dtype=torch.float32)
lam_j      = (j_idx - cumsum) / (L - j_idx).clamp(min=1e-9)
bench("valid mask + j_sat + gather",
      lambda: (
          (top_sorted - 1.0 >= lam_j).long().sum(-1, keepdim=True)
      ))

bench("_clamped_softmax full",
      lambda: AttentionBlock._clamped_softmax(score, float(K)))

# ── 2. Encoder CNN ───────────────────────────────────────────────────────────
print("\n[2] Encoder CNN  (B=%d, 3, %d, %d)" % (B, H, W))

model = sf_seg(num_channels=C).to(device)

bench("encoder forward only",
      lambda: model.attention_block.encoder(x))

# ── 3. Full model ────────────────────────────────────────────────────────────
print("\n[3] Full model forward")

bench("sf_seg.forward (no grad)",
      lambda: model(x))

with torch.enable_grad():
    x_g = x.requires_grad_(True)
    masks, _, _ = model(x_g)
    loss = masks.mean()
    bench("sf_seg.backward",
          lambda: model(x_g)[0].mean().backward())

print()
