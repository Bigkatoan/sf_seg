# sf_seg — Soft-Focus Segmentation

A lightweight person segmentation model built around a single novel mechanism: **budget attention via clamped softmax**. Instead of a full encoder-decoder (U-Net style), sf_seg uses a shallow CNN whose attention head is mathematically constrained to focus on at most `k = focus_size²` pixels per channel.

---

## How It Works

### 1. Encoder

```
Input  (B, 3, H, W)
  Conv2d(3  → C, 3×3, stride=encoder_stride) + ReLU
  Conv2d(C  → C, 3×3) + ReLU
  Conv2d(C  → 2C, 3×3)          ← no activation; channel expand only at the end
```

The encoder widens to `2C` only at the last layer. This keeps intermediate layers at cost `C²` instead of `(2C)² = 4C²`, saving ~2× compute compared to expanding early.

With `encoder_stride=2` (default), the encoder operates at half resolution `(H/2 × W/2)`, giving a **~2.9× speedup** with the same parameter count. Logits are upsampled back to full resolution before sigmoid — gradient flows through unbounded logits, not through a saturated sigmoid at low resolution.

The output is split into two tensors of shape `(B, C, H', W')`:
- **score** — used to compute where to attend
- **features** — the actual content that gets weighted

---

### 2. Clamped Softmax (Budget Attention)

This is the core contribution. Standard softmax distributes attention uniformly across all pixels, making it hard to learn spatial focus. Clamped softmax enforces:

> Each channel attends to **exactly k pixels** in total, with each pixel weight in **[0, 1]**.

**Algorithm** (closed-form, no Python loops, pure CUDA ops):

```
k = focus_size²                           # e.g. 32² = 1024 "focus units"
L = H' × W'                              # total pixels at encoder resolution

p = softmax(score, dim=pixels) × k        # shape (B, C, L), sum = k per channel

# Find saturation threshold λ* in O(k log k) instead of O(L log L):
top_vals  = topk(p, k=k)                  # only top-k can possibly saturate
top_sorted = sort(top_vals, descending)
cumsum    = cumsum(top_sorted)
lam_j     = (j - cumsum) / (L - j)        # λ if exactly j pixels saturate at 1.0
j_sat     = count(top_sorted - 1 >= lam_j)  # actual number of saturated pixels
λ*        = lam_j[j_sat - 1]

attn = clamp(p - λ*, 0, 1)               # ∈ [0,1], sum = k exactly
```

**Why this works:**
- `p` from softmax×k already sums to `k`
- Any pixel with `p_i > 1` gets clamped to `1` (saturated), its excess redistributed via `λ*` to the remaining pixels
- The threshold `λ*` is the unique value that makes the redistribution exact
- **Proof that j\* ≤ k:** if j\* > k, then sum ≥ j\* > k, contradicting sum = k

**Result:** attended_features = attn × features — only the `k` focused pixels contribute meaningfully.

---

### 3. Segmentation Head

```
attended_features  (B, C, H', W')
  → Conv2d(C → 1, 3×3, padding='same')
  → interpolate to (H, W)  [if encoder_stride > 1]
  → Sigmoid
  → mask  (B, 1, H, W)  ∈ [0, 1]
```

The model also returns an **attention guide** for visualization:
```
attn_guide = max(attn, dim=channels)   ∈ [0, 1]
```
A pixel reaching `1.0` means at least one channel attended to it with maximum weight. Using max instead of mean preserves sparsity: with diverse channels, the mean collapses to a uniform constant (~1/L × k), while max retains the actual coverage pattern.

---

### 4. Loss Functions

**Segmentation Loss** (main objective, `--loss-type`):

| Type | Formula |
|---|---|
| `iou` | `1 - (Σ pred·target + ε) / (Σ pred + Σ target - inter + ε)` |
| `mse` | `mean((pred - target)²)` |
| `combine` | `0.1 × IoU + 0.9 × MSE` |
| `bce` | binary cross-entropy |
| `bce_iou` | `BCE + IoU` |

**Foreground Attention Loss** (auxiliary, cooperative with diversity):
```
# Resize GT mask to attention resolution (H', W')
masks_a = interpolate(GT_mask, size=(H', W'), mode='nearest')

# Per-channel fraction of attention budget spent on foreground
fg_ratio[b, c] = Σ(attn[b,c] × masks_a[b]) / Σ(attn[b,c])   # ∈ [0, 1]

loss_attn = 1 - mean(fg_ratio)   # → 0 when every channel focuses on the person
total_loss = main_loss + attn_guide_weight × loss_attn
```
This loss directly measures what fraction of each channel's focus budget falls on the foreground object. It **cooperates** with the diversity loss: diverse channels can cover different parts of the person simultaneously, each contributing to a high fg_ratio.

**Attention Diversity Loss** (auxiliary, regulariser):
```
A  = attn.view(B, C, L)                    # (B, C, L), L = H'×W'
A  = normalize(A, dim=-1)                   # unit-norm per channel
G  = A @ Aᵀ                                # (B, C, C) cosine-similarity Gram matrix
loss_div = mean(off_diag(G)²) / C(C-1)

total_loss = main_loss + diversity_weight × loss_div
```
Penalises cosine similarity between every pair of channels, pushing them to specialise on different spatial regions.

- **loss_div = 0** → all C channels are perfectly orthogonal (ideal)
- **loss_div = 1** → all channels are identical (worst case)

**Why the two auxiliary losses work together:**
The foreground loss wants each channel to attend to the person. The diversity loss wants channels to attend to different regions. Combined: different channels cover different parts of the person — exactly what we want for rich multi-channel attention.

`sf_seg.forward()` returns `(masks, attn_guide, attn)` — the raw `attn` tensor `(B, C, H', W')` is needed to compute both auxiliary losses.

---

### 5. Training Pipeline

| Component | Choice | Reason |
|---|---|---|
| Optimizer | Adam, lr=1e-4 | Adaptive LR, fast convergence |
| LR schedule | CosineAnnealingLR (eta_min = lr × 0.01) | Avoids plateau; smooth decay to near-zero |
| Augmentation | Random horizontal flip (p=0.5) | Doubles effective dataset, improves generalization |
| Mixed precision | AMP (autocast + GradScaler) | 2× speed, ~half VRAM on CUDA |
| DataLoader | pin_memory, non_blocking, prefetch_factor=2 | CPU→GPU transfer overlap |
| Mask resize | NEAREST neighbor | Avoids interpolated non-binary values |
| Image resize | BILINEAR | Better visual quality for RGB |
| Gradient zero | set_to_none=True | Frees memory instead of zero-fill |
| Validation metric | Pixel accuracy + loss | Simple, interpretable |
| Checkpointing | best (val_loss) + last | Safe resume; preserves optimizer + scheduler state |

---

## Model Size

With current config (`num_channels=64`, `image_size=224`, `encoder_stride=2`, `focus_size=32`):
- **Parameters:** ~113k
- **Attention resolution:** 112×112 (half of input)
- **Budget k:** 1024 pixels per channel (= 32²), covering ~8.2% of the attention map

---

## Quick Start

```bash
pip install -r requirements.txt
```

### COCO 2017 (full dataset)
```bash
python download.py --root data --prepare        # requires aria2c for speed
./train.sh
```

### Resume training
```bash
python train_sf_seg.py --resume last
```

---

## Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--num-channels` | 64 | Feature channels C |
| `--focus-size` | 32 | Budget k = focus_size² pixels / channel |
| `--encoder-stride` | 2 | First conv stride (1 = full res, 2 = half res ~3× faster) |
| `--loss-type` | combine | `iou` / `bce` / `mse` / `combine` / `bce_iou` |
| `--attn-guide-weight` | 0.4 | Weight of foreground attention loss (0 = off) |
| `--diversity-weight` | 0.1 | Weight of attention diversity / Gram penalty (0 = off) |
| `--image-size` | 224 | Square input resolution |
| `--lr` | 1e-4 | Adam learning rate (cosine decay to lr × 0.01) |
| `--batch-size` | 32 | Batch size |
| `--epochs` | 100 | Training epochs |

All defaults can be set in `config.json`.

The tqdm progress bar and log file show each loss component and current LR:
```
lr | total | seg | attn | div | acc
```

---

## Visualize Attention

```bash
python visualize_attention.py \
    --checkpoint checkpoints/sf_seg_best.pt \
    --num-images 6 \
    --show-channels 8 \
    --min-range 0.05
```

Each output image shows 4 + N panels:
```
[Input RGB] [GT Mask] [Predicted Mask] [Attention Max] [Ch_i × N]
```

Channels are filtered by **min-max range** (`max − min` over spatial dims):
- **Range low** → the pattern this channel learned does not appear in this image; attention spreads uniformly with no peak — skipped
- **Range high** → the channel found its pattern in this image; clear bright spot present

Channels passing the threshold are sorted by range descending (sharpest first). Each panel title shows `rng` and `max`.

Adjust `--min-range` (default `0.05`): raise to `0.1–0.2` for only the sharpest channels, lower to `0.01` to see all.

---

## Benchmark

```bash
python benchmark.py   # requires CUDA
```

Measures latency of each sub-operation in clamped softmax and the full forward/backward pass.

---

## File Structure

```
sf_seg/
├── sf_seg.py              # Model: AttentionBlock + clamped softmax
├── train_sf_seg.py        # Training script with AMP, checkpointing, logging
├── losses.py              # IoU / BCE / MSE / combined / diversity loss functions
├── visualize_attention.py # Attention map visualization
├── benchmark.py           # Per-op latency benchmark
├── download.py            # COCO 2017 download + mask preparation
├── config.json            # Default hyperparameters
├── requirements.txt       # Python dependencies
└── train.sh               # Convenience wrapper: ./train.sh [extra args]
```
