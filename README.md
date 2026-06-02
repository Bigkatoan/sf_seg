# sf_seg — Soft-Focus Segmentation

A lightweight person segmentation model built around a single novel mechanism: **budget attention via clamped softmax**. Instead of a full encoder-decoder (U-Net style), sf_seg uses a shallow CNN whose attention head is mathematically constrained to focus on at most `k = focus_size²` pixels per channel.

---

## How It Works

### 1. Encoder

```
Input  (B, 3, H, W)
  Conv2d(3  → C, 3×3) + ReLU
  Conv2d(C  → C, 3×3) + ReLU
  Conv2d(C  → 2C, 3×3)          ← no activation; channel expand only at the end
```

The encoder widens to `2C` only at the last layer. This keeps intermediate layers at cost `C²` instead of `(2C)² = 4C²`, saving ~2× compute compared to expanding early.

The output is split into two tensors of shape `(B, C, H, W)`:
- **score** — used to compute where to attend
- **features** — the actual content that gets weighted

---

### 2. Clamped Softmax (Budget Attention)

This is the core contribution. Standard softmax distributes attention uniformly across all pixels, making it hard to learn spatial focus. Clamped softmax enforces:

> Each channel attends to **exactly k pixels** in total, with each pixel weight in **[0, 1]**.

**Algorithm** (closed-form, no Python loops, pure CUDA ops):

```
k = focus_size²                           # e.g. 16² = 256 "focus units"
L = H × W                                 # total pixels, e.g. 128² = 16384

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
attended_features  (B, C, H, W)
  → Conv2d(C → 1, 3×3, padding='same')
  → Sigmoid
  → mask  (B, 1, H, W)  ∈ [0, 1]
```

Additionally, the model returns an **attention guide**:
```
attn_guide = sum(attn, dim=channels) / C   ∈ [0, 1]
```
A pixel reaching `1.0` means every channel attended to it with maximum weight. This is used as a training signal (see Loss section).

---

### 4. Loss Functions

Four losses are available, combinable via `--loss-type`:

**IoU Loss** (default):
```
inter = Σ(pred × target)
union = Σ(pred) + Σ(target) - inter
loss  = 1 - (inter + ε) / (union + ε)
```
Directly optimizes the evaluation metric. Robust to class imbalance.

**BCE Loss:** standard binary cross-entropy per pixel.

**MSE Loss:** mean squared error between predicted probability and binary GT.

**Combined Loss** (used in `config.json`):
```
loss = 0.1 × IoU_loss + 0.9 × MSE_loss
```
IoU ensures global shape correctness; MSE provides dense, smooth gradients.

**Attention Guidance Loss** (auxiliary):
```
# Build soft target from GT mask using separable Gaussian blur:
attn_target = gaussian_blur(GT_mask, kernel=31, sigma=7.0)
attn_target = normalize_per_sample(attn_target)   # peak = 1.0

# Auxiliary loss:
total_loss = main_loss + 0.3 × IoU_loss(attn_guide, attn_target)
```
The GT mask is blurred (not used binary) so attention only needs to focus "approximately" on the person, not pixel-perfectly. This avoids sparse gradients and makes attention training stable.

**Attention Diversity Loss** (auxiliary):
```
A  = attn.view(B, C, L)                    # (B, C, L), L = H×W
A  = normalize(A, dim=-1)                   # unit-norm per channel
G  = A @ Aᵀ                                # (B, C, C) cosine-similarity Gram matrix
loss_div = mean(off_diag(G)²) / C(C-1)

total_loss = main_loss + 0.1 × loss_div
```
Without this loss, channels tend to attend to the same regions (all variance ~0.007). The Gram matrix penalty directly minimises the cosine similarity between every pair of channels, pushing them to specialise on different spatial regions.

- **loss_div = 0** → all C channels are perfectly orthogonal (ideal)
- **loss_div = 1** → all channels are identical (worst case)

`sf_seg.forward()` returns `(masks, attn_guide, attn)` — the raw `attn` tensor `(B, C, H, W)` is needed to compute this loss.

---

### 5. Training Pipeline

| Component | Choice | Reason |
|---|---|---|
| Optimizer | Adam, lr=1e-4 | Adaptive LR, fast convergence |
| Mixed precision | AMP (autocast + GradScaler) | 2× speed, ~half VRAM on CUDA |
| DataLoader | pin_memory, non_blocking, prefetch_factor=2 | CPU→GPU transfer overlap |
| Mask resize | NEAREST neighbor | Avoids interpolated non-binary values |
| Image resize | BILINEAR | Better visual quality for RGB |
| Gradient zero | set_to_none=True | Frees memory instead of zero-fill |
| Validation metric | Pixel accuracy + loss | Simple, interpretable |
| Checkpointing | best (val_loss) + last | Safe resume; preserves optimizer state |

---

## Model Size

With default `num_channels=64`:
- **Parameters:** ~113k
- **Input:** 128×128 RGB
- **Budget k:** 256 pixels per channel (= 16²)

With `num_channels=128`: ~425k parameters.

---

## Quick Start

```bash
pip install -r requirements.txt
```

### Synthetic data (no COCO needed)
```bash
python prepare_toy_data.py --root data --train 200 --val 40
python train_sf_seg.py
```

### COCO 2017 (full dataset)
```bash
python download.py --root data --prepare        # requires aria2c for speed
python train_sf_seg.py --epochs 100 --loss-type combine
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
| `--focus-size` | 16 | Budget k = focus_size² pixels / channel |
| `--loss-type` | combine | `iou` / `bce` / `mse` / `combine` / `bce_iou` |
| `--attn-guide-weight` | 0.3 | Weight of attention guidance auxiliary loss (0 = off) |
| `--attn-blur-sigma` | 7.0 | Gaussian blur sigma for soft attention target |
| `--attn-blur-kernel` | 31 | Gaussian blur kernel size (must be odd) |
| `--diversity-weight` | 0.1 | Weight of attention diversity loss / Gram penalty (0 = off) |
| `--image-size` | 128 | Square input resolution |
| `--lr` | 1e-4 | Adam learning rate |
| `--batch-size` | 64 | Batch size |
| `--epochs` | 100 | Training epochs |

All defaults can be set in `config.json`.

The tqdm progress bar and log file show each loss component separately:
```
total | seg | attn | div | acc
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
[Input RGB] [GT Mask] [Predicted Mask] [Attention Sum] [Ch_i × N]
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
├── losses.py              # IoU / BCE / MSE / combined loss functions
├── visualize_attention.py # Attention map visualization
├── benchmark.py           # Per-op latency benchmark
├── download.py            # COCO 2017 download + mask preparation
├── prepare_toy_data.py    # Synthetic dataset generator (no COCO needed)
├── config.json            # Default hyperparameters
├── requirements.txt       # Python dependencies
└── train.sh               # Convenience wrapper: ./train.sh [extra args]
```
