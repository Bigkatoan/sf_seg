# sf_seg — Multi-scale Sparse-Focus Segmentation

Lightweight semantic segmentation built around **budget-constrained spatial attention via clamped softmax**. Three independent attention heads operate at different image scales; outputs are fused bottom-up in a UNet-style decoder with learned blending at every transition.

Dataset: **ADE20K-150** (150 semantic categories + background).

---

## Architecture

### Overview

```
Input x (B, 3, 224, 224)
  │
  ├─ resize → 14×14   → head_small  → a_small  (B, C, 7,   7  )   global context  (33% coverage)
  ├─ resize → 56×56   → head_medium → a_medium (B, C, 28,  28 )   mid-range       (33% coverage)
  └─ full res 224×224 → head_large  → a_large  (B, C, 112, 112)   fine detail     ( 8% coverage)
                                              attn_l ↗

Decoder (bottom-up):
  a_small  (B, C, 7, 7)
    ↓ upsample ×4 → blend_up_sm (3×3 conv)
    cat[↑, a_medium]  → fuse_sm_med (3×3 × 2)  →  d_med  (B, C,   28,  28)
    ↓ upsample ×4 → blend_up_med (3×3 conv)
    cat[↑, a_large]   → fuse_med_lg (3×3 × 2)  →  d_lg   (B, C/2, 112, 112)
    ↓ upsample ×2 → pre_masks (3×3 conv)
    → masks (1×1 conv) → logits  (B, 151, 224, 224)

Forward returns:
  logits     (B, 151, 224, 224)   segmentation logits
  attn_guide (B,   1, 224, 224)   amax(attn_l) upsampled — visualisation
  attn_l     (B,   C, 112, 112)   raw large-head attention — used for all losses
```

### Attention Head

Each of the three heads is **independent** (no shared weights):

```
x_scale  (B, 3, H_s, W_s)
  Conv2d(3→C, 3×3, stride=2) + ReLU          → (B,  C, H_s/2, W_s/2)
  Conv2d(C→2C, 3×3)                           → (B, 2C, H_s/2, W_s/2)
  chunk(2) ──→ score    (B, C, L)
             → features (B, C, L)    L = H_s/2 × W_s/2

  attn     = clamped_softmax(score, k)    # sparse, ∈ [0,1], Σ=k per channel
  attended = channel_mix(attn × features) # 1×1 conv cross-channel blend
```

### Clamped Softmax (Budget Attention)

Each channel attends to **exactly k locations**, each weight in **[0, 1]**:

```
k = min(focus_size², L − 1)
p = softmax(score) × k              # sum = k, values may exceed 1
# Closed-form Lagrangian solution via topk in O(k log k):
attn = clamp(p − λ*, 0, 1)          # sum = k, each value ∈ [0, 1]
```

### Scale Table (image_size=224, focus_size=32, C=128)

| Head | Input | Attn space | k | L | Coverage |
|:---|:---:|:---:|---:|---:|:---:|
| `head_small` | 14×14 | 7×7 | 16 | 49 | 33% |
| `head_medium` | 56×56 | 28×28 | 256 | 784 | 33% |
| `head_large` | 224×224 | 112×112 | 1024 | 12544 | 8% |

### Decoder Types

| Type | `fuse_med_lg` | Description |
|---|---|---|
| `dense` *(default)* | Conv2d(2C→C/2, 3×3) × 2 | All channels mix freely |
| `sparse` | DW-Conv(2C, 3×3) + Conv1×1(2C→C/2) + Conv(C/2, 3×3) | Spatial per-channel + learned sparse routing with L1 penalty |

Sparse variant exposes a routing weight matrix **W ∈ ℝ^(C/2 × 2C)**: after training, `W[i,j] ≈ 0` means output feature `i` ignores input channel `j` — visualisable channel-to-class assignment.

---

## Loss Functions

Total loss per training step:

```
L = L_seg  +  diversity_weight × L_diversity
           +  attn_guide_weight × L_guide
           +  attn_exclusive_weight × L_exclusive
           [ +  sparse_weight × L_routing   (sparse decoder only) ]
```

### 1. `L_seg` — `pure_focal_iou` *(default)*

Unified Focal-IoU with Spatial Mass Penalty — two objectives in one:

**Present classes** (appear in the GT mask):
```
L_present = (1/N_present) × Σ_{c∈present} (1 − IoU_c)^(γ+1)
```
- Derived from focal loss by replacing `p_t → IoU_c`, `CE → 1−IoU_c`
- γ=2 → exponent = 3 (cubic): easy classes (high IoU) get near-zero weight
- Directly optimises the evaluation metric (IoU), no CE surrogate

**Absent classes** (not in the GT mask):
```
L_absent = absent_weight × (1/N_absent) × Σ_{c∈absent} mean_{x,y}(p_c(x,y))
```
- Pixel-level spatial constraint: model must allocate zero probability mass anywhere for absent classes
- Stronger than IoU-based `no_obj_weight` (which only sees class-level overlap)

Other available loss types: `ce_iou`, `focal_iou`, `focal`, `ce`, `iou`.

### 2. `L_diversity` — Attention Diversity

```
A = L2-normalize(attn_l, dim=-1)       # (B, C, L) unit vectors
G = A @ Aᵀ                             # (B, C, C) cosine similarity
L_diversity = mean(off_diagonal(G)²) / (C × (C−1))
```

Penalises cosine similarity between attention channels — pushes each channel to detect distinct spatial patterns.

### 3. `L_guide` — IoU-Guided Attention Supervision

For each present foreground class `c`:

```
1. Compute IoU(attn_k, GT_mask_c) for all k          [torch.no_grad()]
2. winner k* = argmax_k IoU                           [detached selector]
3. Dice loss: push attn_{k*} to match GT_mask_c       [gradient flows here]
```

Spatial downsampling to 14×14 before bmm keeps compute tractable (~60M ops/batch vs 3.9B at full res). Gradient only flows through the Dice term — selection is stable.

### 4. `L_exclusive` — Winner-Takes-All Specialisation

```
winner_mask  = (iou_per_channel == max_iou).float()   [detached]
non_winner   = iou_per_channel × (1 − winner_mask)
L_exclusive  = mean(non_winner_iou)
```

Paired with `L_guide` creates competitive dynamics:
- `L_guide`: winner channel → **match** its class GT mask
- `L_exclusive`: non-winner channels → **zero overlap** with other classes

After training, the winner assignment matrix (channel → class) makes attention maps usable as standalone class-specific encoders.

### 5. `L_routing` — Sparse Decoder Regularisation *(sparse only)*

```
L_routing = sparse_weight × mean(|W|)    W = routing Conv1×1 weights
```

L1 penalty on the routing matrix. As training progresses, `routing_sparsity` (fraction of weights < 1e-3) is logged each epoch.

---

## Model Size

With `num_channels=128` (default):

| Module | Params |
|---|---|
| head_small + head_medium + head_large | ~946K |
| blend_up_sm + blend_up_med | ~295K |
| fuse_sm_med | ~443K |
| fuse_med_lg (dense) | ~184K |
| pre_masks + masks | ~47K |
| **Total (dense)** | **~1.9M** |

To scale: `num_channels=64` → ~484K · `96` → ~1.1M · `128` → ~1.9M.

---

## Dataset — ADE20K-150

| | |
|---|---|
| Source | [MIT CSAIL ADE20K Challenge 2016](https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) (~922 MB) |
| Train / Val | 20,210 / 2,000 images |
| Classes | 150 semantic categories + background (`num_classes=151`) |
| Mask format | uint8 PNG, pixel = class index 0–150 |
| Avg classes/image | ~10 |

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Training (single GPU)

```bash
# 1. Prepare data (run once, ~922 MB download)
python -m src.dataloaders.ade20k --download

# 2. Train — reads config.json automatically
./train.sh

# Or with CLI overrides
python -m src.training.trainer --epochs 100 --batch-size 16

# Resume from last checkpoint
python -m src.training.trainer --resume last
```

### Training on Colab

```bash
# Colab dual T4 (DDP)
git checkout colab-dual-t4
!bash train.sh          # auto-detects 2 GPUs → torchrun --nproc_per_node=2

# Colab TPU
git checkout colab-tpu
!bash train.sh          # auto-installs torch_xla + downloads dataset
```

---

## Configuration

`config.json` (CLI args override):

```json
{
  "data_root":            "data",
  "epochs":               200,
  "batch_size":           16,
  "lr":                   0.001,
  "num_workers":          8,
  "num_channels":         128,
  "focus_size":           32,
  "image_size":           224,
  "num_classes":          151,
  "loss_type":            "pure_focal_iou",
  "absent_weight":        0.2,
  "diversity_weight":     0.1,
  "attn_guide_weight":    0.5,
  "attn_exclusive_weight":0.3,
  "decoder_type":         "dense",
  "resume":               false
}
```

### To run sparse decoder ablation

```json
{
  "decoder_type":  "sparse",
  "sparse_weight": 0.001
}
```

Log will show `routing sparsity=X% mean=Y` each epoch.

---

## Training Pipeline

| Component | Detail |
|---|---|
| Optimiser | Adam |
| LR schedule | 5-epoch linear warmup → CosineAnnealingLR (→ lr × 0.01) |
| Gradient clipping | `clip_grad_norm(max_norm=1.0)` |
| Mixed precision | AMP (autocast + GradScaler) |
| Class weighting | Median-frequency balancing, cached to `data/class_freq.json` |
| Augmentation | H-flip, rotate ±15°, translate 10%, brightness/contrast/saturation jitter |
| Checkpointing | `sf_seg_best.pt` + `sf_seg_last.pt` with optimizer + scheduler state |

---

## Evaluation

Validation mIoU is computed every epoch from a full confusion matrix and logged to console and `logs/train_log.csv`:

```
Epoch N | lr=X | train loss=X seg=X div=X acc=X mIoU=X | val loss=X acc=X mIoU=X
         top-10 val IoU: wall=0.71  floor=0.68  person=0.58 ...
```

Visualise predictions and attention maps:

```bash
python -m src.visualization.attention \
    --checkpoint checkpoints/sf_seg_best.pt \
    --data-root  data \
    --num-images 8
```

---

## File Structure

```
sf_seg/
├── src/
│   ├── models/
│   │   └── sf_seg.py           # attention_head, sf_seg (decoder_type: dense|sparse)
│   ├── losses/
│   │   └── losses.py           # pure_focal_iou, focal, ce_iou, attention_guide,
│   │                           # attention_exclusivity, diversity
│   ├── dataloaders/
│   │   └── ade20k.py           # download + prepare ADE20K-150
│   ├── training/
│   │   └── trainer.py          # single-GPU: AMP, warmup, grad clip, mIoU logging
│   ├── visualization/
│   │   ├── attention.py        # attention map visualisation
│   │   ├── evaluation.py       # cross-dataset evaluation
│   │   └── architecture.py     # architecture diagram generator
│   └── utils/
│       └── benchmark.py        # latency benchmark (CUDA)
│
├── config.json                 # default hyperparameters
├── train.sh                    # launcher: detects GPU/TPU, calls right trainer
└── archive/                    # legacy COCO scripts (reference only)
```

### Branches

| Branch | Target |
|---|---|
| `main` | Single GPU (local) |
| `colab-dual-t4` | Dual T4 via DDP (Colab / Kaggle) |
| `colab-tpu` | TPU via torch_xla (Colab) |
