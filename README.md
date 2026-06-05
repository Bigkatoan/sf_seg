# sf_seg — Multi-scale Sparse-Focus Segmentation

Lightweight semantic segmentation built around **budget-constrained spatial attention via clamped softmax**. A shared ResNet-style backbone extracts multi-scale features; three lightweight attention heads apply sparse spatial selection at each scale; outputs are fused bottom-up in a UNet-style decoder.

Dataset: **ADE20K-150** (150 semantic categories + background).

---

## Architecture

### Overview

```
Input x (B, 3, 224, 224)
  │
  └─ Shared backbone
       stem   Conv(3→64, 3×3, s=2) + GN + GELU        →  (64,  H/2)
       stage1 BasicBlock(64→128, s=2) + BasicBlock      →  (128, H/4)  ─► head_large
       stage2 BasicBlock(128→128, s=2) + BasicBlock     →  (128, H/8)  ─► head_medium
       stage3 BasicBlock(128→128, s=2) + BasicBlock     →  (128, H/16) ─► head_small

  Attention heads (on backbone features, not raw RGB):
       head_large  (128, H/4)  → a_large  (B, C, H/4,  W/4 )   fine detail    (25% coverage)
       head_medium (128, H/8)  → a_medium (B, C, H/8,  W/8 )   mid-range      (25% coverage)
       head_small  (128, H/16) → a_small  (B, C, H/16, W/16)   global context  (5% coverage)
                                         attn_l ↗ (head_large)

  Decoder (bottom-up):
       a_small  → upsample → blend_up_sm → cat(a_medium) → fuse_sm_med → d_med  (C, H/8)
       d_med    → upsample → blend_up_med → cat(a_large) → fuse_med_lg → d_lg   (C/2, H/4)
       d_lg     → upsample → pre_masks → masks → logits  (B, 151, H, W)

  Forward returns:
       logits     (B, 151, 224, 224)   segmentation logits
       attn_guide (B,   1, 224, 224)   amax(attn_l) upsampled — visualisation
       attn_l     (B,   C,  56,  56)   raw head_large attention — used for losses
```

### BasicBlock (ResNet-style)

```
x ──┬── Conv(3×3) + GN + GELU ── Conv(3×3) + GN ──┬── GELU ──►
    └──── shortcut (1×1 conv + GN  if shape changes) ┘
```

### Attention Head

Each head operates on **backbone features** (not raw RGB), applying lightweight projection → sparse selection → channel mix:

```
x  (B, C, H, W)   ← backbone feature at this scale
  DWConv(3×3) + Conv(1×1 → 2C) + GN   # spatial context + score|features split
  chunk(2) ──► score    (B, C, L)
             → features (B, C, L)      L = H × W

  attn     = clamped_softmax(score, k)    # sparse ∈ [0,1], Σ=k per channel
  attended = channel_mix(attn × features) # 1×1 conv + GELU
```

### Clamped Softmax (Budget Attention)

Each channel attends to **exactly k locations**, each weight in **[0, 1]**:

```
k = min(focus_size², L − 1)
p = softmax(score) × k              # sum = k, values may exceed 1
# Closed-form Lagrangian solution via topk — O(k log k), no Python loop:
attn = clamp(p − λ*, 0, 1)          # sum = k, each value ∈ [0, 1]
```

### Scale Table (image_size=224, focus_size=28, C=128)

| Head | Feature input | Attention space | k | L | Coverage |
|:---|:---:|:---:|---:|---:|:---:|
| `head_small` | H/16 = 14×14 | 14×14 | 9 | 196 | 4.6% |
| `head_medium` | H/8 = 28×28 | 28×28 | 196 | 784 | 25% |
| `head_large` | H/4 = 56×56 | 56×56 | 784 | 3136 | 25% |

---

## Two Model Variants

### A — Custom Backbone (`sf_seg.py`)

ResNet-style backbone built from scratch with GroupNorm throughout. Trained from random init or with `pretrain_encoder.py` on ImageNet.

| Module | Params |
|---|---|
| Backbone (stem + 3 stages) | 1,742,400 |
| Attention heads (×3) | 152,448 |
| Decoder + classifier | 969,687 |
| **Total** | **2,864,535** |

### B — ResNet-18 Backbone (`sf_seg_r18.py`)

torchvision ResNet-18 with **ImageNet-1K pretrained weights** loaded automatically. Adapter 1×1 convs unify non-uniform channels (64/128/256) to C before the attention heads.

```
ResNet-18 stem: Conv(7×7, s=2) + BN + ReLU + MaxPool → H/4
layer1 (64ch)  → Adapter(64→C)  → head_large
layer2 (128ch) → Adapter(128→C) → head_medium
layer3 (256ch) → Adapter(256→C) → head_small
```

| Module | Params |
|---|---|
| ResNet-18 (layer1-3) | 2,782,784 |
| Adapters (×3) | 58,112 |
| Attention heads (×3) | 152,448 |
| Decoder + classifier | 969,687 |
| **Total** | **3,963,031** |

### Comparison

| | Custom (`sf_seg.py`) | ResNet-18 (`sf_seg_r18.py`) |
|--|--|--|
| Backbone | Custom ResNet BasicBlocks | torchvision ResNet-18 |
| Pretrained | No (or custom ImageNet pretrain) | ImageNet-1K out of the box |
| Normalization | GroupNorm throughout | BatchNorm in backbone, GN in adapters/decoder |
| Total params | 2.86M | 3.96M |

---

## Loss Functions

Total loss per training step:

```
L = L_seg  +  diversity_weight × L_diversity
           +  attn_guide_weight × L_guide
           +  attn_exclusive_weight × L_exclusive
```

### `L_seg` — Segmentation loss

Default: **`focal_iou`** — focal cross-entropy (stable gradient via `p_c − 1(c==t)`) + soft IoU. Available options: `pure_focal_iou`, `ce_iou`, `focal`, `ce`, `iou`.

### `L_diversity` — Attention Diversity

Penalises cosine similarity between attention channels via Gram matrix off-diagonal:

```
G = normalize(attn) @ normalize(attn)ᵀ        # (B, C, C)
L_diversity = mean(off_diagonal(G)²) / C(C−1)
```

### `L_guide` — IoU-Guided Attention Supervision

Assigns each attention channel to its best-matching GT class (via IoU, detached), then trains that channel to match the class mask via Dice loss. Spatial downsampling to 14×14 keeps compute tractable.

### `L_exclusive` — Winner-Takes-All Specialisation

```
L_exclusive = mean(non_winner IoU)    [winner selected via no_grad argmax]
```

Paired with `L_guide`: winner channels → match their class; non-winner channels → zero overlap with other classes. After training, each channel specialises for one class — attention maps become class-specific encoders.

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Training

```bash
# Prepare data (run once, ~922 MB download)
python -m src.dataloaders.ade20k --download

# Train — reads config.json automatically
./train.sh

# Choose backbone explicitly
python -m src.training.trainer --backbone custom    # custom ResNet (default)
python -m src.training.trainer --backbone resnet18  # ImageNet pretrained

# CLI overrides
python -m src.training.trainer --backbone resnet18 --epochs 100 --batch-size 16

# Resume
python -m src.training.trainer --resume last
```

### TensorBoard

Each backbone logs to its own subdirectory — compare both runs side-by-side:

```bash
tensorboard --logdir logs/tensorboard
# → http://localhost:6006
```

Logged per epoch:
- **Scalars**: `Loss/train`, `Loss/val`, `mIoU/train`, `mIoU/val`, `Accuracy`, `LR`, `SegLoss`, `DivLoss`
- **Images** (every 5 epochs): sample image, GT mask, predicted mask, head_large attention map
- **Graph**: full model architecture (pushed once at startup)

---

## Configuration

`config.json` (all fields overridable via CLI `--arg`):

```json
{
  "data_root":             "data",
  "epochs":                200,
  "batch_size":            16,
  "lr":                    3e-4,
  "num_workers":           8,
  "num_channels":          128,
  "focus_size":            28,
  "image_size":            224,
  "num_classes":           151,
  "loss_type":             "focal_iou",
  "absent_weight":         0.2,
  "diversity_weight":      0.1,
  "attn_guide_weight":     0.3,
  "attn_exclusive_weight": 0.2,
  "decoder_type":          "dense",
  "backbone":              "resnet18",
  "resume":                false
}
```

---

## Training Pipeline

| Component | Detail |
|---|---|
| Optimiser | Adam |
| LR schedule | 5-epoch linear warmup → CosineAnnealingLR (→ lr × 0.01) |
| Gradient clipping | `clip_grad_norm(max_norm=1.0)` |
| Mixed precision | AMP (autocast + GradScaler) |
| Augmentation | Noise (σ 0.01–0.04), brightness/contrast jitter |
| Checkpointing | `sf_seg_best.pt` + `sf_seg_last.pt` with full optimizer + scheduler state |

---

## Evaluation

Validation mIoU is computed every epoch from a full GPU-resident confusion matrix:

```
Epoch N | lr=X | train loss=X seg=X div=X acc=X mIoU=X | val loss=X acc=X mIoU=X
         top-10 val IoU: wall=0.71  floor=0.68  person=0.58 ...
```

Logged to `logs/train_log.csv` and TensorBoard.

---

## File Structure

```
sf_seg/
├── src/
│   ├── models/
│   │   ├── sf_seg.py           # custom ResNet backbone + attention + decoder
│   │   └── sf_seg_r18.py       # ResNet-18 backbone (ImageNet pretrained)
│   ├── losses/
│   │   └── losses.py           # focal_iou, attention_guide, attention_exclusivity, diversity
│   ├── dataloaders/
│   │   └── ade20k.py           # download + prepare ADE20K-150
│   ├── training/
│   │   └── trainer.py          # AMP, warmup, grad clip, mIoU, TensorBoard
│   ├── visualization/
│   │   ├── attention.py        # attention map visualisation
│   │   └── evaluation.py       # cross-dataset evaluation
│   └── utils/
│       └── benchmark.py        # latency benchmark (CUDA)
│
├── config.json                 # default hyperparameters
├── train.sh                    # launcher script
├── pretrain_encoder.py         # ImageNet backbone pretraining (custom backbone)
├── download_imagenet.py        # download ILSVRC/imagenet-1k via HuggingFace
└── pack_webdataset.py          # convert ImageFolder → WebDataset tar shards
```
