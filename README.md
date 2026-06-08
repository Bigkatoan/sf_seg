# sf_seg — Multi-scale Sparse-Focus Segmentation

Lightweight semantic segmentation built around **budget-constrained spatial attention via clamped softmax**. A shared custom backbone extracts multi-scale features; three lightweight attention heads apply sparse spatial selection at each scale; outputs are fused bottom-up in a UNet-style decoder.

Dataset: **ADE20K-150** (150 semantic categories + background = 151 classes).

---

## Architecture

```
Input x (B, 3, 224, 224)
  │
  └─ Backbone
       stem   Conv(3→64, 3×3, s=2) + GELU          →  (64,  H/2)
       stage1 BasicBlock(64→128, s=2) + BasicBlock  →  (128, H/4)  ─► head_large
       stage2 BasicBlock(128→128, s=2) + BasicBlock →  (128, H/8)  ─► head_medium
       stage3 BasicBlock(128→128, s=2) + BasicBlock →  (128, H/16) ─► head_small
  │
  ├─ Attention heads (on backbone features, not raw RGB)
  │    head_large  → a_large  (B, 128, H/4,  W/4 )   fine detail     25% coverage
  │    head_medium → a_medium (B, 128, H/8,  W/8 )   mid-range       25% coverage
  │    head_small  → a_small  (B, 128, H/16, W/16)   global context   5% coverage
  │
  └─ Decoder (bottom-up UNet)
       a_small → upsample → blend_up_sm → cat(a_medium) → fuse_sm_med → d_med  (128, H/8)
       d_med   → upsample → blend_up_med → cat(a_large) → fuse_med_lg → d_lg   ( 64, H/4)
       d_lg    → upsample → pre_masks → masks → logits                          (151, H, W)

  Returns:
       logits     (B, 151, 224, 224)   segmentation logits
       attn_guide (B,   1, 224, 224)   amax(attn_l) upsampled — visualisation
       attn_l     (B, 128,  56,  56)   raw head_large attention — used for loss
```

### BasicBlock

No normalization — GELU activation, `bias=True` on all convolutions.

```
x ──┬── Conv(3×3) + GELU ── Conv(3×3) ──┬── GELU ──►
    └──── shortcut (1×1 conv if shape changes) ┘
```

### Attention Head

```
x  (B, C, H, W)   ← backbone feature at this scale
  DWConv(3×3) → Conv(1×1, out=2C)   # spatial context + score|features split
  chunk(2) ──► score    (B, C, H×W)
             → features (B, C, H×W)

  attn     = clamped_softmax(score, k)    # sparse ∈ [0,1], Σ=k per channel
  attended = channel_mix(attn × features) # 1×1 conv + GELU
```

### Clamped Softmax (Budget Attention)

Each channel attends to **exactly k locations**, each weight in **[0, 1]**, closed-form via topk — no Python loop:

```
k = min(focus_size², H×W − 1)
p = softmax(score) × k          # Σ=k, values may exceed 1
attn = clamp(p − λ*, 0, 1)      # Σ=k, each value ∈ [0, 1]
```

### Scale Table (`image_size=224`, `focus_size=28`, `C=128`)

| Head | Feature level | Grid | k | L | Coverage |
|:---|:---:|:---:|---:|---:|:---:|
| `head_large` | H/4 | 56×56 | 784 | 3136 | 25% |
| `head_medium` | H/8 | 28×28 | 196 | 784 | 25% |
| `head_small` | H/16 | 14×14 | 9 | 196 | 4.6% |

---

## Two Model Variants

### A — Custom Backbone (`sf_seg.py`)

ResNet-style backbone built from scratch. **No normalization** (GroupNorm removed), GELU throughout. Pretrain on ImageNet via `pretrain_encoder.py` or train from scratch.

| Module | Detail |
|---|---|
| Backbone | stem + 3 stages of 2× BasicBlock |
| Normalization | None |
| Activation | GELU |
| Pretrained | No (or via `pretrain_encoder.py`) |

### B — ResNet-18 Backbone (`sf_seg_r18.py`)

torchvision ResNet-18 with **ImageNet-1K pretrained weights** loaded automatically. Adapter 1×1 convs project non-uniform ResNet channels (64/128/256) to C before the attention heads.

```
stem: Conv(7×7, s=2) + BN + ReLU + MaxPool → H/4
layer1 (64ch)  → Adapter(64→C,  GN+GELU) → head_large
layer2 (128ch) → Adapter(128→C, GN+GELU) → head_medium
layer3 (256ch) → Adapter(256→C, GN+GELU) → head_small
```

| Module | Detail |
|---|---|
| Backbone | torchvision ResNet-18 layer1–3 |
| Normalization | BatchNorm (backbone), GroupNorm (adapters/decoder) |
| Activation | ReLU (backbone), GELU (adapters/decoder) |
| Pretrained | ImageNet-1K out of the box |

### Choosing

| | Custom | ResNet-18 |
|--|--|--|
| Backbone weights | Random init or custom ImageNet pretrain | ImageNet-1K automatic |
| Param count | ~2.5M | ~4.0M |
| Config | `"backbone": "custom"` | `"backbone": "resnet18"` |
| Use when | Want full control / lighter model | Want strong pretrained features fast |

---

## Loss (`sf_loss`)

```
L = L_seg  +  diversity_weight × L_div
```

`attn_guide_weight` and `attn_exclusive_weight` exist but are set to 0 by default — empirical results showed they converge to a fixed equilibrium without improving mIoU while consuming gradient budget.

### `L_seg` — Focal CE + Soft IoU

```
L_seg = focal_w × focal_CE  +  iou_w × soft_IoU
```

Focal CE uses full-resolution per-pixel CE (`BxHxW`, cheap). Soft IoU downsamples logits to 56×56 before softmax — reduces peak memory from ~2 GB to ~120 MB at B=32, C=151.

Default: `focal_w=0.6`, `iou_w=0.5`. Absent classes weighted by `no_obj_weight=0.1` in IoU.

### `L_div` — Attention Diversity

Penalises cosine similarity between attention channels via Gram matrix off-diagonal:

```
G = normalize(attn) @ normalize(attn)ᵀ    # (B, C, C)
L_div = mean(off_diagonal(G)²) / C(C−1)
```

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Prepare Data

```bash
# Download and prepare ADE20K-150 (~922 MB)
python -m src.dataloaders.ade20k --download
```

### Training

```bash
# Train — reads config.json automatically
./train.sh

# Select backbone explicitly
python -m src.training.trainer --backbone custom    # custom GELU (default)
python -m src.training.trainer --backbone resnet18  # ImageNet pretrained ResNet-18

# Freeze ResNet-18 backbone — train only adapters + heads + decoder (~1.2M params)
python -m src.training.trainer --backbone resnet18 --freeze-backbone

# Override config values via CLI
python -m src.training.trainer --epochs 200 --batch-size 16 --lr 3e-4

# Resume from last checkpoint (preserves optimizer + scheduler state)
python -m src.training.trainer --resume last

# Cosine restart — loads weights only, resets optimizer + scheduler
# Set restart: true and new lr in config.json, then:
python -m src.training.trainer --resume last --restart
```

### ImageNet Pretraining (custom backbone only)

Trains the full sf_seg feature pipeline (backbone + attention heads + decoder) as an ImageNet classifier. The classification head is discarded after training; backbone weights are saved for fine-tuning.

Supports both ImageFolder and WebDataset formats.

```bash
python scripts/pretrain_encoder.py \
  --data /path/to/imagenet_wds \   # WebDataset shards (train-*.tar) or ImageFolder root
  --num-channels 128 \             # must match config.json
  --focus-size 28 \                # must match config.json
  --image-size 224 \
  --batch-size 256 \
  --epochs 50

# After pretraining, add to config.json:
# "encoder_pretrained": "checkpoints/pretrain/enc_best.pt"
```

### Multi-GPU (2× T4, Kaggle)

```bash
git checkout feat/multi-gpu
# config.json on this branch: batch_size=32, lr=4e-4
./train.sh
```

Uses `nn.DataParallel` — no code changes needed, auto-detects GPU count.

### TensorBoard

```bash
tensorboard --logdir logs/tensorboard
# → http://localhost:6006
```

Logged per epoch: `Loss/train`, `Loss/val`, `mIoU/train`, `mIoU/val`, `Accuracy`, `LR`, `SegLoss`, `DivLoss`, sample images + attention maps (every 5 epochs).

---

## Configuration (`config.json`)

```json
{
  "data_root":             "data",
  "epochs":                400,
  "batch_size":            32,
  "lr":                    1e-4,
  "num_workers":           8,
  "image_size":            224,
  "num_channels":          128,
  "focus_size":            28,
  "encoder_stride":        1,
  "num_classes":           151,
  "backbone":              "resnet18",
  "freeze_backbone":       true,
  "encoder_pretrained":    null,
  "loss_type":             "focal_iou",
  "iou_w":                 0.5,
  "diversity_weight":      0.1,
  "absent_weight":         0.2,
  "attn_guide_weight":     0.0,
  "attn_exclusive_weight": 0.0,
  "decoder_type":          "dense",
  "resume":                false,
  "restart":               false,
  "aug_hflip":             true,
  "aug_resized_crop":      true,
  "aug_color_jitter":      true,
  "aug_gaussian_noise":    false,
  "aug_cutout":            true,
  "aug_shift":             true
}
```

All fields are overridable via CLI `--arg value`.

| Key | Values | Notes |
|---|---|---|
| `backbone` | `custom` / `resnet18` | custom = GELU no-norm; resnet18 = ImageNet pretrained |
| `freeze_backbone` | `false` / `true` | only applies to `resnet18`; freezes stem+layer1-3 |
| `encoder_pretrained` | path or `null` | path to `enc_best.pt` from `pretrain_encoder.py` (custom only) |
| `loss_type` | `focal_iou` / `focal` / `ce_iou` / `ce` | affects val loss only; train always uses `sf_loss` |
| `iou_w` | float | soft-IoU weight in `L_seg`; `0` = pure focal CE |
| `resume` | `false` / `"last"` / path | continue from checkpoint |
| `restart` | `false` / `true` | load weights only, reset optimizer + scheduler |

---

## Training Pipeline

| Component | Detail |
|---|---|
| Optimiser | Adam (`weight_decay=1e-4`), only trainable params |
| LR schedule | 5-epoch linear warmup → CosineAnnealingLR (`η_min = lr × 0.05`) |
| Cosine restart | `restart: true` — loads weights only, resets optimiser + scheduler |
| Freeze backbone | `freeze_backbone: true` — ResNet-18 stem+layer1-3 frozen; ~40-50% faster per step |
| Gradient clipping | `clip_grad_norm(max_norm=1.0)` |
| Mixed precision | AMP (`autocast` + `GradScaler`) |
| Augmentation | hflip, resized-crop (scale 0.7–1.4), color jitter, cutout, shift (±10%) |
| Checkpoints | `sf_seg_best.pt` (best val loss) + `sf_seg_last.pt`; CSV log appends across restarts |

---

## File Structure

```
sf_seg/
├── src/
│   ├── models/
│   │   ├── sf_seg.py            # custom backbone (GELU, no norm) + attention + decoder
│   │   └── sf_seg_r18.py        # ResNet-18 backbone (ImageNet pretrained)
│   ├── losses/
│   │   ├── sf_loss.py           # unified loss: focal_CE + soft_IoU + diversity
│   │   └── losses.py            # standalone loss functions
│   ├── dataloaders/
│   │   ├── ade20k.py            # download + prepare ADE20K-150
│   │   ├── sampler.py           # distributed / weighted sampler
│   │   └── utils.py             # dataloader utilities
│   └── training/
│       └── trainer.py           # AMP, warmup, grad clip, mIoU, TensorBoard
│
├── scripts/
│   ├── pretrain_encoder.py      # ImageNet backbone pretraining (custom backbone)
│   ├── download_imagenet.py     # download ImageNet-1K via HuggingFace
│   ├── pack_webdataset.py       # convert ImageFolder → WebDataset tar shards
│   ├── noise_aug.py             # augmentation visualisation
│   ├── benchmark.py             # latency benchmark (CUDA)
│   ├── attention.py             # attention map visualisation
│   ├── architecture.py          # architecture diagram
│   └── evaluation.py            # evaluation utilities
│
├── docs/
│   ├── architecture.png
│   ├── architecture.svg
│   └── sf_seg_methods.txt
│
├── config.json                  # default hyperparameters
├── train.sh                     # launcher script
├── setup.sh                     # venv setup
└── requirements.txt
```
