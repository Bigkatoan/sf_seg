#!/bin/bash
# Train sf_seg on ADE20K-150 — Colab dual T4 edition.
# Usage (in Colab): !bash train.sh [extra args]
# Just run this script — it handles everything automatically.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Install requirements ───────────────────────────────────────────────────
echo "=== Checking requirements ==="
pip install -q tqdm Pillow torchvision

# ── 2. Download / prepare ADE20K dataset if not ready ─────────────────────────
DATA_TRAIN="data/images/train"
if [ ! -d "$DATA_TRAIN" ] || [ -z "$(ls -A "$DATA_TRAIN" 2>/dev/null)" ]; then
    echo "=== Dataset not found — downloading ADE20K ==="
    python -m src.dataloaders.ade20k --download
else
    COUNT=$(ls "$DATA_TRAIN" | wc -l)
    echo "=== Dataset ready ($COUNT training images) ==="
fi

# ── 3. Detect GPU count and launch accordingly ────────────────────────────────
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo "=== Detected $GPU_COUNT GPU(s) ==="

if [ "$GPU_COUNT" -ge 2 ]; then
    echo "=== Starting DDP training on $GPU_COUNT GPUs ==="
    torchrun \
        --nproc_per_node="$GPU_COUNT" \
        --master_port=29500 \
        -m src.training.trainer_ddp "$@"
elif [ "$GPU_COUNT" -eq 1 ]; then
    echo "=== Starting single-GPU training ==="
    python -m src.training.trainer "$@"
else
    echo "=== No GPU found — training on CPU ==="
    python -m src.training.trainer "$@"
fi
