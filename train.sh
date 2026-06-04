#!/bin/bash
# Train sf_seg on ADE20K-150 — Colab TPU edition.
# Usage (in Colab): !bash train.sh [extra args]
# Just run this script — it handles everything automatically.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Install torch_xla if running in Colab ──────────────────────────────────
if python -c "import google.colab" 2>/dev/null; then
    echo "=== Colab detected — installing torch_xla ==="
    pip install torch_xla[tpu] \
        -f https://storage.googleapis.com/libtpu-releases/index.html \
        -q --no-warn-script-location
else
    echo "=== Not in Colab — skipping torch_xla install ==="
fi

# ── 2. Install other Python requirements ──────────────────────────────────────
echo "=== Checking Python requirements ==="
pip install -q tqdm Pillow torchvision

# ── 3. Download / prepare ADE20K dataset if not ready ─────────────────────────
DATA_TRAIN="data/images/train"
if [ ! -d "$DATA_TRAIN" ] || [ -z "$(ls -A "$DATA_TRAIN" 2>/dev/null)" ]; then
    echo "=== Dataset not found — downloading ADE20K ==="
    python -m src.dataloaders.ade20k --download
else
    COUNT=$(ls "$DATA_TRAIN" | wc -l)
    echo "=== Dataset ready ($COUNT training images) ==="
fi

# ── 4. Run training ───────────────────────────────────────────────────────────
echo "=== Starting training ==="
python -m src.training.trainer_tpu "$@"
