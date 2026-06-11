#!/bin/bash
# SF-Seg training on ADE20K-150
# Usage: ./train.sh [extra args passed to trainer]
#
# First time:
#   python -m src.dataloaders.ade20k --download
#
# Resume:
#   ./train.sh --resume last

source venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m src.training.trainer "$@"
