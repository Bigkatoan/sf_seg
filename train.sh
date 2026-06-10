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
python -m src.training.trainer "$@"
