#!/bin/bash
# Train sf_seg on ADE20K-150 semantic segmentation.
# Usage: ./train.sh [extra args]
# Prepare data first: python -m src.dataloaders.ade20k --download
rm -rf outputs checkpoints logs
python -m src.training.trainer "$@"
