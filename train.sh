#!/bin/bash
# Train sf_seg on ADE20K-150 semantic segmentation.
# Usage: ./train.sh [extra args]
# Prepare data first: python prepare_ade20k.py --download
rm -rf outputs checkpoints logs
python train_sf_seg.py "$@"
