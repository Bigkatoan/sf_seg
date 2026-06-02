#!/bin/bash
# Train sf_seg on COCO2017 person segmentation.
# Usage: ./train.sh [extra args]
# Example: ./train.sh --epochs 50 --loss-type combine
rm -rf outputs checkpoints logs
python train_sf_seg.py "$@"
