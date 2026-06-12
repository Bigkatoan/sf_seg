#!/bin/bash
# Stage-1: pretrain SFBackbone + presence head trên ADE20K (multi-label
# "class nào có mặt trong ảnh"). Nhanh (~1-2 phút/epoch, backbone-only).
#
# Usage:
#   ./train_backbone.sh                      # 40 epochs mặc định
#   ./train_backbone.sh --epochs 60
#   ./train_backbone.sh --resume last        # train tiếp từ epoch đã dừng
#
# Stage-2 (sau khi xong):
#   ./train.sh --resume checkpoints/backbone_pres.pt --restart --backbone-lr-factor 0.1

source venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m src.training.pretrain_presence "$@"
