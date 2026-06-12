#!/bin/bash
# Stage-0: pretrain SFBackbone trên ImageNet-1k (classification 1000 class).
#
# Chuẩn bị data (~155GB, cần tài khoản Kaggle):
#   pip install kaggle && kaggle competitions download -c imagenet-object-localization-challenge
#   unzip → data/imagenet/train/<wnid>/*.JPEG
#   Val cần gom theo wnid từ LOC_val_solution.csv:
#     python scripts/restructure_in1k_val.py   # (tạo nếu cần)
#
# Usage:
#   ./train_imagenet.sh                      # 100 epochs, bs 256, cosine
#   ./train_imagenet.sh --resume last        # train tiếp
#   ./train_imagenet.sh --epochs 50          # recipe ngắn
#
# Sau khi xong:
#   ./train.sh --resume checkpoints/backbone_in1k.pt --restart

source venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m src.training.pretrain_imagenet "$@"
