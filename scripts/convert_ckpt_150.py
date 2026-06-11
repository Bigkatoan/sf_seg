#!/usr/bin/env python3
"""Convert a 151-class checkpoint to a 150-class warm-start init.

Standard ADE20K protocol drops label 0 (unlabeled): new class c = old class
c+1. All backbone/decoder/attention weights transfer unchanged; classifier
convs (masks final layer + 3 aux heads) keep channels 1..150 and drop the
old background channel 0.

Usage:
    python -m scripts.convert_ckpt_150 \
        --src checkpoints/sf_seg_best.pt --dst checkpoints/sf_seg_150_init.pt
"""
from __future__ import annotations

import argparse
import torch

# Final-layer keys whose out_channels dimension is num_classes
CLS_KEYS = [
    'masks.2.weight', 'masks.2.bias',
    'aux_cls_tiny.weight', 'aux_cls_tiny.bias',
    'aux_cls_small.weight', 'aux_cls_small.bias',
    'aux_cls_medium.weight', 'aux_cls_medium.bias',
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='checkpoints/sf_seg_best.pt')
    ap.add_argument('--dst', default='checkpoints/sf_seg_150_init.pt')
    args = ap.parse_args()

    ckpt  = torch.load(args.src, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)

    converted = 0
    for k in CLS_KEYS:
        if k not in state:
            raise KeyError(f"expected classifier key missing: {k}")
        if state[k].shape[0] != 151:
            raise ValueError(f"{k}: expected 151 out-channels, got {state[k].shape[0]}")
        state[k] = state[k][1:151].clone()   # drop old background channel 0
        converted += 1

    torch.save({'model_state_dict': state, 'epoch': 0}, args.dst)
    print(f"Converted {converted} classifier tensors 151→150  "
          f"(src epoch={ckpt.get('epoch', '?')})")
    print(f"Saved → {args.dst}")
    print("Train with: ./train.sh --resume checkpoints/sf_seg_150_init.pt --restart")


if __name__ == '__main__':
    main()
