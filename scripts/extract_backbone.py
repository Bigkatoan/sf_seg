#!/usr/bin/env python3
"""Warm-start init for the wide-decoder (decoder_dim=256) architecture.

Copies every tensor whose name+shape matches from an existing checkpoint
(backbone, attention heads, fuse_ts/tsm, projs, aux heads, hr_adapt) into a
freshly initialized wide-decoder model. New layers (fuse_tsml, pre_masks,
proj_hr, hr_fuse, masks) keep their fresh init.

Usage:
    python -m scripts.extract_backbone \
        --src checkpoints/sf_seg_best.pt --dst checkpoints/sf_seg_dec256_init.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.models.sf_seg_v2 import sf_seg_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src',    default='checkpoints/sf_seg_best.pt')
    ap.add_argument('--dst',    default='checkpoints/sf_seg_dec256_init.pt')
    ap.add_argument('--config', default='config.json')
    args = ap.parse_args()

    cfg   = json.loads(Path(args.config).read_text())
    model = sf_seg_v2(
        num_channels=cfg['num_channels'], focus_size=cfg['focus_size'],
        num_classes=cfg['num_classes'], backbone_variant=cfg['backbone_variant'],
        dw_kernel=cfg['dw_kernel'], decoder_dim=cfg.get('decoder_dim', 256),
        hr_dim=cfg.get('hr_dim', 96))
    new_state = model.state_dict()

    ckpt = torch.load(args.src, map_location='cpu', weights_only=False)
    old  = ckpt.get('model_state_dict', ckpt)

    moved, skipped, fresh = 0, [], []
    n_moved = 0
    for k, v in new_state.items():
        if k in old and old[k].shape == v.shape:
            new_state[k] = old[k].clone()
            moved += 1
            n_moved += v.numel()
        elif k in old:
            skipped.append(f"{k}  {tuple(old[k].shape)} → {tuple(v.shape)}")
        else:
            fresh.append(k)

    torch.save({'model_state_dict': new_state, 'epoch': 0}, args.dst)
    total = sum(v.numel() for v in new_state.values())
    print(f"src       : {args.src} (epoch={ckpt.get('epoch', '?')})")
    print(f"transferred: {moved} tensors, {n_moved/1e6:.2f}M / {total/1e6:.2f}M params")
    print(f"shape-changed (fresh init): {len(skipped)}")
    for s in skipped:
        print(f"  {s}")
    print(f"new keys (fresh init): {len(fresh)}")
    for s in fresh:
        print(f"  {s}")
    print(f"\nSaved → {args.dst}")
    print("Train with: ./train.sh --resume checkpoints/sf_seg_dec256_init.pt --restart --lr 5e-4")


if __name__ == '__main__':
    main()
