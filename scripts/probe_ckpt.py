#!/usr/bin/env python3
"""Probe a checkpoint on ADE20K val: per-class IoU distribution, predicted-class
coverage, confusion toward dominant classes.

Usage:
    python -m scripts.probe_ckpt [--checkpoint checkpoints/sf_seg_best.pt] [--max-images 500]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.sf_seg_v2 import sf_seg_v2
from src.training.trainer import ADE20KDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/sf_seg_best.pt')
    ap.add_argument('--config',     default='config.json')
    ap.add_argument('--max-images', type=int, default=2000)
    ap.add_argument('--batch-size', type=int, default=16)
    args = ap.parse_args()

    cfg    = json.loads(Path(args.config).read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C      = cfg['num_classes']

    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    # Ưu tiên arch flags lưu trong checkpoint (config có thể đã đổi sau khi train)
    def af(key, default):
        return ckpt[key] if isinstance(ckpt, dict) and key in ckpt else cfg.get(key, default)
    _am = af('attn_masks', cfg.get('attn_masks'))
    model = sf_seg_v2(
        num_channels=cfg['num_channels'], focus_size=cfg['focus_size'],
        num_classes=C, backbone_variant=cfg['backbone_variant'],
        dw_kernel=cfg['dw_kernel'],
        decoder_dim=af('decoder_dim', 256), hr_dim=af('hr_dim', 96),
        attn_masks=tuple(_am) if _am else None,
        budget_ladder=af('budget_ladder', False),
        pos_encode=af('pos_encode', False),
        enable_ensemble=af('enable_ensemble', False),
        attn_temperature=af('attn_temperature', 1.0),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {args.checkpoint}  (epoch={ckpt.get('epoch', '?')})")

    val_ds = ADE20KDataset(
        Path(cfg['data_root']) / 'images' / 'val',
        Path(cfg['data_root']) / 'masks' / 'val',
        image_size=cfg['image_size'], augment=False)
    if args.max_images < len(val_ds):
        val_ds.pairs = val_ds.pairs[:args.max_images]
    loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    inter = torch.zeros(C, dtype=torch.float64)
    union = torch.zeros(C, dtype=torch.float64)
    gt_px = torch.zeros(C, dtype=torch.float64)   # GT pixels per class
    pr_px = torch.zeros(C, dtype=torch.float64)   # predicted pixels per class

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for imgs, masks in tqdm(loader, desc='probe'):
            imgs, masks = imgs.to(device), masks.to(device)
            logits, _, _ = model(imgs)
            pred  = logits.argmax(1)
            valid = masks != 255
            p, t  = pred[valid], masks[valid]
            for c_blk in range(0, C, 64):
                cs = torch.arange(c_blk, min(c_blk + 64, C), device=device)
                pm = p.unsqueeze(0) == cs.unsqueeze(1)
                tm = t.unsqueeze(0) == cs.unsqueeze(1)
                inter[cs.cpu()] += (pm & tm).sum(1).double().cpu()
                union[cs.cpu()] += (pm | tm).sum(1).double().cpu()
                gt_px[cs.cpu()] += tm.sum(1).double().cpu()
                pr_px[cs.cpu()] += pm.sum(1).double().cpu()

    present  = gt_px > 0
    iou      = torch.where(union > 0, inter / union.clamp(min=1), torch.zeros_like(union))
    miou     = iou[present].mean().item()

    n_pred   = int((pr_px > 0).sum())
    n_gt     = int(present.sum())
    iou_p    = iou[present]

    print(f"\n=== Probe results ({len(val_ds)} val images) ===")
    print(f"mIoU (present classes)     : {miou:.4f}")
    print(f"GT classes present          : {n_gt}/{C}")
    print(f"Classes EVER predicted      : {n_pred}/{C}")
    print(f"Classes with IoU > 0.01     : {int((iou_p > 0.01).sum())}/{n_gt}")
    print(f"Classes with IoU > 0.10     : {int((iou_p > 0.10).sum())}/{n_gt}")
    print(f"Classes with IoU = 0        : {int((iou_p == 0).sum())}/{n_gt}")

    # Pixel share of predictions vs GT for top dominant classes
    tot_pr, tot_gt = pr_px.sum(), gt_px.sum()
    top = pr_px.argsort(descending=True)[:15]
    print(f"\n{'cls':>4} {'pred%':>7} {'gt%':>7} {'IoU':>6}")
    for c in top:
        print(f"{int(c):>4} {100*pr_px[c]/tot_pr:>6.2f}% {100*gt_px[c]/tot_gt:>6.2f}% {iou[c]:>6.3f}")

    # Histogram of per-class IoU
    bins = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 1.01]
    hist = np.histogram(iou_p.numpy(), bins=bins)[0]
    print("\nIoU histogram (present classes):")
    for i in range(len(hist)):
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]}")


if __name__ == '__main__':
    main()
