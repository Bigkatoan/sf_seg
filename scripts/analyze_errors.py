#!/usr/bin/env python3
"""Decompose val errors: long-tail mIoU arithmetic, confusion pairs,
boundary-vs-interior errors. Answers: why does mIoU stay low when
attention/localization looks good?

Usage:
    python -m scripts.analyze_errors [--checkpoint checkpoints/sf_seg_best.pt]
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


def boundary_mask(t: torch.Tensor, width: int = 2) -> torch.Tensor:
    """(B,H,W) labels → bool mask of pixels within `width` px of a class edge."""
    x   = t.float().unsqueeze(1)
    k   = 2 * width + 1
    dil = F.max_pool2d(x, k, stride=1, padding=width)
    ero = -F.max_pool2d(-x, k, stride=1, padding=width)
    return ((dil - ero) > 0.5).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/sf_seg_best.pt')
    ap.add_argument('--config',     default='config.json')
    ap.add_argument('--max-images', type=int, default=500)
    ap.add_argument('--batch-size', type=int, default=4)
    args = ap.parse_args()

    cfg    = json.loads(Path(args.config).read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C      = cfg['num_classes']

    names = {}
    p = Path(cfg['data_root']) / 'cat_to_idx.json'
    if p.exists():
        raw = json.loads(p.read_text()).get('idx_to_name', {})
        names = {int(k) - 1: v.split(',')[0] for k, v in raw.items() if int(k) >= 1}

    model = sf_seg_v2(
        num_channels=cfg['num_channels'], focus_size=cfg['focus_size'],
        num_classes=C, backbone_variant=cfg['backbone_variant'],
        dw_kernel=cfg['dw_kernel']).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.eval()
    print(f"Loaded {args.checkpoint} (epoch={ckpt.get('epoch', '?')})")

    val_ds = ADE20KDataset(
        Path(cfg['data_root']) / 'images' / 'val',
        Path(cfg['data_root']) / 'masks' / 'val',
        image_size=cfg['image_size'], augment=False)
    val_ds.pairs = val_ds.pairs[:args.max_images]
    loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    conf = torch.zeros(C, C, dtype=torch.long, device=device)   # [gt, pred]
    err_bnd = err_int = tot_bnd = tot_int = 0

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for imgs, masks in tqdm(loader, desc='analyze'):
            imgs, masks = imgs.to(device), masks.to(device)
            logits, _, _ = model(imgs)
            pred  = logits.argmax(1)
            valid = masks != 255

            bnd   = boundary_mask(torch.where(valid, masks, 0), width=2) & valid
            inter = valid & ~bnd
            wrong = (pred != masks)
            err_bnd += (wrong & bnd).sum().item();   tot_bnd += bnd.sum().item()
            err_int += (wrong & inter).sum().item(); tot_int += inter.sum().item()

            idx = masks[valid] * C + pred[valid]
            conf += torch.bincount(idx, minlength=C * C).view(C, C)

    conf   = conf.cpu().double()
    gt_px  = conf.sum(1)
    inter_ = conf.diag()
    union  = gt_px + conf.sum(0) - inter_
    iou    = torch.where(union > 0, inter_ / union.clamp(min=1), torch.zeros(C))
    present = gt_px > 0

    print(f"\n=== mIoU arithmetic ({int(present.sum())} present classes) ===")
    print(f"overall mIoU                 : {iou[present].mean():.4f}")
    order = gt_px.argsort(descending=True)
    top30 = order[:30]
    rest  = order[30:][present[order[30:]]]
    print(f"mIoU top-30 frequent classes : {iou[top30].mean():.4f} "
          f"(= {100 * gt_px[top30].sum() / gt_px.sum():.1f}% of pixels)")
    print(f"mIoU remaining {len(rest):>3} classes    : {iou[rest].mean():.4f} "
          f"(= {100 * gt_px[rest].sum() / gt_px.sum():.1f}% of pixels)")

    print(f"\n=== boundary vs interior errors (±2px of class edge) ===")
    print(f"boundary pixels: {100 * tot_bnd / (tot_bnd + tot_int):.1f}% of valid, "
          f"error rate {100 * err_bnd / max(tot_bnd, 1):.1f}%")
    print(f"interior pixels: error rate {100 * err_int / max(tot_int, 1):.1f}%")
    print(f"share of all errors at boundary: {100 * err_bnd / max(err_bnd + err_int, 1):.1f}%")

    print(f"\n=== top confusion pairs (% of GT class pixels) ===")
    conf_n = conf / gt_px.clamp(min=1).unsqueeze(1)
    off = conf_n.clone(); off.fill_diagonal_(0)
    # weight by gt pixel count so we see globally significant confusions
    flat = (off * gt_px.unsqueeze(1)).flatten()
    topk = flat.topk(20).indices
    print(f"{'GT':<18} {'→ predicted':<18} {'%GT':>6} {'GT IoU':>7}")
    for t in topk:
        g, pr = int(t) // C, int(t) % C
        print(f"{names.get(g, g):<18} {names.get(pr, pr):<18} "
              f"{100 * off[g, pr]:>5.1f}% {iou[g]:>7.3f}")

    print(f"\n=== recall vs precision (where do present-class pixels go) ===")
    recall = inter_ / gt_px.clamp(min=1)
    prec   = inter_ / conf.sum(0).clamp(min=1)
    lowrec = [(float(recall[c]), float(prec[c]), c) for c in range(C)
              if present[c] and gt_px[c] > 1000]
    lowrec.sort()
    print("10 worst-recall classes (>1000 GT px):")
    for r, pcs, c in lowrec[:10]:
        print(f"  {names.get(c, c):<22} recall={r:.3f} precision={pcs:.3f} IoU={iou[c]:.3f}")


if __name__ == '__main__':
    main()
