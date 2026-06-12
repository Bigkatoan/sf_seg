#!/usr/bin/env python3
"""Gom ImageNet val images theo wnid để dùng được với ImageFolder.

Kaggle (imagenet-object-localization-challenge) giải nén ra:
    ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_00000001.JPEG  (phẳng, không thư mục)
    LOC_val_solution.csv  (ImageId, PredictionString = 'wnid x y w h ...')

Script chuyển thành:
    data/imagenet/val/<wnid>/ILSVRC2012_val_*.JPEG

Usage:
    python -m scripts.restructure_in1k_val \
        --val-dir ILSVRC/Data/CLS-LOC/val \
        --solution LOC_val_solution.csv \
        --out data/imagenet/val
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-dir',  required=True)
    ap.add_argument('--solution', required=True)
    ap.add_argument('--out',      default='data/imagenet/val')
    ap.add_argument('--move', action='store_true',
                    help='move thay vì copy (tiết kiệm 6.7GB)')
    ap.add_argument('--symlink', action='store_true',
                    help='symlink thay vì copy (0 byte, nhanh nhất — giữ nguyên cache)')
    args = ap.parse_args()

    val_dir = Path(args.val_dir)
    out     = Path(args.out)

    mapping: dict[str, str] = {}
    with open(args.solution) as f:
        for row in csv.DictReader(f):
            # PredictionString: 'n01751748 117 86 379 256 ...' — wnid đầu tiên
            mapping[row['ImageId']] = row['PredictionString'].split()[0]

    n_done = n_miss = 0
    if args.symlink:
        op = lambda s, d: Path(d).symlink_to(Path(s).resolve())
    elif args.move:
        op = shutil.move
    else:
        op = shutil.copy2
    for img_id, wnid in tqdm(mapping.items(), desc='restructure val'):
        src = val_dir / f'{img_id}.JPEG'
        if not src.exists():
            n_miss += 1
            continue
        dst_dir = out / wnid
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        op(str(src), str(dst))
        n_done += 1

    print(f'done: {n_done} images → {out}  (missing: {n_miss})')
    print(f'classes: {len(set(mapping.values()))}')


if __name__ == '__main__':
    main()
