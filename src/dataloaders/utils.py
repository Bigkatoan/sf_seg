#!/usr/bin/env python3
"""
Rebuild images/masks from existing processed data:
  - Resize to target image_size (default 224)
  - Filter: keep only images where person mask >= min_person_pixels (default 24*24=576)
  - Replaces data/images/ and data/masks/ in-place (writes to _new/ first, then swaps)

Usage:
    python prepare_data.py
    python prepare_data.py --image-size 224 --min-person-pixels 576 --workers 8
"""
from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def process_split(
    src_img: Path,
    src_mask: Path,
    dst_img: Path,
    dst_mask: Path,
    image_size: int,
    min_pixels: int,
    workers: int,
) -> tuple[int, int]:
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_mask.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for p in src_img.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    def worker(img_path: Path) -> str:
        mask_path = src_mask / (img_path.stem + ".png")
        if not mask_path.exists():
            return "no_mask"
        try:
            mask = Image.open(mask_path).convert("L")
            mask_r = mask.resize((image_size, image_size), Image.NEAREST)
            if (np.array(mask_r) > 0).sum() < min_pixels:
                return "too_small"
            img = Image.open(img_path).convert("RGB")
            img.resize((image_size, image_size), Image.BILINEAR).save(dst_img / img_path.name)
            mask_r.save(dst_mask / (img_path.stem + ".png"))
            return "ok"
        except Exception:
            return "error"

    results: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in tqdm(ex.map(worker, images), total=len(images), desc=src_img.parent.name + "/" + src_img.name):
            results.append(r)

    kept = results.count("ok")
    dropped = results.count("too_small")
    other = len(results) - kept - dropped
    print(f"  kept={kept}  filtered_too_small={dropped}  other={other}")
    return kept, dropped


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--min-person-pixels", type=int, default=24 * 24,
                   help="Minimum foreground pixels (>0) in resized mask (default 576 = 24x24)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    root = Path(args.root)
    src_images = root / "images"
    src_masks  = root / "masks"
    dst_images = root / "images_new"
    dst_masks  = root / "masks_new"

    for split in ("train", "val"):
        print(f"\n[{split}]")
        process_split(
            src_img=src_images / split,
            src_mask=src_masks / split,
            dst_img=dst_images / split,
            dst_mask=dst_masks / split,
            image_size=args.image_size,
            min_pixels=args.min_person_pixels,
            workers=args.workers,
        )

    print("\nSwapping directories...")
    shutil.rmtree(src_images)
    shutil.rmtree(src_masks)
    dst_images.rename(src_images)
    dst_masks.rename(src_masks)
    print(f"Done. Dataset rebuilt at {root}/images/ and {root}/masks/")
    print(f"  image_size={args.image_size}  min_person_pixels={args.min_person_pixels}")


if __name__ == "__main__":
    main()
