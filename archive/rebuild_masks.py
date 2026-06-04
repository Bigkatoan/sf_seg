#!/usr/bin/env python3
"""
Rebuild multi-class segmentation masks from COCO annotations for existing processed images.

Each mask pixel value = class index (0=background, 1-80=sequential COCO class index).
A category mapping is saved to <root>/cat_to_idx.json.

Usage:
    python rebuild_masks.py
    python rebuild_masks.py --root data \
        --ann data/coco2017/annotations_tmp/annotations/instances_train2017.json \
        --workers 8
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def build_fname_to_id(coco) -> dict:
    """Build filename → image_id lookup from a COCO object."""
    d: dict[str, int] = {}
    for info in coco.loadImgs(coco.getImgIds()):
        d[info["file_name"]]                    = info["id"]
        d[info["file_name"].rsplit(".", 1)[0]]  = info["id"]
    return d


def rebuild_split(split: str, img_dir: Path, mask_dir: Path,
                  coco, cat_id_to_idx: dict, img_fname_to_id: dict,
                  workers: int):
    mask_dir.mkdir(parents=True, exist_ok=True)
    img_files = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"\n[{split}] {len(img_files)} images → {mask_dir}")

    stats = {"ok": 0, "no_ann": 0, "missing": 0, "err": 0}

    def worker(img_path: Path) -> str:
        img_id = img_fname_to_id.get(img_path.name) or img_fname_to_id.get(img_path.stem)
        if img_id is None:
            return "missing"
        try:
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns    = coco.loadAnns(ann_ids)
            if not anns:
                # No annotations → save blank background mask
                target_size = Image.open(img_path).size
                Image.fromarray(np.zeros(target_size[::-1], dtype=np.uint8)).save(
                    mask_dir / (img_path.stem + ".png")
                )
                return "no_ann"

            img_info    = coco.loadImgs([img_id])[0]
            h, w        = img_info["height"], img_info["width"]
            target_size = Image.open(img_path).size  # (W, H) of already-processed image

            mask = np.zeros((h, w), dtype=np.uint8)
            # Sort by area descending so smaller objects paint on top
            for ann in sorted(anns, key=lambda a: a.get("area", 0), reverse=True):
                class_idx = cat_id_to_idx.get(ann.get("category_id"), 0)
                if class_idx == 0:
                    continue
                m = coco.annToMask(ann).astype(bool)
                mask[m] = class_idx

            mask_resized = Image.fromarray(mask).resize(target_size, Image.NEAREST)
            mask_resized.save(mask_dir / (img_path.stem + ".png"))
            return "ok"
        except Exception as e:
            return "err"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in tqdm(ex.map(worker, img_files), total=len(img_files)):
            stats[r] = stats.get(r, 0) + 1

    print(f"  ok={stats['ok']}  no_ann={stats['no_ann']}  "
          f"missing={stats['missing']}  err={stats['err']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",    default="data")
    p.add_argument("--ann",
                   default="data/coco2017/annotations_tmp/annotations/instances_train2017.json",
                   help="COCO annotations for the train split")
    p.add_argument("--val-ann",
                   default="data/coco2017/annotations_tmp/annotations/instances_val2017.json",
                   help="COCO annotations for the val split (default: instances_val2017.json)")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    root     = Path(args.root)
    ann_path = Path(args.ann)

    if not ann_path.exists():
        raise FileNotFoundError(f"Train annotations not found: {ann_path}")

    try:
        from pycocotools.coco import COCO
    except ImportError:
        raise ImportError("pycocotools not installed; run: pip install pycocotools")

    # Load train annotations — used to derive the shared category mapping
    print(f"Loading train annotations from {ann_path} ...")
    coco_train = COCO(str(ann_path))

    # Build category ID -> sequential class index (1-based, 0=background)
    cat_ids       = sorted(coco_train.getCatIds())
    cat_id_to_idx = {cid: i + 1 for i, cid in enumerate(cat_ids)}
    idx_to_name   = {0: "background"}
    for i, cid in enumerate(cat_ids):
        idx_to_name[i + 1] = coco_train.loadCats([cid])[0]["name"]

    num_classes = len(cat_ids) + 1
    print(f"Categories: {len(cat_ids)} foreground + 1 background = {num_classes} total")

    # Save mapping
    mapping_path = root / "cat_to_idx.json"
    with open(mapping_path, "w") as f:
        json.dump({
            "cat_id_to_idx": {str(k): v for k, v in cat_id_to_idx.items()},
            "idx_to_name":   {str(k): v for k, v in idx_to_name.items()},
            "num_classes":   num_classes,
        }, f, indent=2)
    print(f"Saved category mapping → {mapping_path}")

    # Determine which COCO object to use for each split
    split_coco: dict[str, object] = {"train": coco_train}

    val_ann_path = Path(args.val_ann)
    if val_ann_path.exists():
        print(f"Loading val annotations from {val_ann_path} ...")
        split_coco["val"] = COCO(str(val_ann_path))
    else:
        print(f"Val annotations not found ({val_ann_path}); falling back to train annotations for val split.")
        split_coco["val"] = coco_train

    for split in ("train", "val"):
        img_dir  = root / "images" / split
        mask_dir = root / "masks"  / split
        if not img_dir.exists():
            print(f"[{split}] images dir not found, skipping: {img_dir}")
            continue
        coco_split = split_coco[split]
        fname_to_id = build_fname_to_id(coco_split)
        rebuild_split(split, img_dir, mask_dir, coco_split, cat_id_to_idx,
                      fname_to_id, workers=args.workers)

    print(f"\nDone.  Mask values: 0=background, 1-{len(cat_ids)}=COCO classes")
    print(f"Set  num_classes={num_classes}  in config.json before training.")


if __name__ == "__main__":
    main()
