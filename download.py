#!/usr/bin/env python3
"""
Download COCO2017 using `aria2c` (if available) and prepare dataset
ready for training.

This script will:
- Download `train2017.zip` and `annotations_trainval2017.zip` using aria2c
    (falls back to requests if aria2c is not installed).
- Extract images and annotations, keep only images that contain `person`.
        - Build binary person masks and resize both images and masks to a configurable
            square size (default 480x480).
- Produce the folder layout:
        <root>/images/train/...
        <root>/images/val/...
        <root>/masks/train/...
        <root>/masks/val/...

Usage examples:
    # download and prepare full train set (may be large)
    python download.py --root data --prepare

    # use a smaller subset for quick experiments
    python download.py --root data --prepare --max-images 200

Notes:
- `aria2c` is recommended for fast parallel downloads (install via system package manager).
- Building masks requires `pycocotools`, `Pillow`, and `numpy`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

try:
    from urllib.parse import urljoin
except Exception:
    from urlparse import urljoin

import requests
from tqdm import tqdm
import subprocess
import random
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os


COCO_BASE = "http://images.cocodataset.org/"
URLS = {
    "train_zip": urljoin(COCO_BASE, "zips/train2017.zip"),
    "val_zip": urljoin(COCO_BASE, "zips/val2017.zip"),
    "annotations_zip": urljoin(COCO_BASE, "annotations/annotations_trainval2017.zip"),
}


def download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def aria2c_download(url: str, dest: Path) -> None:
    """Use aria2c for fast parallel download; fall back to `download_file` if aria2c missing."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    aria = shutil.which("aria2c")
    if aria:
        cmd = [aria, "-x", "16", "-s", "16", "-k", "1M", "-d", str(dest.parent), "-o", dest.name, url]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    else:
        print("aria2c not found; falling back to requests for:", url)
        download_file(url, dest)


def extract_zip(zip_path: Path, out_dir: Path, members=None) -> None:
    with zipfile.ZipFile(zip_path, "r") as z:
        if members:
            z.extractall(path=out_dir, members=members)
        else:
            z.extractall(path=out_dir)


def download_annotations(root: Path) -> Path:
    """Download annotations zip and return path to instances_train2017.json"""
    ann_zip = root / "annotations_trainval2017.zip"
    if not ann_zip.exists():
        print("Downloading annotations (small zip)...")
        aria2c_download(URLS["annotations_zip"], ann_zip)
    tmp_dir = root / "annotations_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting annotations...")
    extract_zip(ann_zip, tmp_dir)
    # common path inside zip: annotations/instances_train2017.json
    inst = tmp_dir / "annotations" / "instances_train2017.json"
    if not inst.exists():
        raise FileNotFoundError("instances_train2017.json not found in annotations zip")
    return inst


def get_all_image_infos(inst_json_path: Path) -> dict:
    """Return dict image_id -> image_info for all annotated images, plus category mapping."""
    print("Loading annotation JSON (may be large)...")
    with open(inst_json_path, "r") as f:
        ann = json.load(f)

    # Build category ID -> sequential class index (1-based, 0=background)
    cat_ids = sorted(c["id"] for c in ann.get("categories", []))
    cat_id_to_idx = {cid: i + 1 for i, cid in enumerate(cat_ids)}
    idx_to_name = {0: "background"}
    for c in ann.get("categories", []):
        idx_to_name[cat_id_to_idx[c["id"]]] = c["name"]

    img_map = {img["id"]: img for img in ann.get("images", [])}
    image_ids = set()
    for a in ann.get("annotations", []):
        image_ids.add(a.get("image_id"))

    info = {iid: img_map[iid] for iid in image_ids if iid in img_map}
    return {"cat_id_to_idx": cat_id_to_idx, "idx_to_name": idx_to_name, "images": info}


def download_image_file(file_name: str, split: str, out_dir: Path) -> Path:
    url = urljoin(COCO_BASE, f"{split}2017/{file_name}")
    dest = out_dir / file_name
    if not dest.exists():
        aria2c_download(url, dest)
    return dest


def build_all_class_masks(inst_json_path: Path, selected_images: dict,
                          cat_id_to_idx: dict, out_mask_dir: Path) -> None:
    """Build multi-class masks where each pixel = class index (0=background, 1-80=COCO class)."""
    try:
        from pycocotools.coco import COCO
    except Exception as e:
        print("pycocotools not installed; cannot build masks:", e)
        return

    coco = COCO(str(inst_json_path))
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    for img_id, info in tqdm(selected_images.items(), desc="Building masks"):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        if not anns:
            continue
        h, w = info["height"], info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)  # 0 = background
        # Sort by area descending so smaller objects paint on top
        for ann in sorted(anns, key=lambda a: a.get("area", 0), reverse=True):
            class_idx = cat_id_to_idx.get(ann.get("category_id"), 0)
            if class_idx == 0:
                continue
            m = coco.annToMask(ann).astype(bool)
            mask[m] = class_idx
        mask_img = Image.fromarray(mask)  # uint8 grayscale, values 0..num_classes-1
        name = info["file_name"].rsplit('.', 1)[0] + ".png"
        mask_img.save(out_mask_dir / name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", help="dataset root; final layout: <root>/images/<train|val>/ and <root>/masks/<train|val>/")
    parser.add_argument("--prepare", action="store_true", help="Download zips (aria2c) and prepare resized dataset")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of person images to process (0 = all)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers to process images (default=all CPU cores)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction to reserve for validation")
    parser.add_argument("--image-size", type=int, default=480, help="Square size to which images and masks are resized (default 480)")
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw extracted COCO folders (don't delete)")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    if not args.prepare:
        print("Nothing to do. Use --prepare to download and build dataset.")
        return

    # download zips
    print("Downloading annotations zip...")
    ann_zip = root / "annotations_trainval2017.zip"
    if not ann_zip.exists():
        aria2c_download(URLS["annotations_zip"], ann_zip)

    print("Downloading train2017 images zip (this is large)...")
    train_zip = root / "train2017.zip"
    if not train_zip.exists():
        aria2c_download(URLS["train_zip"], train_zip)

    # extract into a temp raw folder
    raw_dir = root / "coco_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting annotations...")
    extract_zip(ann_zip, raw_dir)
    print("Extracting train images (this may take a while)...")
    extract_zip(train_zip, raw_dir)

    inst = raw_dir / "annotations" / "instances_train2017.json"
    if not inst.exists():
        raise FileNotFoundError("instances_train2017.json not found after extracting annotations")

    # read annotations and collect all annotated images
    print("Parsing annotations for all categories...")
    with open(inst, "r") as f:
        ann = json.load(f)

    cat_ids = sorted(c["id"] for c in ann.get("categories", []))
    cat_id_to_idx = {cid: i + 1 for i, cid in enumerate(cat_ids)}
    idx_to_name = {0: "background"}
    for c in ann.get("categories", []):
        idx_to_name[cat_id_to_idx[c["id"]]] = c["name"]

    img_map = {img["id"]: img for img in ann.get("images", [])}
    image_ids = set()
    for a in ann.get("annotations", []):
        image_ids.add(a.get("image_id"))
    images_info = {iid: img_map[iid] for iid in image_ids if iid in img_map}
    print(f"Found {len(images_info)} annotated images across {len(cat_ids)} categories")

    # Save category mapping
    mapping_path = root / "cat_to_idx.json"
    with open(mapping_path, "w") as f:
        json.dump({
            "cat_id_to_idx": {str(k): v for k, v in cat_id_to_idx.items()},
            "idx_to_name":   {str(k): v for k, v in idx_to_name.items()},
            "num_classes":   len(cat_ids) + 1,
        }, f, indent=2)
    print(f"Saved category mapping → {mapping_path}")

    # limit if requested
    img_items = list(images_info.items())
    if args.max_images and args.max_images > 0:
        img_items = img_items[: args.max_images]

    # split into train / val
    random.shuffle(img_items)
    n_val = max(1, int(len(img_items) * args.val_split))
    val_items = dict(img_items[:n_val])
    train_items = dict(img_items[n_val:])

    images_src_dir = raw_dir / "train2017"
    out_images_train = root / "images" / "train"
    out_images_val = root / "images" / "val"
    out_masks_train = root / "masks" / "train"
    out_masks_val = root / "masks" / "val"
    for d in [out_images_train, out_images_val, out_masks_train, out_masks_val]:
        d.mkdir(parents=True, exist_ok=True)

    # build dataset (images + masks), resizing to requested image size
    try:
        from pycocotools.coco import COCO
    except Exception:
        print("pycocotools not installed; please pip install pycocotools")
        return

    coco = COCO(str(inst))
    def process_items(items: dict, split: str, workers: int = None, image_size: int = 480):
        if not items:
            return
        if workers is None or workers <= 0:
            workers = os.cpu_count() or 4

        def worker(entry):
            img_id, info = entry
            fname = info["file_name"]
            src = images_src_dir / fname
            if not src.exists():
                return (fname, "missing", None)
            try:
                anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
                if not anns:
                    return (fname, "no_anns", None)
                h, w = info["height"], info["width"]
                mask = np.zeros((h, w), dtype=np.uint8)  # 0 = background
                # Sort by area desc so smaller objects paint over larger ones
                for ann_item in sorted(anns, key=lambda a: a.get("area", 0), reverse=True):
                    class_idx = cat_id_to_idx.get(ann_item.get("category_id"), 0)
                    if class_idx == 0:
                        continue
                    m = coco.annToMask(ann_item).astype(bool)
                    mask[m] = class_idx

                img = Image.open(src).convert("RGB")
                img_resized  = img.resize((image_size, image_size), Image.BILINEAR)
                mask_resized = Image.fromarray(mask).resize((image_size, image_size), Image.NEAREST)

                out_img_dir  = out_images_train if split == "train" else out_images_val
                out_mask_dir = out_masks_train  if split == "train" else out_masks_val
                img_resized.save(out_img_dir / fname)
                mask_resized.save(out_mask_dir / (fname.rsplit(".", 1)[0] + ".png"))
                return (fname, "ok", None)
            except Exception as e:
                return (fname, "err", str(e))

        entries = list(items.items())
        results = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for res in tqdm(ex.map(worker, entries), total=len(entries), desc=f"Processing {split}"):
                results.append(res)
        # optionally, report errors
        errs = [r for r in results if r[1] == "err"]
        if errs:
            print(f"{len(errs)} errors during processing {split}, first example:", errs[0])

    workers = args.workers
    process_items(train_items, "train", workers=workers, image_size=args.image_size)
    process_items(val_items, "val", workers=workers, image_size=args.image_size)

    if not args.keep_raw:
        print("Removing raw extracted files to save space...")
        try:
            shutil.rmtree(raw_dir)
        except Exception as e:
            print("Failed to remove raw dir:", e)

    print("Done. Prepared dataset in:")
    print(" -", root / "images")
    print(" -", root / "masks")
    # end of preparation


if __name__ == "__main__":
    main()
