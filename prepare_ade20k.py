#!/usr/bin/env python3
"""
Download (if needed) and prepare ADE20K Challenge 2016 for sf_seg.

Source : https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
Output :
    data/images/train/   20,210 RGB JPEGs  (resized to --image-size)
    data/images/val/      2,000 RGB JPEGs
    data/masks/train/    20,210 uint8 PNGs  (class indices 0-150, nearest resize)
    data/masks/val/       2,000 uint8 PNGs
    data/cat_to_idx.json  ADE20K 150-class name mapping

Mask values:  0 = background (unlabelled)
              1-150 = ADE20K semantic class index (same as objectInfo150.txt)

Usage:
    python prepare_ade20k.py                        # use already-downloaded zip
    python prepare_ade20k.py --download             # download first
    python prepare_ade20k.py --image-size 224 --workers 8
    python prepare_ade20k.py --download --clean-zip # delete zip after extract
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ADE20K_URL  = "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
ADE20K_ZIP  = "data/ade20k/ADEChallengeData2016.zip"
ADE20K_ROOT = "data/ade20k/ADEChallengeData2016"


# ── Download ──────────────────────────────────────────────────────────────────

def download(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    aria = shutil.which("aria2c")
    if aria:
        cmd = [aria, "-x", "16", "-s", "16", "-k", "1M", "-c",
               "-d", str(dest.parent), "-o", dest.name, ADE20K_URL]
        print("Downloading with aria2c …")
        subprocess.run(cmd, check=True)
    else:
        import requests
        print("Downloading with requests (aria2c not found) …")
        with requests.get(ADE20K_URL, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk); bar.update(len(chunk))


# ── Class mapping ─────────────────────────────────────────────────────────────

def build_cat_to_idx(obj_info_path: Path) -> dict:
    idx_to_name = {"0": "background"}
    cat_id_to_idx = {}
    for line in obj_info_path.read_text().strip().split("\n")[1:]:
        parts = line.split()
        idx  = int(parts[0])
        name = " ".join(parts[4:])
        idx_to_name[str(idx)]  = name
        cat_id_to_idx[str(idx)] = idx
    return {
        "cat_id_to_idx": cat_id_to_idx,
        "idx_to_name":   idx_to_name,
        "num_classes":   151,
        "dataset":       "ADE20K-150",
    }


# ── Prepare one split ─────────────────────────────────────────────────────────

def prepare_split(src_img_dir: Path, src_mask_dir: Path,
                  dst_img_dir: Path, dst_mask_dir: Path,
                  image_size: int, workers: int) -> tuple[int, int]:
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(src_img_dir.glob("*.jpg"))

    def worker(img_path: Path) -> str:
        stem      = img_path.stem
        mask_path = src_mask_dir / (stem + ".png")
        if not mask_path.exists():
            return "no_mask"
        try:
            img  = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)                  # palette/L mode, uint8 class index

            img_r  = img.resize((image_size, image_size), Image.BILINEAR)
            mask_r = mask.resize((image_size, image_size), Image.NEAREST)

            img_r.save(dst_img_dir / img_path.name)
            mask_r.save(dst_mask_dir / (stem + ".png"))

            # Sanity: verify mask values stay in [0, 150]
            arr = np.array(mask_r)
            if arr.max() > 150:
                return "warn_overflow"
            return "ok"
        except Exception as e:
            return f"err:{e}"

    results: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in tqdm(ex.map(worker, images), total=len(images),
                      desc=f"  {src_img_dir.parent.name}/{src_img_dir.name}"):
            results.append(r)

    ok   = results.count("ok")
    warn = sum(1 for r in results if r.startswith("warn"))
    err  = sum(1 for r in results if r.startswith("err"))
    print(f"    ok={ok}  warn={warn}  err={err}  total={len(results)}")
    return ok, err


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",        default="data",
                    help="Dataset root (default: data)")
    ap.add_argument("--ade-zip",     default=ADE20K_ZIP,
                    help="Path to ADE20K zip (used if --download not set)")
    ap.add_argument("--image-size",  type=int, default=224)
    ap.add_argument("--workers",     type=int, default=8)
    ap.add_argument("--download",    action="store_true",
                    help="Download zip if not present")
    ap.add_argument("--clean-zip",   action="store_true",
                    help="Delete zip after extraction to save space")
    ap.add_argument("--clean-coco",  action="store_true",
                    help="Delete previous data/images and data/masks (COCO data)")
    args = ap.parse_args()

    root    = Path(args.root)
    zip_path = Path(args.ade_zip)
    raw_root = Path(ADE20K_ROOT)

    # ── Download ──────────────────────────────────────────────────────────────
    if args.download and not zip_path.exists():
        download(zip_path)

    # ── Extract ───────────────────────────────────────────────────────────────
    if not raw_root.exists():
        if not zip_path.exists():
            raise FileNotFoundError(
                f"ADE20K zip not found at {zip_path}.\n"
                f"Run: python prepare_ade20k.py --download")
        print(f"Extracting {zip_path} …")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(zip_path.parent)
        print("Extraction complete.")
        if args.clean_zip:
            zip_path.unlink()
            print(f"Deleted {zip_path}")

    # ── Verify source ─────────────────────────────────────────────────────────
    for sub in ("images/training", "images/validation",
                "annotations/training", "annotations/validation"):
        if not (raw_root / sub).exists():
            raise FileNotFoundError(f"Expected directory not found: {raw_root / sub}")

    # ── Optional: remove old COCO data ────────────────────────────────────────
    if args.clean_coco:
        for d in (root / "images", root / "masks"):
            if d.exists():
                print(f"Removing {d} …")
                shutil.rmtree(d)

    # ── Class mapping ─────────────────────────────────────────────────────────
    obj_info = raw_root / "objectInfo150.txt"
    cat_map  = build_cat_to_idx(obj_info)
    cat_map_path = root / "cat_to_idx.json"
    with open(cat_map_path, "w") as f:
        json.dump(cat_map, f, indent=2)
    print(f"Saved class mapping → {cat_map_path}  ({cat_map['num_classes']} classes)")

    # ── Invalidate COCO class-frequency cache ─────────────────────────────────
    freq_cache = root / "class_freq.json"
    if freq_cache.exists():
        freq_cache.unlink()
        print(f"Removed stale class-frequency cache: {freq_cache}")

    # ── Prepare splits ────────────────────────────────────────────────────────
    split_map = {
        "train": ("training",   "training"),
        "val":   ("validation", "validation"),
    }
    for split, (img_sub, mask_sub) in split_map.items():
        print(f"\n[{split}]")
        prepare_split(
            src_img_dir  = raw_root / "images"      / img_sub,
            src_mask_dir = raw_root / "annotations" / mask_sub,
            dst_img_dir  = root / "images" / split,
            dst_mask_dir = root / "masks"  / split,
            image_size   = args.image_size,
            workers      = args.workers,
        )

    print(f"\nDone. Dataset ready at {root}/images/ and {root}/masks/")
    print(f"  image_size={args.image_size}  num_classes=151  (0=background, 1-150=ADE20K)")
    print(f"\nUpdate config.json:  \"num_classes\": 151")


if __name__ == "__main__":
    main()
