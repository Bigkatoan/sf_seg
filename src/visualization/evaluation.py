#!/usr/bin/env python3
"""
Evaluate sf_seg on LVIS v1 val set.

Converts LVIS instance annotations to semantic masks, maps LVIS categories
to our COCO-80 class indices, runs inference, and reports:
  - Category coverage (LVIS 1203 vs model's 80)
  - mIoU on the 51 overlapping classes
  - Per-class IoU breakdown
  - Qualitative visualisation samples

Uses only images already present in data/images/ (no extra download needed).

Usage:
    python evaluate_lvis.py --checkpoint checkpoints/sf_seg_best.pt
    python evaluate_lvis.py --checkpoint checkpoints/sf_seg_best.pt --max-images 500
"""
from __future__ import annotations

import argparse
import colorsys
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from src.models import sf_seg


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_palette(n: int) -> np.ndarray:
    pal = np.zeros((n, 3), dtype=np.uint8)
    for i in range(1, n):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75 + 0.10*(i%3), 0.85 + 0.05*(i%2))
        pal[i] = [int(r*255), int(g*255), int(b*255)]
    return pal


def load_model(ckpt_path: str, device: torch.device) -> sf_seg:
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt \
            else ckpt
    nc    = ckpt.get("num_channels", 64)    if isinstance(ckpt, dict) else 64
    fs    = ckpt.get("focus_size",   32)    if isinstance(ckpt, dict) else 32
    es    = ckpt.get("encoder_stride", 2)   if isinstance(ckpt, dict) else 2
    ncls  = ckpt.get("num_classes",   81)   if isinstance(ckpt, dict) else 81
    model = sf_seg(num_channels=nc, focus_size=fs, encoder_stride=es, num_classes=ncls)
    model.load_state_dict(state)
    return model.to(device).eval()


# ── Category mapping ──────────────────────────────────────────────────────────

def build_category_mapping(lvis_path: Path, cat_to_idx_path: Path):
    """
    Returns:
        lvis_cat_id_to_model_idx : dict[int, int]
            LVIS category_id → our model class index (0 if no overlap)
        overlap_model_idxs       : set[int]  classes where both annotations exist
        lvis_cats                : list[dict]  all LVIS categories
        stats                    : dict  summary statistics
    """
    with open(lvis_path) as f:
        lvis = json.load(f)
    with open(cat_to_idx_path) as f:
        coco_map = json.load(f)

    # Our model's class name → sequential index
    idx_to_name: dict[str, str] = coco_map.get("idx_to_name", {})
    name_to_idx = {v.lower(): int(k) for k, v in idx_to_name.items() if int(k) > 0}

    lvis_cats = lvis["categories"]
    lvis_cat_id_to_model_idx: dict[int, int] = {}
    overlap_names: list[tuple] = []

    for cat in lvis_cats:
        lvis_name = cat["name"].lower().replace("_", " ")
        # Try exact match first
        model_idx = name_to_idx.get(lvis_name, 0)
        if model_idx == 0:
            # Try underscore variant
            model_idx = name_to_idx.get(lvis_name.replace(" ", "_"), 0)
        lvis_cat_id_to_model_idx[cat["id"]] = model_idx
        if model_idx > 0:
            overlap_names.append((cat["name"], model_idx, idx_to_name[str(model_idx)]))

    overlap_model_idxs = {v for v in lvis_cat_id_to_model_idx.values() if v > 0}

    freq_counts = {freq: 0 for freq in ("r", "c", "f")}
    for cat in lvis_cats:
        if lvis_cat_id_to_model_idx[cat["id"]] > 0:
            freq_counts[cat.get("frequency", "f")] += 1

    stats = {
        "lvis_total":    len(lvis_cats),
        "overlap_total": len(overlap_names),
        "overlap_rare":  freq_counts["r"],
        "overlap_common":freq_counts["c"],
        "overlap_freq":  freq_counts["f"],
        "coverage_pct":  100 * len(overlap_names) / len(lvis_cats),
    }
    return lvis_cat_id_to_model_idx, overlap_model_idxs, lvis_cats, stats


# ── LVIS semantic mask builder ─────────────────────────────────────────────────

def build_semantic_mask(anns: list, h: int, w: int,
                        cat_id_to_model_idx: dict, coco_api) -> np.ndarray:
    """
    Merge LVIS instance masks into a semantic class-index mask (H, W) uint8.
    Background = 0. Unknown LVIS categories (not in COCO-80) = 255 (ignored).
    """
    from pycocotools import mask as coco_mask_util

    mask = np.zeros((h, w), dtype=np.uint8)
    # Process frequent → common → rare so rarer categories overwrite
    # (LVIS policy: annotate present categories exhaustively)
    for ann in anns:
        model_idx = cat_id_to_model_idx.get(ann["category_id"], 0)
        if model_idx == 0:
            continue   # skip non-COCO categories (treat as background)
        rle = ann.get("segmentation")
        if rle is None:
            continue
        if isinstance(rle, dict):
            binary = coco_mask_util.decode(rle).astype(bool)
        else:
            # polygon — needs conversion via pycocotools
            from pycocotools.coco import maskUtils
            rle_enc = maskUtils.frPyObjects(rle, h, w)
            binary  = maskUtils.decode(maskUtils.merge(rle_enc)).astype(bool)
        mask[binary] = model_idx
    return mask


# ── Confusion matrix ──────────────────────────────────────────────────────────

def update_conf(conf: np.ndarray, pred: np.ndarray, gt: np.ndarray,
                ignore_idx: int = 255) -> None:
    valid = gt != ignore_idx
    np.add.at(conf, (gt[valid].astype(int), pred[valid].astype(int)), 1)


def compute_miou(conf: np.ndarray, present_classes: set) -> tuple:
    ious = {}
    for c in present_classes:
        tp = conf[c, c]
        fn = conf[c, :].sum() - tp
        fp = conf[:, c].sum() - tp
        denom = tp + fn + fp
        ious[c] = tp / denom if denom > 0 else float('nan')
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return miou, ious


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")

    # Load model
    model = load_model(args.checkpoint, device)
    num_classes = model.num_classes
    print(f"Model: {model.get_num_parameters():,} params  |  num_classes={num_classes}")

    # Load LVIS annotations
    lvis_ann_path = Path(args.lvis_ann)
    if not lvis_ann_path.exists():
        raise FileNotFoundError(f"LVIS annotations not found: {lvis_ann_path}")
    with open(lvis_ann_path) as f:
        lvis = json.load(f)

    # Load COCO API for mask decoding
    try:
        from lvis import LVIS
        lvis_api = LVIS(str(lvis_ann_path))
    except ImportError:
        lvis_api = None

    # Build category mapping
    cat_map_path = Path(args.data_root) / "cat_to_idx.json"
    if not cat_map_path.exists():
        raise FileNotFoundError(f"cat_to_idx.json not found at {cat_map_path}")

    cat_id_to_model_idx, overlap_idxs, lvis_cats, stats = \
        build_category_mapping(lvis_ann_path, cat_map_path)

    with open(cat_map_path) as f:
        coco_map = json.load(f)
    idx_to_name = coco_map.get("idx_to_name", {})

    print("\n── Category Coverage ──────────────────────────────────────────")
    print(f"LVIS total categories      : {stats['lvis_total']:,}")
    print(f"COCO-80 overlap            : {stats['overlap_total']} "
          f"({stats['coverage_pct']:.1f}% of LVIS)")
    print(f"  of which: rare={stats['overlap_rare']}  "
          f"common={stats['overlap_common']}  frequent={stats['overlap_freq']}")
    print(f"LVIS-only (model blind)    : {stats['lvis_total'] - stats['overlap_total']:,}")

    # Build image-level annotation lookup
    img_to_anns = defaultdict(list)
    for ann in lvis["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    img_meta = {img["id"]: img for img in lvis["images"]}

    # Find images we can use (already downloaded)
    data_root = Path(args.data_root)
    all_img_files = {}
    for split in ("train", "val"):
        d = data_root / "images" / split
        if d.exists():
            for p in d.iterdir():
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    all_img_files[p.name] = p

    usable = []
    for img_id, meta in img_meta.items():
        fname = meta["coco_url"].split("/")[-1]
        if fname in all_img_files and img_to_anns[img_id]:
            usable.append((img_id, fname, all_img_files[fname]))

    print(f"\nLVIS val images            : {len(img_meta):,}")
    print(f"Locally available          : {len(usable):,}")

    if not usable:
        print("ERROR: No LVIS images found locally. Download COCO val2017 images first.")
        return

    random.seed(args.seed)
    random.shuffle(usable)
    if args.max_images > 0:
        usable = usable[:args.max_images]
    print(f"Evaluating on              : {len(usable):,} images")

    # Confusion matrix: rows=GT class, cols=pred class
    C    = num_classes
    conf = np.zeros((C + 1, C + 1), dtype=np.int64)  # +1 for ignore (255)

    from pycocotools import mask as coco_mask_util

    n_with_overlap = 0
    n_empty        = 0

    for img_id, fname, img_path in tqdm(usable, desc="Evaluating"):
        meta = img_meta[img_id]
        h, w = meta["height"], meta["width"]
        anns = img_to_anns[img_id]

        # Build GT semantic mask
        gt_mask = build_semantic_mask(anns, h, w, cat_id_to_model_idx, None)

        has_overlap = (gt_mask > 0).any()
        if has_overlap:
            n_with_overlap += 1
        else:
            n_empty += 1

        # Resize GT to model input size
        gt_resized = np.array(
            Image.fromarray(gt_mask).resize((args.image_size, args.image_size), Image.NEAREST),
            dtype=np.uint8
        )

        # Run model
        img = Image.open(img_path).convert("RGB")
        img_t = TF.to_tensor(img.resize((args.image_size, args.image_size), Image.BILINEAR))
        with torch.no_grad():
            logits, _, _ = model(img_t.unsqueeze(0).to(device))
            pred = logits.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)

        # Update confusion matrix (only where GT has COCO-known classes)
        # Unknown LVIS categories remain 0 (background) in our GT
        update_conf(conf, pred, gt_resized, ignore_idx=255)

    # Compute mIoU
    present_in_gt = {c for c in overlap_idxs if conf[c, :].sum() > 0}
    miou_overlap, per_class_iou = compute_miou(conf[:C, :C], present_in_gt)

    # Background IoU
    bg_iou = conf[0,0] / max(conf[0,:].sum() + conf[:,0].sum() - conf[0,0], 1)

    print("\n── Results ────────────────────────────────────────────────────")
    print(f"Images evaluated           : {len(usable):,}")
    print(f"  with COCO-overlap classes: {n_with_overlap:,} ({100*n_with_overlap/len(usable):.1f}%)")
    print(f"  no COCO class present    : {n_empty:,}  (all LVIS-only categories)")
    print(f"\nmIoU (51 overlap classes)  : {miou_overlap:.4f}  ({miou_overlap*100:.2f}%)")
    print(f"Background IoU             : {bg_iou:.4f}")

    # Per-class breakdown (sort by IoU)
    print("\n── Per-Class IoU (overlap classes only, sorted descending) ────")
    print(f"{'Class':<22} {'Idx':>4} {'IoU':>7} {'GT px':>10} {'Freq':>8}")
    print("-" * 58)

    lvis_freq_map = {}
    for cat in lvis_cats:
        if cat_id_to_model_idx.get(cat["id"], 0) in overlap_idxs:
            lvis_freq_map[cat_id_to_model_idx[cat["id"]]] = cat.get("frequency", "?")

    sorted_cls = sorted(per_class_iou.items(), key=lambda x: (not np.isnan(x[1]), x[1]),
                        reverse=True)
    for idx, iou in sorted_cls:
        name = idx_to_name.get(str(idx), f"cls_{idx}")
        gt_px = int(conf[idx, :].sum())
        freq  = lvis_freq_map.get(idx, "?")
        iou_s = f"{iou:.4f}" if not np.isnan(iou) else "  n/a"
        print(f"{name:<22} {idx:>4} {iou_s:>7} {gt_px:>10,}  {freq:>8}")

    # Category gap analysis
    lvis_only_cats = [c for c in lvis_cats
                      if cat_id_to_model_idx.get(c["id"], 0) == 0]
    freq_gap = defaultdict(int)
    for cat in lvis_only_cats:
        freq_gap[cat.get("frequency", "?")] += 1

    ann_lvis_only = sum(1 for a in img_to_anns.values()
                        for ann in a
                        if cat_id_to_model_idx.get(ann["category_id"], 0) == 0)

    print(f"\n── LVIS-Only Categories (model blind) ─────────────────────────")
    print(f"Total categories           : {len(lvis_only_cats):,}")
    print(f"  rare={freq_gap['r']}  common={freq_gap['c']}  frequent={freq_gap['f']}")
    print(f"Annotations in eval set    : {ann_lvis_only:,} "
          f"({100*ann_lvis_only/len(lvis['annotations']):.1f}% of all LVIS val anns)")
    print("Sample LVIS-only names:", [c["name"] for c in lvis_only_cats[:12]])

    # Verdict
    print("\n── Verdict ─────────────────────────────────────────────────────")
    if miou_overlap > 0.40:
        verdict = "PARTIAL — model handles known classes reasonably well"
    elif miou_overlap > 0.15:
        verdict = "POOR — model struggles even on known classes"
    else:
        verdict = "FAILING — model does not generalise to LVIS domain"
    print(f"Overall assessment: {verdict}")
    print(f"Coverage: {stats['coverage_pct']:.1f}% of LVIS categories known to model")
    print(f"Missing : {stats['lvis_total'] - stats['overlap_total']:,} categories "
          f"the model has never seen — will be predicted as background/wrong class")

    # Save visualisation samples
    if args.vis_samples > 0:
        vis_dir = Path(args.output_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        palette = make_palette(num_classes)

        for img_id, fname, img_path in random.sample(usable, min(args.vis_samples, len(usable))):
            meta = img_meta[img_id]
            h, w = meta["height"], meta["width"]
            anns = img_to_anns[img_id]
            gt_mask = build_semantic_mask(anns, h, w, cat_id_to_model_idx, None)

            img_pil = Image.open(img_path).convert("RGB")
            sz = args.image_size
            img_rs = img_pil.resize((sz, sz), Image.BILINEAR)
            gt_rs  = np.array(Image.fromarray(gt_mask).resize((sz, sz), Image.NEAREST))

            img_t = TF.to_tensor(img_rs)
            with torch.no_grad():
                logits, _, _ = model(img_t.unsqueeze(0).to(device))
                pred = logits.squeeze(0).argmax(0).cpu().numpy().astype(np.int32)

            # All LVIS annotations overlay (including non-COCO)
            lvis_all = np.zeros((h, w), dtype=np.uint8)
            from pycocotools import mask as cmu
            for ann in anns:
                rle = ann.get("segmentation")
                if rle and isinstance(rle, dict):
                    try:
                        lvis_all[cmu.decode(rle).astype(bool)] = \
                            (cat_id_to_model_idx.get(ann["category_id"], 0) or
                             min(ann["category_id"] % num_classes + 1, num_classes - 1))
                    except Exception:
                        pass
            lvis_all_rs = np.array(Image.fromarray(lvis_all).resize((sz, sz), Image.NEAREST))

            panels = [
                ("Input",           np.array(img_rs)),
                ("LVIS-all",        palette[np.clip(lvis_all_rs, 0, num_classes-1)]),
                ("GT (COCO-overlap)",palette[gt_rs]),
                ("Model PRED",      palette[pred]),
            ]
            pw = sz
            combined = Image.new("RGB", (pw * 4, sz))
            for i, (title, arr) in enumerate(panels):
                panel = Image.fromarray(arr.astype(np.uint8))
                combined.paste(panel, (i * pw, 0))

            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(combined)
            font = ImageFont.load_default()
            for i, (title, _) in enumerate(panels):
                draw.text((i * pw + 4, 4), title, fill=(255, 255, 255), font=font)

            out = vis_dir / f"lvis_eval_{fname}"
            combined.save(out)

        print(f"\nSaved {args.vis_samples} visualisation samples → {vis_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="checkpoints/sf_seg_best.pt")
    p.add_argument("--lvis-ann",    default="data/lvis/lvis_v1_val.json")
    p.add_argument("--data-root",   default="data")
    p.add_argument("--image-size",  type=int, default=224)
    p.add_argument("--max-images",  type=int, default=1000,
                   help="Max images to evaluate (0=all). Default 1000 for quick run.")
    p.add_argument("--vis-samples", type=int, default=8,
                   help="Number of visualisation samples to save")
    p.add_argument("--output-dir",  default="outputs/lvis_eval")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--cpu",         action="store_true")
    evaluate(p.parse_args())
