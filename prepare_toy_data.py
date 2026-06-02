#!/usr/bin/env python3
"""Generate a small synthetic dataset (images + person-like masks) at 128x128.

Creates directory structure:
  <root>/images/train  <root>/images/val
  <root>/masks/train   <root>/masks/val

This is intended for quick local testing when COCO data is not available.
"""
import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw


def make_person_mask(size=(128, 128)):
    w, h = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    # Draw a simple human-like shape: ellipse for head + rectangle/ellipse body
    # Randomize position and scale
    cx = random.randint(w // 4, 3 * w // 4)
    cy = random.randint(h // 4, 3 * h // 4)
    head_r = random.randint(w // 16, w // 8)
    draw.ellipse([cx - head_r, cy - head_r - 10, cx + head_r, cy + head_r - 10], fill=255)
    body_w = head_r * 2
    body_h = random.randint(h // 4, h // 2)
    draw.rectangle([cx - body_w // 2, cy, cx + body_w // 2, cy + body_h], fill=255)
    # optionally add limbs
    if random.random() > 0.5:
        draw.line([cx, cy + body_h // 2, cx - body_w, cy + body_h], fill=255, width=3)
        draw.line([cx, cy + body_h // 2, cx + body_w, cy + body_h], fill=255, width=3)
    return mask


def make_image_from_mask(mask: Image.Image):
    # create a colored background and paste a colored ellipse where mask is
    w, h = mask.size
    bg = Image.new("RGB", mask.size, tuple(random.randint(50, 200) for _ in range(3)))
    fg = Image.new("RGB", mask.size, tuple(random.randint(0, 255) for _ in range(3)))
    # use mask as alpha to composite foreground over background
    img = Image.composite(fg, bg, mask)
    return img


def generate(root: Path, n_train: int = 32, n_val: int = 8, seed: int = 42):
    random.seed(seed)
    out_images_train = root / "images" / "train"
    out_images_val = root / "images" / "val"
    out_masks_train = root / "masks" / "train"
    out_masks_val = root / "masks" / "val"
    for d in [out_images_train, out_images_val, out_masks_train, out_masks_val]:
        d.mkdir(parents=True, exist_ok=True)

    for i in range(n_train):
        mask = make_person_mask()
        img = make_image_from_mask(mask)
        img.save(out_images_train / f"train_{i:04d}.jpg", quality=90)
        mask.save(out_masks_train / f"train_{i:04d}.png")

    for i in range(n_val):
        mask = make_person_mask()
        img = make_image_from_mask(mask)
        img.save(out_images_val / f"val_{i:04d}.jpg", quality=90)
        mask.save(out_masks_val / f"val_{i:04d}.png")

    print(f"Generated {n_train} train and {n_val} val samples under {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", help="dataset root")
    parser.add_argument("--train", type=int, default=32, help="number of training samples to generate")
    parser.add_argument("--val", type=int, default=8, help="number of validation samples to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(Path(args.root), args.train, args.val, args.seed)
