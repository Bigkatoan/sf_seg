"""noise_aug.py — visualise augmentation presets on real training images.

Usage:
    python noise_aug.py                        # saves grid to aug_samples/
    python noise_aug.py --sigma 0.03 0.06      # override gaussian sigma range
    python noise_aug.py --brightness 0.6 1.4   # override brightness range
    python noise_aug.py --n 6                  # number of source images
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont


# ── Augment presets ────────────────────────────────────────────────────────────

PRESETS = {
    "none":             dict(noise=0.00, brightness=(1.0, 1.0), contrast=(1.0, 1.0)),
    "light":            dict(noise=0.02, brightness=(0.85, 1.15), contrast=(0.90, 1.10)),
    "medium (default)": dict(noise=0.05, brightness=(0.75, 1.25), contrast=(0.80, 1.20)),
    "heavy":            dict(noise=0.08, brightness=(0.60, 1.40), contrast=(0.70, 1.30)),
    "noise only":       dict(noise=0.05, brightness=(1.00, 1.00), contrast=(1.00, 1.00)),
    "light+noise":      dict(noise=0.03, brightness=(0.80, 1.20), contrast=(0.85, 1.15)),
}


def apply_aug(img_t: torch.Tensor, noise: float,
              brightness: tuple, contrast: tuple) -> torch.Tensor:
    """img_t: (3, H, W) float in [0,1]. Returns augmented tensor."""
    t = img_t.clone()
    if noise > 0:
        t = (t + torch.randn_like(t) * noise).clamp(0.0, 1.0)
    b = random.uniform(*brightness)
    t = (t * b).clamp(0.0, 1.0)
    mean = t.mean()
    c = random.uniform(*contrast)
    t = ((t - mean) * c + mean).clamp(0.0, 1.0)
    return t


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def label_image(img: Image.Image, text: str, font_size: int = 12) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, out.width, font_size + 4], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 0), font=font)
    return out


def make_grid(images: list[Image.Image], cols: int) -> Image.Image:
    w, h = images[0].size
    rows  = (len(images) + cols - 1) // cols
    grid  = Image.new("RGB", (w * cols, h * rows), (30, 30, 30))
    for i, img in enumerate(images):
        grid.paste(img, (i % cols * w, i // cols * h))
    return grid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root",  default="data")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--n",          type=int, default=4,
                    help="Number of source images to sample")
    ap.add_argument("--sigma",      type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"),
                    help="Override gaussian noise sigma range (e.g. --sigma 0.02 0.06)")
    ap.add_argument("--brightness", type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"))
    ap.add_argument("--contrast",   type=float, nargs=2, default=None,
                    metavar=("MIN", "MAX"))
    ap.add_argument("--out",        default="aug_samples")
    args = ap.parse_args()

    train_dir = Path(args.data_root) / "images" / "train"
    if not train_dir.exists():
        print(f"[error] {train_dir} not found — set --data-root")
        return

    imgs_paths = sorted(train_dir.glob("*.jpg")) + sorted(train_dir.glob("*.png"))
    if not imgs_paths:
        print(f"[error] no images in {train_dir}")
        return

    random.seed(42)
    sample_paths = random.sample(imgs_paths, min(args.n, len(imgs_paths)))
    sz = args.image_size
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Override presets if CLI args provided
    presets = dict(PRESETS)
    if any([args.sigma, args.brightness, args.contrast]):
        presets["custom"] = dict(
            noise=args.sigma[1] if args.sigma else 0.05,
            brightness=tuple(args.brightness) if args.brightness else (0.75, 1.25),
            contrast=tuple(args.contrast)   if args.contrast   else (0.80, 1.20),
        )

    for preset_name, cfg in presets.items():
        cells: list[Image.Image] = []
        for path in sample_paths:
            img = Image.open(path).convert("RGB").resize((sz, sz), Image.BILINEAR)
            img_t = TF.to_tensor(img).float()
            aug_t = apply_aug(img_t,
                              noise=cfg["noise"],
                              brightness=cfg["brightness"],
                              contrast=cfg["contrast"])
            # Original | Augmented side by side
            left  = label_image(img, "original")
            right = label_image(tensor_to_pil(aug_t), "augmented")
            pair  = Image.new("RGB", (sz * 2, sz))
            pair.paste(left,  (0,  0))
            pair.paste(right, (sz, 0))
            cells.append(pair)

        grid = make_grid(cells, cols=2)
        # Add preset info banner at bottom
        banner_h = 28
        final = Image.new("RGB", (grid.width, grid.height + banner_h), (20, 20, 20))
        final.paste(grid, (0, 0))
        draw  = ImageDraw.Draw(final)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        info = (f"{preset_name} | noise σ={cfg['noise']:.3f} | "
                f"brightness {cfg['brightness']} | contrast {cfg['contrast']}")
        draw.text((8, grid.height + 6), info, fill=(200, 200, 200), font=font)

        fname = preset_name.replace(" ", "_").replace("(", "").replace(")", "") + ".png"
        out_path = out_dir / fname
        final.save(out_path)
        print(f"saved → {out_path}")

    print(f"\nAll samples saved to {out_dir}/")
    print("Review them and pick a preset, then update trainer.py sigma/brightness ranges.")


if __name__ == "__main__":
    main()
