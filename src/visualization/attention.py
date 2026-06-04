#!/usr/bin/env python3
"""
Visualize multi-scale attention maps from sf_seg checkpoints.

Displays per-scale attention (small / medium / large head) and individual
channel maps for the selected head, alongside the colorised GT and predicted
class masks.

Usage:
    python visualize_attention.py
    python visualize_attention.py --checkpoint checkpoints/sf_seg_best.pt \
        --num-images 6 --show-channels 8 --head large --min-range 0.05
"""
import argparse
import colorsys
import json
import random
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.models import sf_seg


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_palette(n: int) -> np.ndarray:
    pal = np.zeros((n, 3), dtype=np.uint8)
    for i in range(1, n):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75 + 0.10 * (i % 3), 0.85 + 0.05 * (i % 2))
        pal[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return pal


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> sf_seg:
    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt \
            else ckpt

    if isinstance(ckpt, dict):
        num_channels   = ckpt.get("num_channels",   64)
        focus_size     = ckpt.get("focus_size",      32)
        encoder_stride = ckpt.get("encoder_stride",   2)
        num_classes    = ckpt.get("num_classes",      81)
    else:
        num_channels, focus_size, encoder_stride, num_classes = 64, 32, 2, 81

    # Fallback: infer num_channels from weight shape if missing from checkpoint
    for key, w in state.items():
        if "enc2.weight" in key and "head_large" in key:
            num_channels = w.shape[1]   # enc2: Conv(C→2C), in_channels = C
            break

    model = sf_seg(num_channels=num_channels, focus_size=focus_size,
                   encoder_stride=encoder_stride, num_classes=num_classes)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded checkpoint: num_channels={num_channels}  focus_size={focus_size}  "
          f"num_classes={num_classes}  params={model.get_num_parameters():,}")
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_model(model: sf_seg, img_t: torch.Tensor, device: torch.device):
    """
    Returns:
        logits       : (num_classes, H, W) — raw logits on CPU
        attn_small   : (C, H/32, W/32)
        attn_medium  : (C, H/8,  W/8)
        attn_large   : (C, H/2,  W/2)
    """
    x = img_t.unsqueeze(0).to(device)
    H, W = x.shape[2], x.shape[3]

    import torch.nn.functional as F

    x_s = F.interpolate(x, size=(max(H // 16, 2), max(W // 16, 2)),
                         mode='bilinear', align_corners=False)
    x_m = F.interpolate(x, size=(H // 4, W // 4),
                         mode='bilinear', align_corners=False)

    _, attn_s = model.head_small(x_s)
    _, attn_m = model.head_medium(x_m)
    _, attn_l = model.head_large(x)

    logits, _, _ = model(x)

    return (logits.squeeze(0).cpu(),
            attn_s.squeeze(0).cpu(),
            attn_m.squeeze(0).cpu(),
            attn_l.squeeze(0).cpu())


# ── Sample loading ────────────────────────────────────────────────────────────

def load_sample(img_path: Path, mask_path: Path, image_size: int, num_classes: int):
    img  = Image.open(img_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    mask = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.NEAREST)
    img_t = TF.to_tensor(img)
    if num_classes > 1:
        mask_np = np.array(mask, dtype=np.int32)          # class indices 0..C-1
    else:
        mask_np = (np.array(mask) > 127).astype(np.uint8) # binary
    return img_t, mask_np, img


# ── Channel selection ─────────────────────────────────────────────────────────

def select_channels(attn: torch.Tensor, min_range: float, max_show: int):
    """Pick channels with max-min range >= min_range, sorted descending."""
    rng  = attn.amax(dim=[1, 2]) - attn.amin(dim=[1, 2])   # (C,)
    ok   = (rng >= min_range).nonzero(as_tuple=True)[0]
    if len(ok) == 0:
        return [], [], []
    order  = rng[ok].argsort(descending=True)
    picked = ok[order[:max_show]]
    return picked.tolist(), rng[picked].tolist(), attn.amax(dim=[1,2])[picked].tolist()


# ── Visualise one sample ──────────────────────────────────────────────────────

def visualise_sample(img_path, mask_path, model, device, args, palette, cat_names):
    img_t, mask_np, img_pil = load_sample(img_path, mask_path, args.image_size, model.num_classes)
    logits, attn_s, attn_m, attn_l = run_model(model, img_t, device)

    head_map = {"small": attn_s, "medium": attn_m, "large": attn_l}
    attn_sel = head_map[args.head]

    indices, ranges, maxvals = select_channels(attn_sel, args.min_range, args.show_channels)
    if not indices:
        print(f"  Skip {img_path.name}: no channels with range >= {args.min_range} "
              f"in '{args.head}' head. Try --min-range 0.01 or --head large")
        return False

    # Build GT and PRED images
    if model.num_classes > 1:
        gt_rgb   = palette[mask_np]
        pred_cls = logits.argmax(0).numpy().astype(np.int32)
        pred_rgb = palette[pred_cls]
        pred_vis = pred_rgb
        gt_vis   = gt_rgb
    else:
        import torch
        gt_vis   = (mask_np * 255).astype(np.uint8)
        pred_vis = (torch.sigmoid(logits).squeeze(0).numpy() * 255).astype(np.uint8)

    # Attention summaries (normalised to [0,1])
    def norm(t): return (t - t.min()) / (t.max() - t.min() + 1e-8)

    sum_s = norm(attn_s.sum(0)).numpy()
    sum_m = norm(attn_m.sum(0)).numpy()
    sum_l = norm(attn_l.sum(0)).numpy()

    # Layout: [RGB | GT | PRED | attn_s | attn_m | attn_l | ch0 | ch1 | ...]
    n_fixed = 6
    ncols   = n_fixed + len(indices)
    fig = plt.figure(figsize=(ncols * 2.4, 3.6))
    gs  = gridspec.GridSpec(1, ncols, figure=fig, wspace=0.06)

    def show(ax, data, title, cmap="gray", vmin=None, vmax=None):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(title, fontsize=8, pad=3)
        ax.axis("off")

    show(fig.add_subplot(gs[0]), np.array(img_pil),  "RGB")
    show(fig.add_subplot(gs[1]), gt_vis,              "GT")
    show(fig.add_subplot(gs[2]), pred_vis,            "PRED")
    show(fig.add_subplot(gs[3]), sum_s, "Attn small\n(global)",  cmap="hot")
    show(fig.add_subplot(gs[4]), sum_m, "Attn medium\n(mid)",    cmap="hot")
    show(fig.add_subplot(gs[5]), sum_l, f"Attn large\n(local)",  cmap="hot")

    for col, (ch_idx, rng, mx) in enumerate(zip(indices, ranges, maxvals)):
        amap = attn_sel[ch_idx].numpy()
        ax   = fig.add_subplot(gs[n_fixed + col])
        im   = ax.imshow(amap, cmap="hot", vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f"Ch {ch_idx} [{args.head}]\nrng={rng:.3f}", fontsize=7, pad=2)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Class legend for multi-class
    if model.num_classes > 1:
        pred_ids = sorted(set(pred_cls.flat) - {0})[:6]
        names    = [cat_names.get(str(c), str(c)) if cat_names else str(c) for c in pred_ids]
        legend   = "predicted: " + ", ".join(names) if names else ""
        if legend:
            fig.text(0.01, 0.01, legend, fontsize=7, color='white',
                     bbox=dict(facecolor='black', alpha=0.5, pad=2))

    fig.suptitle(img_path.name, fontsize=9, y=1.01)
    out = Path(args.output_dir) / f"attn_{img_path.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}  ({len(indices)} channels shown from '{args.head}' head)")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def visualize(args):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = load_model(args.checkpoint, device)
    palette  = _make_palette(model.num_classes)

    cat_names = None
    cat_map   = Path(args.data_root) / "cat_to_idx.json"
    if cat_map.exists():
        cat_names = json.load(open(cat_map)).get("idx_to_name", {})

    val_img_dir  = Path(args.data_root) / "images" / "val"
    val_mask_dir = Path(args.data_root) / "masks"  / "val"
    pairs = sorted([
        (p, val_mask_dir / (p.stem + ".png"))
        for p in val_img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        and (val_mask_dir / (p.stem + ".png")).exists()
    ])

    random.seed(args.seed)
    samples  = random.sample(pairs, min(args.num_images, len(pairs)))
    saved, skipped = 0, 0
    for img_path, mask_path in samples:
        ok = visualise_sample(img_path, mask_path, model, device, args, palette, cat_names)
        saved += ok; skipped += not ok

    print(f"\nDone: {saved} saved, {skipped} skipped.")
    if skipped:
        print(f"Try --min-range {args.min_range * 0.5:.3f} or --head large to see more channels.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="checkpoints/sf_seg_best.pt")
    p.add_argument("--data-root",     default="data")
    p.add_argument("--image-size",    type=int,   default=224)
    p.add_argument("--num-images",    type=int,   default=4)
    p.add_argument("--show-channels", type=int,   default=8,
                   help="Max individual channels to show from selected head")
    p.add_argument("--head",          default="large", choices=["small", "medium", "large"],
                   help="Which attention head to show individual channels for")
    p.add_argument("--min-range",     type=float, default=0.05,
                   help="Min spatial range to include a channel (0=all, 0.05=active only)")
    p.add_argument("--output-dir",    default="outputs/attention_vis")
    p.add_argument("--seed",          type=int,   default=42)
    visualize(p.parse_args())
