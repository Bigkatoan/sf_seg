"""Per-epoch visualization: masks, attention heatmaps, composite overlays.

Folder structure per epoch:
  outputs/epoch_NNNN/
    masks/      — rgb, gt, pred, confidence heatmap
    attn/       — per-head overlay (heatmap blended onto original image)
    composite/  — one image per sample: masks row + all attention overlays
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF


# ── Colormaps ─────────────────────────────────────────────────────────────────

def _jet(x: np.ndarray) -> np.ndarray:
    """
    Jet colormap: blue→cyan→yellow→red.
    Input: (H, W) float [0, 1].  Output: (H, W, 3) uint8.
    """
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * x - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1.0), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


# ── Image utilities ───────────────────────────────────────────────────────────

def _overlay(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Blend a jet heatmap onto an RGB image.
    rgb:  (H, W, 3) uint8
    heat: (H, W, 3) uint8  (jet-coloured)
    Returns (H, W, 3) uint8
    """
    return np.clip(
        (1 - alpha) * rgb.astype(np.float32) + alpha * heat.astype(np.float32),
        0, 255
    ).astype(np.uint8)


def _label(img: Image.Image, text: str, font=None) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    tw = len(text) * 6 + 6
    draw.rectangle([0, 0, tw, 13], fill=(0, 0, 0, 200))
    draw.text((3, 2), text, fill=(255, 255, 255), font=font)
    return out


def _resize_np(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.array(Image.fromarray(arr).resize((w, h), Image.BILINEAR))


def _to_rgb(img_t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float → (H, W, 3) uint8"""
    return (img_t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _upsample(sal: torch.Tensor, H: int, W: int) -> np.ndarray:
    """(h, w) float tensor → (H, W) float numpy, bilinear upsample."""
    up = F.interpolate(sal.float().unsqueeze(0).unsqueeze(0),
                       (H, W), mode='bilinear', align_corners=False)[0, 0]
    return up.cpu().numpy()


def _hstack(images: list[np.ndarray], gap: int = 2,
            bg: tuple = (20, 20, 20)) -> np.ndarray:
    H = max(im.shape[0] for im in images)
    W = sum(im.shape[1] for im in images) + gap * (len(images) - 1)
    out = np.full((H, W, 3), bg, dtype=np.uint8)
    x = 0
    for im in images:
        h, w = im.shape[:2]
        out[:h, x:x+w] = im
        x += w + gap
    return out


def _vstack(images: list[np.ndarray], gap: int = 2,
            bg: tuple = (20, 20, 20)) -> np.ndarray:
    W = max(im.shape[1] for im in images)
    H = sum(im.shape[0] for im in images) + gap * (len(images) - 1)
    out = np.full((H, W, 3), bg, dtype=np.uint8)
    y = 0
    for im in images:
        h, w = im.shape[:2]
        out[y:y+h, :w] = im
        y += h + gap
    return out


# ── Semantic mask palette ─────────────────────────────────────────────────────

def _palette(n: int) -> np.ndarray:
    import colorsys
    pal = np.zeros((n, 3), dtype=np.uint8)
    for i in range(1, n):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75 + 0.10 * (i % 3), 0.85 + 0.05 * (i % 2))
        pal[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return pal


# ── Core per-sample function ──────────────────────────────────────────────────

def visualize_sample(
    img_t:      torch.Tensor,   # (3, H, W) float [0,1]
    mask_t:     torch.Tensor,   # (H, W) long
    logits:     torch.Tensor,   # (C, H, W)
    attns:      dict,           # {'tiny':…,'small':…,'medium':…,'large':…}
    out_dir:    Path,
    idx:        int,
    num_classes: int,
    font=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W  = img_t.shape[1], img_t.shape[2]
    pal   = _palette(num_classes)
    rgb   = _to_rgb(img_t)                          # (H,W,3) uint8

    # ── Prediction ────────────────────────────────────────────────────────────
    probs       = torch.softmax(logits.float(), dim=0)
    conf, pred  = probs.max(dim=0)
    pred_np     = pred.cpu().numpy().astype(np.int32)
    gt_np       = mask_t.cpu().numpy().astype(np.int32)
    conf_np     = _norm01(conf.cpu().numpy())

    gt_rgb   = pal[gt_np]
    pred_rgb = pal[pred_np]
    conf_rgb = _jet(conf_np)

    # ── masks/ ────────────────────────────────────────────────────────────────
    masks_dir = out_dir / 'masks'
    masks_dir.mkdir(exist_ok=True)
    Image.fromarray(rgb).save(masks_dir      / f'{idx:03d}_rgb.png')
    Image.fromarray(gt_rgb).save(masks_dir   / f'{idx:03d}_gt.png')
    Image.fromarray(pred_rgb).save(masks_dir / f'{idx:03d}_pred.png')
    Image.fromarray(conf_rgb).save(masks_dir / f'{idx:03d}_conf.png')

    # ── attn/  — one subfolder per scale, avg + up to 8 heads as overlays ────
    attn_dir = out_dir / 'attn'
    attn_dir.mkdir(exist_ok=True)
    for name, sal in attns.items():
        scale_dir = attn_dir / name
        scale_dir.mkdir(exist_ok=True)
        # avg overlay (most useful single view)
        avg   = _norm01(_upsample(sal.float().mean(0), H, W))
        Image.fromarray(_overlay(rgb, _jet(avg))).save(
            scale_dir / f'{idx:03d}_avg.png')
        # per-head overlays, capped at 8
        n_show = min(sal.shape[0], 8)
        for k in range(n_show):
            arr   = _norm01(_upsample(sal[k], H, W))
            Image.fromarray(_overlay(rgb, _jet(arr))).save(
                scale_dir / f'{idx:03d}_h{k:02d}.png')

    # ── composite/ — one tall image with everything ───────────────────────────
    comp_dir = out_dir / 'composite'
    comp_dir.mkdir(exist_ok=True)
    _save_composite(rgb, gt_rgb, pred_rgb, conf_rgb, attns, H, W, idx, comp_dir, font)


def _save_composite(rgb, gt_rgb, pred_rgb, conf_rgb,
                    attns, H, W, idx, comp_dir, font):
    """
    Layout:
      Row 0: Input | GT | Prediction | Confidence
      Row N: [scale name] avg-overlay | head0-overlay | head1-overlay | ...

    Each cell is CELL×CELL pixels.
    """
    CELL = 224

    def cell(arr: np.ndarray, label: str) -> np.ndarray:
        im = Image.fromarray(arr).resize((CELL, CELL), Image.BILINEAR)
        im = _label(im, label, font)
        return np.array(im)

    rows = []

    # Row 0: masks
    row0 = _hstack([
        cell(rgb,      'Input'),
        cell(gt_rgb,   'GT mask'),
        cell(pred_rgb, 'Prediction'),
        cell(conf_rgb, 'Confidence'),
    ])
    rows.append(row0)

    # One row per attention scale
    for name, sal in attns.items():
        n_heads = sal.shape[0]
        n_show  = min(n_heads, 8)   # show up to 8 heads per row

        # avg overlay first, then per-head
        avg    = _norm01(_upsample(sal.float().mean(0), H, W))
        cells  = [cell(_overlay(rgb, _jet(avg)),   f'{name} avg')]
        for k in range(n_show):
            arr   = _norm01(_upsample(sal[k], H, W))
            cells.append(cell(_overlay(rgb, _jet(arr)), f'{name}[{k}]'))

        rows.append(_hstack(cells))

    canvas = _vstack(rows, gap=4)
    Image.fromarray(canvas).save(comp_dir / f'{idx:03d}_composite.png')


# ── Epoch entry point ─────────────────────────────────────────────────────────

@torch.no_grad()
def save_epoch_outputs(
    model,
    val_dataset,
    device: torch.device,
    out_root: Path,
    epoch: int,
    num_classes: int,
    n_samples: int = 6,
    fixed_indices: Optional[list] = None,
    cat_names=None,
):
    """
    Run model on fixed val samples and save all visualizations.

    outputs/epoch_NNNN/
      masks/        rgb / gt / pred / conf
      attn/tiny/    avg + per-head overlays
      attn/small/   …
      attn/medium/  …
      attn/large/   …
      composite/    everything in one image per sample
    """
    model.eval()
    epoch_dir = Path(out_root) / f'epoch_{epoch:04d}'

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if fixed_indices is None:
        rng     = np.random.default_rng(seed=42)
        indices = rng.choice(len(val_dataset),
                             size=min(n_samples, len(val_dataset)),
                             replace=False).tolist()
    else:
        indices = list(fixed_indices)[:n_samples]

    for si, ds_idx in enumerate(indices):
        img_t, mask_t = val_dataset[ds_idx]
        logits, _, _  = model(img_t.unsqueeze(0).to(device))   # populates _attns
        logits        = logits[0].cpu()
        attns         = {k: v[0].cpu() for k, v in model._attns.items()}

        visualize_sample(
            img_t       = img_t,
            mask_t      = mask_t,
            logits      = logits,
            attns       = attns,
            out_dir     = epoch_dir,
            idx         = si,
            num_classes = num_classes,
            font        = font,
        )

    return epoch_dir
