#!/usr/bin/env python3
"""
Visualize sf_seg attention maps và top-down guidance cascade.

Layout (2 hàng × 5 cột):
  Row 0:  Input │ head_tiny (global) │ guide→small │ head_small │ guide→medium
  Row 1:  Pred  │ head_medium        │ guide→large │ head_large  │ guidance weights

Mỗi attention map = amax over channels (max activation tại mỗi spatial position),
upsampled về 512×512 để so sánh với ảnh gốc.

Usage:
  # không cần data — dùng random noise
  python scripts/attention.py

  # dùng ảnh thật + checkpoint
  python scripts/attention.py --image data/images/val/ADE_val_00000001.jpg \\
                               --checkpoint checkpoints/sf_seg_last.pt

  # nhiều ảnh từ val set
  python scripts/attention.py --checkpoint checkpoints/sf_seg_last.pt \\
                               --data-root data --num-images 6
"""
import argparse
import colorsys
import json
import random
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Palette ───────────────────────────────────────────────────────────────────

def _make_palette(n: int) -> np.ndarray:
    pal = np.zeros((n, 3), dtype=np.uint8)
    for i in range(1, n):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75 + 0.1 * (i % 3), 0.85 + 0.05 * (i % 2))
        pal[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return pal


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str | None, backbone: str, num_channels: int,
               focus_size: int, num_classes: int, device: torch.device):
    if backbone == 'resnet18':
        from src.models.sf_seg_r18 import sf_seg
    else:
        from src.models.sf_seg import sf_seg

    model = sf_seg(num_channels=num_channels, focus_size=focus_size,
                   num_classes=num_classes)

    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd   = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        # honour checkpoint metadata if present
        if isinstance(ckpt, dict):
            num_channels = ckpt.get('num_channels', num_channels)
            focus_size   = ckpt.get('focus_size',   focus_size)
            num_classes  = ckpt.get('num_classes',  num_classes)
            model = sf_seg(num_channels=num_channels, focus_size=focus_size,
                           num_classes=num_classes)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[warn] missing keys: {missing[:3]}{'…' if len(missing)>3 else ''}")
        print(f"Loaded {ckpt_path}  (C={num_channels}, fs={focus_size}, cls={num_classes})")
    else:
        print(f"No checkpoint — using random-init model  (C={num_channels}, fs={focus_size})")

    return model.to(device).eval()


# ── Forward with hook capture ─────────────────────────────────────────────────

@torch.no_grad()
def run_model(model, img_t: torch.Tensor, device: torch.device):
    """
    Returns:
        logits : (num_classes, H, W) cpu
        attns  : dict[str → (C, H', W')] — raw attention maps per head (cpu)
        guides : dict[str → (1, H', W')] — guidance signals passed between heads (cpu)
        gw     : dict[str → float]       — guidance_weight scalars
    """
    attns = {}

    # forward hooks capture each head's output[1] = attn map
    hooks = []
    for name in ('tiny', 'small', 'medium', 'large'):
        def _hook(module, inp, out, _n=name):
            attns[_n] = out[1].squeeze(0).detach().cpu()   # (C, H, W)
        hooks.append(getattr(model, f'head_{name}').register_forward_hook(_hook))

    logits, _, _ = model(img_t.unsqueeze(0).to(device))

    for h in hooks:
        h.remove()

    # Reconstruct guidance signals from captured attention maps
    # (these are what gets passed between heads in extract_features)
    guides = {}
    size_s = attns['small'].shape[1:]   # (H/16, W/16)
    size_m = attns['medium'].shape[1:]  # (H/8,  W/8)
    size_l = attns['large'].shape[1:]   # (H/4,  W/4)

    guides['tiny→small']  = F.interpolate(
        attns['tiny'].amax(0, keepdim=True).unsqueeze(0),
        size_s, mode='bilinear', align_corners=False).squeeze(0)   # (1, H/16, W/16)
    guides['small→medium'] = F.interpolate(
        attns['small'].amax(0, keepdim=True).unsqueeze(0),
        size_m, mode='bilinear', align_corners=False).squeeze(0)
    guides['medium→large'] = F.interpolate(
        attns['medium'].amax(0, keepdim=True).unsqueeze(0),
        size_l, mode='bilinear', align_corners=False).squeeze(0)

    # guidance_weight scalars (learned)
    gw = {}
    for name in ('small', 'medium', 'large'):
        w = getattr(model, f'head_{name}').guidance_weight
        gw[name] = w.item() if w is not None else 0.0

    return logits.squeeze(0).cpu(), attns, guides, gw


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(t: torch.Tensor) -> np.ndarray:
    """Normalise tensor to [0,1] numpy array."""
    t = t.float()
    mn, mx = t.min(), t.max()
    return ((t - mn) / (mx - mn + 1e-8)).numpy()


def _attn_summary(attn: torch.Tensor, target_hw) -> np.ndarray:
    """Max over channels, upsample to target_hw, return [0,1] numpy."""
    summary = attn.amax(0, keepdim=True).unsqueeze(0)          # (1, 1, H', W')
    up = F.interpolate(summary, target_hw, mode='bilinear', align_corners=False)
    return _norm(up.squeeze())


def _overlay(img_np: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend heatmap (viridis) over RGB image."""
    cmap  = plt.cm.get_cmap('inferno')
    h_rgb = (cmap(heat)[:, :, :3] * 255).astype(np.uint8)
    return ((1 - alpha) * img_np + alpha * h_rgb).clip(0, 255).astype(np.uint8)


# ── Single-image figure ───────────────────────────────────────────────────────

def visualise_one(img_np: np.ndarray, mask_np: np.ndarray | None,
                  logits: torch.Tensor, attns: dict, guides: dict, gw: dict,
                  palette: np.ndarray, cat_names: dict | None,
                  out_path: Path, title: str = ""):
    H, W = img_np.shape[:2]
    hw   = (H, W)

    # ── compute summaries ─────────────────────────────────────────────────────
    # Each head: max over channels, upsampled to full res
    tiny_sum   = _attn_summary(attns['tiny'],   hw)
    small_sum  = _attn_summary(attns['small'],  hw)
    med_sum    = _attn_summary(attns['medium'], hw)
    large_sum  = _attn_summary(attns['large'],  hw)

    # Guidance signals upsampled to full res
    guide_ts = _norm(F.interpolate(guides['tiny→small'].unsqueeze(0),
                                    hw, mode='bilinear', align_corners=False).squeeze())
    guide_sm = _norm(F.interpolate(guides['small→medium'].unsqueeze(0),
                                    hw, mode='bilinear', align_corners=False).squeeze())
    guide_ml = _norm(F.interpolate(guides['medium→large'].unsqueeze(0),
                                    hw, mode='bilinear', align_corners=False).squeeze())

    # Prediction
    pred_cls = logits.argmax(0).numpy().astype(np.int32)
    pred_rgb = palette[pred_cls]

    # GT
    gt_rgb = palette[mask_np] if mask_np is not None else np.zeros((H, W, 3), np.uint8)

    # ── layout 3 rows × 5 cols ────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 13))
    fig.suptitle(title, fontsize=11, y=0.99)
    gs  = gridspec.GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.05)

    def show(ax, data, ttl, cmap='viridis', overlay=None):
        if overlay is not None:
            ax.imshow(overlay)
            ax.imshow(data, cmap=cmap, alpha=0.6, vmin=0, vmax=1)
        else:
            ax.imshow(data, cmap=cmap if data.ndim == 2 else None,
                      vmin=(0 if data.ndim == 2 else None),
                      vmax=(1 if data.ndim == 2 else None))
        ax.set_title(ttl, fontsize=9, pad=4)
        ax.axis('off')

    # Row 0: Input  |  Tiny (global overview)  |  guide tiny→small  |  Small  |  guide small→med
    show(fig.add_subplot(gs[0, 0]), img_np,    "Input")
    show(fig.add_subplot(gs[0, 1]), tiny_sum,  f"head_tiny\n(global, 16×16→{H}×{W})",
         cmap='inferno', overlay=img_np)
    show(fig.add_subplot(gs[0, 2]), guide_ts,
         f"guidance tiny→small\n(w={gw.get('small', 0):.4f})", cmap='plasma', overlay=img_np)
    show(fig.add_subplot(gs[0, 3]), small_sum, f"head_small\n(32×32→{H}×{W})",
         cmap='inferno', overlay=img_np)
    show(fig.add_subplot(gs[0, 4]), guide_sm,
         f"guidance small→medium\n(w={gw.get('medium', 0):.4f})", cmap='plasma', overlay=img_np)

    # Row 1: guide small→med  |  Medium  |  guide med→large  |  Large  |  GT
    show(fig.add_subplot(gs[1, 0]), med_sum,   f"head_medium\n(64×64→{H}×{W})",
         cmap='inferno', overlay=img_np)
    show(fig.add_subplot(gs[1, 1]), guide_ml,
         f"guidance medium→large\n(w={gw.get('large', 0):.4f})", cmap='plasma', overlay=img_np)
    show(fig.add_subplot(gs[1, 2]), large_sum, f"head_large\n(128×128→{H}×{W})",
         cmap='inferno', overlay=img_np)
    show(fig.add_subplot(gs[1, 3]), gt_rgb,    "Ground Truth")
    show(fig.add_subplot(gs[1, 4]), pred_rgb,  "Prediction")

    # Row 2: channel-level detail for head_large (most informative)
    large_attn = attns['large']   # (C, H/4, W/4)
    C_show = min(5, large_attn.shape[0])
    # Pick channels with highest spatial variance (most "active")
    var = large_attn.var(dim=[1, 2])
    top_ch = var.argsort(descending=True)[:C_show]
    for i, ch in enumerate(top_ch):
        a = _norm(F.interpolate(large_attn[ch].unsqueeze(0).unsqueeze(0),
                                 hw, mode='bilinear', align_corners=False).squeeze())
        show(fig.add_subplot(gs[2, i]), a,
             f"head_large ch {ch.item()}\n(var={var[ch]:.4f})", cmap='hot', overlay=img_np)

    # ── guidance weight bar (inside last row if fewer channels) ───────────────
    # Add text annotation on figure
    weight_txt = "\n".join([
        "guidance_weight (init=0, learned):  ",
        f"  head_small  : {gw.get('small',  0):+.6f}",
        f"  head_medium : {gw.get('medium', 0):+.6f}",
        f"  head_large  : {gw.get('large',  0):+.6f}",
        "",
        "  > 0 → follow upper head's focus",
        "  < 0 → attend complementary regions",
    ])
    fig.text(0.01, 0.02, weight_txt, fontsize=8,
             family='monospace', color='white',
             bbox=dict(facecolor='#222', alpha=0.85, pad=6, boxstyle='round'))

    # Predicted classes legend
    if cat_names:
        top_cls = [c for c in sorted(set(pred_cls.flat)) if c > 0][:6]
        names   = [cat_names.get(str(c), f'cls{c}') for c in top_cls]
        fig.text(0.5, 0.02, "pred: " + " · ".join(names), fontsize=8,
                 ha='center', color='white',
                 bbox=dict(facecolor='#222', alpha=0.7, pad=4, boxstyle='round'))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches='tight', facecolor='#111')
    plt.close(fig)
    print(f"  → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    import torchvision.transforms.functional as TF
    from PIL import Image

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = load_model(args.checkpoint, args.backbone, args.num_channels,
                         args.focus_size, args.num_classes, device)
    palette = _make_palette(args.num_classes)

    cat_names = None
    cat_json  = Path(args.data_root) / 'cat_to_idx.json'
    if cat_json.exists():
        cat_names = json.load(open(cat_json)).get('idx_to_name', {})

    out_dir = Path(args.output_dir)
    sz      = args.image_size

    # ── collect image/mask pairs ──────────────────────────────────────────────
    pairs = []

    if args.image:
        img_p = Path(args.image)
        mask_p = Path(args.mask) if args.mask else None
        pairs.append((img_p, mask_p))

    else:
        img_dir  = Path(args.data_root) / 'images' / 'val'
        mask_dir = Path(args.data_root) / 'masks'  / 'val'
        if img_dir.exists():
            found = [
                (p, mask_dir / (p.stem + '.png'))
                for p in sorted(img_dir.iterdir())
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
                and (mask_dir / (p.stem + '.png')).exists()
            ]
            random.seed(args.seed)
            pairs = random.sample(found, min(args.num_images, len(found)))

    # ── synthetic fallback ────────────────────────────────────────────────────
    if not pairs:
        print("No data found — generating synthetic checkerboard input.")
        img_t = torch.zeros(3, sz, sz)
        # checkerboard pattern so attention has something to "see"
        for c in range(3):
            for i in range(0, sz, 64):
                for j in range(0, sz, 64):
                    if (i // 64 + j // 64) % 2 == 0:
                        img_t[c, i:i+64, j:j+64] = 0.8
            img_t[c] += torch.randn(sz, sz) * 0.05
        img_t = img_t.clamp(0, 1)
        img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        logits, attns, guides, gw = run_model(model, img_t, device)
        visualise_one(img_np, None, logits, attns, guides, gw,
                      palette, cat_names,
                      out_dir / 'attention_synthetic.png',
                      title='Synthetic input (no checkpoint / no data)')
        return

    # ── real images ───────────────────────────────────────────────────────────
    for img_path, mask_path in pairs:
        img_pil = Image.open(img_path).convert('RGB').resize((sz, sz), Image.BILINEAR)
        img_t   = TF.to_tensor(img_pil)
        img_np  = np.array(img_pil)

        mask_np = None
        if mask_path and mask_path.exists():
            mask_np = np.array(
                Image.open(mask_path).convert('L').resize((sz, sz), Image.NEAREST),
                dtype=np.int32)

        logits, attns, guides, gw = run_model(model, img_t, device)
        visualise_one(img_np, mask_np, logits, attns, guides, gw,
                      palette, cat_names,
                      out_dir / f'attention_{img_path.stem}.png',
                      title=img_path.name)

    print(f"\nDone. Saved to {out_dir}/")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',    default=None,
                   help='Path to sf_seg_last.pt / sf_seg_best.pt')
    p.add_argument('--backbone',      default='custom', choices=['custom', 'resnet18'])
    p.add_argument('--num-channels',  type=int, default=25)
    p.add_argument('--focus-size',    type=int, default=64)
    p.add_argument('--num-classes',   type=int, default=151)
    p.add_argument('--image-size',    type=int, default=512)
    p.add_argument('--image',         default=None,  help='Single image path')
    p.add_argument('--mask',          default=None,  help='Corresponding mask path')
    p.add_argument('--data-root',     default='data')
    p.add_argument('--num-images',    type=int, default=4)
    p.add_argument('--output-dir',    default='outputs/attention_vis')
    p.add_argument('--seed',          type=int, default=42)
    main(p.parse_args())
