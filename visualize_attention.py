#!/usr/bin/env python3
"""Visualize attention masks của sf_seg từ checkpoint."""
import random
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision.transforms.functional as TF

from sf_seg import sf_seg


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    num_channels   = ckpt.get("num_channels", 64)   if isinstance(ckpt, dict) else 64
    focus_size     = ckpt.get("focus_size", 16)      if isinstance(ckpt, dict) else 16
    encoder_stride = ckpt.get("encoder_stride", 1)   if isinstance(ckpt, dict) else 1

    # Fallback: đọc num_channels từ weight shape nếu không có trong checkpoint
    w = state.get("attention_block.encoder.4.weight")
    if w is not None:
        num_channels = w.shape[0] // 2

    model = sf_seg(num_channels=num_channels, focus_size=focus_size, encoder_stride=encoder_stride)
    model.load_state_dict(state)
    model.to(device).eval()
    feat_res = f"{128 // encoder_stride}×{128 // encoder_stride}" if encoder_stride > 1 else "128×128"
    print(f"Loaded: num_channels={num_channels} focus_size={focus_size} "
          f"encoder_stride={encoder_stride} feat_res={feat_res} "
          f"params={model.get_num_parameters():,}")
    return model


@torch.no_grad()
def get_attention_maps(model: sf_seg, img_t: torch.Tensor, device: torch.device):
    """Trả về attention maps (total_C, H, W) và predicted mask (H, W)."""
    x = img_t.unsqueeze(0).to(device)
    pred, _, attn = model(x)          # attn: (1, total_C, H, W) — tất cả blocks
    return attn.squeeze(0).cpu(), pred.squeeze().cpu()


def load_sample(img_path: Path, mask_path: Path, image_size: int):
    img  = Image.open(img_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    mask = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.NEAREST)
    img_t  = TF.to_tensor(img)
    mask_t = (TF.to_tensor(mask) > 0.5).float().squeeze(0)
    return img_t, mask_t, img


def select_channels(attn_maps: torch.Tensor, min_range: float, max_show: int):
    """
    Chọn channels có min-max range >= min_range, sort theo range giảm dần.

    Range = max - min trên spatial dims:
    - Range thấp: pattern mà channel này học không xuất hiện trong ảnh →
      attention phân tán đều, tất cả pixels ≈ k/L ≈ 0.016, không có gì để show
    - Range cao: channel tìm thấy pattern của nó trong ảnh →
      có điểm sáng gần 1.0, phần còn lại gần 0

    Trả về: (indices, ranges, maxvals) của các channel được chọn.
    """
    ch_max   = attn_maps.amax(dim=[1, 2])   # (C,)
    ch_min   = attn_maps.amin(dim=[1, 2])   # (C,)
    ch_range = ch_max - ch_min              # (C,)

    active = (ch_range >= min_range).nonzero(as_tuple=True)[0]
    if len(active) == 0:
        return [], [], []

    order   = ch_range[active].argsort(descending=True)
    picked  = active[order[:max_show]]
    return picked.tolist(), ch_range[picked].tolist(), ch_max[picked].tolist()


def visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device)

    val_img_dir  = Path(args.data_root) / "images" / "val"
    val_mask_dir = Path(args.data_root) / "masks"  / "val"

    pairs = sorted([
        (p, val_mask_dir / (p.stem + ".png"))
        for p in val_img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        and (val_mask_dir / (p.stem + ".png")).exists()
    ])

    random.seed(args.seed)
    samples = random.sample(pairs, min(args.num_images, len(pairs)))

    skipped = 0
    for img_path, mask_path in samples:
        img_t, mask_t, img_pil = load_sample(img_path, mask_path, args.image_size)
        attn_maps, pred = get_attention_maps(model, img_t, device)

        C = attn_maps.shape[0]
        indices, ranges, maxvals = select_channels(attn_maps, args.min_range, args.show_channels)

        if not indices:
            print(f"  Skip {img_path.name}: không có channel nào tìm thấy pattern "
                  f"trong ảnh này (range < {args.min_range} với mọi channel)")
            skipped += 1
            continue

        print(f"  {img_path.name}: {len(indices)}/{C} channels có pattern xuất hiện "
              f"trong ảnh (range >= {args.min_range}), showing top {len(indices)}")

        # Layout: 4 cột cố định + các channel active
        ncols = 4 + len(indices)
        fig = plt.figure(figsize=(ncols * 2.5, 3.5))
        gs  = gridspec.GridSpec(1, ncols, figure=fig, wspace=0.05)

        def show(ax, data, title, cmap="gray", vmin=None, vmax=None):
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        show(fig.add_subplot(gs[0]), img_pil,                    "Image")
        show(fig.add_subplot(gs[1]), mask_t.numpy(),             "GT mask")
        show(fig.add_subplot(gs[2]), pred.numpy(),               "Pred mask", vmin=0, vmax=1)
        show(fig.add_subplot(gs[3]), attn_maps.sum(0).numpy(),   "Attn sum",  cmap="hot")

        for col, (ch_idx, rng, mx) in enumerate(zip(indices, ranges, maxvals)):
            amap = attn_maps[ch_idx].numpy()
            ax   = fig.add_subplot(gs[4 + col])
            im   = ax.imshow(amap, cmap="hot", vmin=0, vmax=1)
            ax.set_title(f"Ch {ch_idx}\nrng={rng:.3f} max={mx:.2f}", fontsize=7)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(img_path.name, fontsize=9)
        out_path = Path(args.output_dir) / f"attn_{img_path.stem}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    if skipped:
        print(f"\n{skipped}/{len(samples)} ảnh bị skip vì không có channel nào "
              f"tìm thấy pattern phù hợp. Giảm --min-range (hiện tại={args.min_range}) "
              f"để hiển thị thêm.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="checkpoints/sf_seg_last.pt")
    p.add_argument("--data-root",     default="data")
    p.add_argument("--image-size",    type=int,   default=224)
    p.add_argument("--num-images",    type=int,   default=4)
    p.add_argument("--show-channels", type=int,   default=8,
                   help="Số channel hiển thị tối đa (chọn top range)")
    p.add_argument("--min-range",     type=float, default=0.05,
                   help="Ngưỡng min-max range: chỉ hiển thị channel tìm thấy pattern trong ảnh "
                        "(0=tất cả, khuyến nghị 0.05-0.2)")
    p.add_argument("--output-dir",    default="outputs/attention_vis")
    p.add_argument("--seed",          type=int,   default=42)
    args = p.parse_args()
    visualize(args)
