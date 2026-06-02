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

from sf_seg import sf_seg, AttentionBlock


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        num_channels = ckpt.get("num_channels", 64)
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
        num_channels = 64

    # Đọc num_channels từ weight shape thực tế
    for k, v in state.items():
        if "encoder.0.weight" in k:  # Conv(3 → C), shape (C, 3, 3, 3)
            num_channels = v.shape[0]
            break

    # focus_k từ shape của encoder output (2C)
    for k, v in state.items():
        if "encoder.4.weight" in k:  # Conv(C → 2C), shape (2C, C, 3, 3)
            num_channels = v.shape[0] // 2
            break

    model = sf_seg(num_channels=num_channels)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded model: num_channels={num_channels}, params={model.get_num_parameters():,}")
    return model


@torch.no_grad()
def get_attention_maps(model: sf_seg, img_t: torch.Tensor, device: torch.device):
    """Trả về attention maps (N, H, W) và predicted mask (H, W)."""
    x = img_t.unsqueeze(0).to(device)             # (1, 3, H, W)

    # Chạy thủ công để lấy intermediate tensors
    ab = model.attention_block
    out = ab.encoder(x)                            # (1, 2C, H, W)
    score, _ = out.chunk(2, dim=1)                 # (1, C, H, W)
    B, N, H, W = score.shape
    attn = AttentionBlock._clamped_softmax(
        score.view(B, N, H * W), float(ab.focus_k)
    ).view(B, N, H, W)                             # (1, C, H, W), mỗi pixel ∈ [0,1]

    pred, _ = model(x)
    pred = pred.squeeze()                          # (H, W)

    return attn.squeeze(0).cpu(), pred.cpu()       # (C, H, W), (H, W)


def load_sample(img_path: Path, mask_path: Path, image_size: int):
    img  = Image.open(img_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    mask = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.NEAREST)
    img_t  = TF.to_tensor(img)
    mask_t = (TF.to_tensor(mask) > 0.5).float().squeeze(0)
    return img_t, mask_t, img


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

    for img_path, mask_path in samples:
        img_t, mask_t, img_pil = load_sample(img_path, mask_path, args.image_size)
        attn_maps, pred = get_attention_maps(model, img_t, device)

        C = attn_maps.shape[0]
        show_n = min(args.show_channels, C)

        # Chọn các channel có variance cao nhất (tập trung rõ nhất)
        variances   = attn_maps.var(dim=[1, 2])
        top_indices = variances.topk(show_n).indices.tolist()

        ncols = 4 + show_n  # img | gt | pred | attn_sum | channel maps
        fig = plt.figure(figsize=(ncols * 2.5, 3.5))
        gs  = gridspec.GridSpec(1, ncols, figure=fig, wspace=0.05)

        def show(ax, data, title, cmap="gray", vmin=None, vmax=None):
            ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=8)
            ax.axis("off")

        show(fig.add_subplot(gs[0]), img_pil,              "Image")
        show(fig.add_subplot(gs[1]), mask_t.numpy(),        "GT mask")
        show(fig.add_subplot(gs[2]), pred.numpy(),          "Pred mask",  vmin=0, vmax=1)
        show(fig.add_subplot(gs[3]), attn_maps.sum(0).numpy(), "Attn sum", cmap="hot")

        for col, ch_idx in enumerate(top_indices):
            amap = attn_maps[ch_idx].numpy()
            ax   = fig.add_subplot(gs[4 + col])
            im   = ax.imshow(amap, cmap="hot", vmin=0, vmax=1)
            ax.set_title(f"Ch {ch_idx}\nvar={variances[ch_idx]:.3f}", fontsize=7)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(img_path.name, fontsize=9)
        out_path = Path(args.output_dir) / f"attn_{img_path.stem}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    default="checkpoints/sf_seg_last.pt")
    p.add_argument("--data-root",     default="data")
    p.add_argument("--image-size",    type=int, default=128)
    p.add_argument("--num-images",    type=int, default=4)
    p.add_argument("--show-channels", type=int, default=8,
                   help="Số channel attention hiển thị (chọn top variance)")
    p.add_argument("--output-dir",    default="outputs/attention_vis")
    p.add_argument("--seed",          type=int, default=42)
    args = p.parse_args()
    visualize(args)
