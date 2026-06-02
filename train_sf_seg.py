#!/usr/bin/env python3
"""Training script for person segmentation using the sf_seg model."""
from __future__ import annotations

import argparse
import csv
import logging
import random
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
import numpy as np
from tqdm import tqdm

from sf_seg import sf_seg
from losses import iou_loss, combine_losses, mse_loss, diversity_loss
import torch.nn.functional as F



class SegmentationDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, image_size: int = 128):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = int(image_size)
        self.images = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        self.pairs = []
        for p in self.images:
            mask_p = self.masks_dir / (p.stem + '.png')
            if mask_p.exists():
                self.pairs.append((p, mask_p))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, mask_p = self.pairs[idx]
        img = Image.open(img_p).convert('RGB')
        mask = Image.open(mask_p).convert('L')
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        img_t = TF.to_tensor(img)
        mask_t = TF.to_tensor(mask)
        mask_t = (mask_t > 0.5).float()
        return img_t, mask_t


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        p = (pred > 0.5).float()
        correct = (p == target).float().mean().item()
    return correct


def save_validation_sample(model, dataset: Dataset, device: torch.device, out_dir: Path, epoch: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = random.randrange(len(dataset))
    img_t, mask_t = dataset[idx]
    model.eval()
    with torch.no_grad():
        pred, _, _ = model(img_t.unsqueeze(0).to(device))
        pred = pred.cpu().squeeze(0)
    pred_mask = (pred > 0.5).float()
    img_pil = TF.to_pil_image(img_t)
    gt_pil = TF.to_pil_image(mask_t.squeeze(0))
    pred_pil = TF.to_pil_image(pred_mask.squeeze(0))

    def mask_to_rgb(mask_pil: Image.Image, color=(255, 0, 0)) -> Image.Image:
        m = np.array(mask_pil)
        if m.max() <= 1:
            m = (m * 255).astype(np.uint8)
        mask_bool = m > 127
        rgb = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
        rgb[mask_bool] = color
        return Image.fromarray(rgb)

    gt_rgb = mask_to_rgb(gt_pil)
    pred_rgb = mask_to_rgb(pred_pil)
    w, h = img_pil.size
    gt_rgb = gt_rgb.resize((w, h), Image.NEAREST)
    pred_rgb = pred_rgb.resize((w, h), Image.NEAREST)

    combined = Image.new('RGB', (w * 3, h))
    combined.paste(img_pil, (0, 0))
    combined.paste(gt_rgb, (w, 0))
    combined.paste(pred_rgb, (w * 2, 0))

    try:
        draw = ImageDraw.Draw(combined)
        font = ImageFont.load_default()
        label_y = 4
        draw.text((w // 2 - 16, label_y), "RGB", fill=(255, 255, 255), font=font)
        draw.text((w + w // 2 - 12, label_y), "GT", fill=(255, 255, 255), font=font)
        draw.text((2 * w + w // 2 - 22, label_y), "PRED", fill=(255, 255, 255), font=font)
    except Exception:
        pass

    target_h = 480
    target_w = int(combined.width * (target_h / combined.height))
    combined = combined.resize((target_w, target_h), Image.BILINEAR)
    combined.save(out_dir / f"epoch_{epoch}_sample_{idx}_combined.png")


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    data_root = Path(args.data_root)
    train_images = data_root / 'images' / 'train'
    train_masks = data_root / 'masks' / 'train'
    val_images = data_root / 'images' / 'val'
    val_masks = data_root / 'masks' / 'val'

    train_ds = SegmentationDataset(train_images, train_masks, image_size=args.image_size)
    val_ds = SegmentationDataset(val_images, val_masks, image_size=args.image_size)

    if len(train_ds) == 0:
        raise RuntimeError(f"No training samples found in {train_images}")
    if len(val_ds) == 0:
        print("Warning: no validation samples found; using train split for validation metrics")
        val_ds = train_ds

    pin_memory = device.type == 'cuda'
    persistent = args.num_workers > 0
    prefetch_factor = 2 if args.num_workers > 0 else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory,
        prefetch_factor=prefetch_factor, persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
        prefetch_factor=prefetch_factor, persistent_workers=persistent,
    )

    model = sf_seg(num_channels=args.num_channels, focus_size=args.focus_size,
                   encoder_stride=args.encoder_stride)
    print(f"Model parameters: {model.get_num_parameters():,}")
    model.to(device)
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=use_amp)

    # sf_seg always outputs probabilities (sigmoid applied in forward), so only iou/bce with probs
    if args.loss_type in ("bce", "bce_iou"):
        criterion = nn.BCELoss()
    elif args.loss_type == "iou":
        criterion = None
    elif args.loss_type == "combine":
        criterion = None
    elif args.loss_type == "mse":
        criterion = None
    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler(log_dir / 'train_sf_seg.log'), logging.StreamHandler()],
    )
    csv_file = open(log_dir / 'train_sf_seg_log.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'epoch',
        'train_loss', 'train_seg', 'train_attn', 'train_div', 'train_acc',
        'val_loss',   'val_seg',   'val_attn',   'val_div',   'val_acc',
    ])

    out_samples = Path(args.output_dir)
    out_samples.mkdir(parents=True, exist_ok=True)
    checkpoints = Path(args.checkpoint_dir)
    checkpoints.mkdir(parents=True, exist_ok=True)

    for p in checkpoints.glob("sf_seg_epoch_*.pt"):
        try:
            p.unlink()
        except Exception:
            pass

    best_val_loss = float("inf")
    best_model_path = checkpoints / "sf_seg_best.pt"
    last_model_path = checkpoints / "sf_seg_last.pt"

    start_epoch = 1
    resume_arg = getattr(args, "resume", None)
    if resume_arg:
        if str(resume_arg).lower() in ("last", "auto"):
            resume_fp = checkpoints / "sf_seg_last.pt"
        else:
            resume_fp = Path(resume_arg)
        if resume_fp.exists():
            try:
                ckpt = torch.load(resume_fp, map_location=device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                    if "optimizer_state_dict" in ckpt:
                        try:
                            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                        except Exception:
                            logging.warning("Couldn't load optimizer state; optimizer reinitialized")
                    if "scheduler_state_dict" in ckpt:
                        try:
                            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                        except Exception:
                            pass
                    if "epoch" in ckpt:
                        start_epoch = int(ckpt.get("epoch", 0)) + 1
                    if "best_val_loss" in ckpt:
                        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
                    logging.info(f"Resumed from {resume_fp}, starting at epoch {start_epoch}")
                else:
                    model.load_state_dict(ckpt)
                    logging.info(f"Loaded model state dict from {resume_fp}")
            except Exception as e:
                logging.warning(f"Failed to load checkpoint '{resume_fp}': {e}")
        else:
            logging.warning(f"Resume checkpoint not found: {resume_fp}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = dict(loss=0., seg=0., attn=0., div=0., acc=0.)
        seen = 0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for images, masks in train_iter:
            images = images.to(device, non_blocking=pin_memory)
            masks = masks.to(device, non_blocking=pin_memory)

            with autocast('cuda', enabled=use_amp):
                preds, attn_guide, attn = model(images)
                if args.loss_type == "iou":
                    seg = iou_loss(preds, masks)
                elif args.loss_type == "bce":
                    seg = criterion(preds, masks)
                elif args.loss_type == "combine":
                    seg = combine_losses(preds, masks)
                elif args.loss_type == "mse":
                    seg = mse_loss(preds, masks)
                else:  # bce_iou
                    seg = criterion(preds, masks) + iou_loss(preds, masks)

                attn_l = torch.tensor(0., device=device)
                if args.attn_guide_weight > 0:
                    _, _, H_a, W_a = attn.shape
                    masks_a = F.interpolate(masks.float(), size=(H_a, W_a), mode='nearest')
                    fg = (attn * masks_a).sum(dim=[2, 3])
                    tot = attn.sum(dim=[2, 3]).clamp(min=1e-8)
                    attn_l = args.attn_guide_weight * (1.0 - (fg / tot).mean())

                div_l = torch.tensor(0., device=device)
                if args.diversity_weight > 0:
                    div_l = args.diversity_weight * diversity_loss(attn)

                loss = seg + attn_l + div_l

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            b = images.size(0)
            batch_acc = pixel_accuracy(preds, masks)
            running['loss'] += loss.item() * b
            running['seg']  += seg.item()  * b
            running['attn'] += attn_l.item() * b
            running['div']  += div_l.item()  * b
            running['acc']  += batch_acc * b
            seen += b
            train_iter.set_postfix(
                total=f"{loss.item():.4f}",
                seg=f"{seg.item():.4f}",
                attn=f"{attn_l.item():.4f}",
                div=f"{div_l.item():.4f}",
                acc=f"{batch_acc:.4f}",
            )

        train_loss = running['loss'] / seen
        train_seg  = running['seg']  / seen
        train_attn = running['attn'] / seen
        train_div  = running['div']  / seen
        train_acc  = running['acc']  / seen

        model.eval()
        vrun = dict(loss=0., seg=0., attn=0., div=0., acc=0.)
        v_seen = 0
        with torch.inference_mode():
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]  ", leave=False)
            for images, masks in val_iter:
                images = images.to(device, non_blocking=pin_memory)
                masks = masks.to(device, non_blocking=pin_memory)

                with autocast('cuda', enabled=use_amp):
                    preds, attn_guide, attn = model(images)
                    if args.loss_type == "iou":
                        seg = iou_loss(preds, masks)
                    elif args.loss_type == "bce":
                        seg = criterion(preds, masks)
                    elif args.loss_type == "combine":
                        seg = combine_losses(preds, masks)
                    elif args.loss_type == "mse":
                        seg = mse_loss(preds, masks)
                    else:  # bce_iou
                        seg = criterion(preds, masks) + iou_loss(preds, masks)

                    attn_l = torch.tensor(0., device=device)
                    if args.attn_guide_weight > 0:
                        _, _, H_a, W_a = attn.shape
                        masks_a = F.interpolate(masks.float(), size=(H_a, W_a), mode='nearest')
                        fg = (attn * masks_a).sum(dim=[2, 3])
                        tot = attn.sum(dim=[2, 3]).clamp(min=1e-8)
                        attn_l = args.attn_guide_weight * (1.0 - (fg / tot).mean())

                    # diversity_loss is a train-only regulariser — skip on val
                    loss = seg + attn_l

                b = images.size(0)
                batch_acc = pixel_accuracy(preds, masks)
                vrun['loss'] += loss.item() * b
                vrun['seg']  += seg.item()  * b
                vrun['attn'] += attn_l.item() * b
                vrun['acc']  += batch_acc * b
                v_seen += b
                val_iter.set_postfix(
                    total=f"{loss.item():.4f}",
                    seg=f"{seg.item():.4f}",
                    attn=f"{attn_l.item():.4f}",
                    acc=f"{batch_acc:.4f}",
                )

        val_loss = vrun['loss'] / v_seen if v_seen else 0.
        val_seg  = vrun['seg']  / v_seen if v_seen else 0.
        val_attn = vrun['attn'] / v_seen if v_seen else 0.
        val_div  = vrun['div']  / v_seen if v_seen else 0.
        val_acc  = vrun['acc']  / v_seen if v_seen else 0.

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        logging.info(
            f"Epoch {epoch}/{args.epochs} | lr={current_lr:.2e} | "
            f"train  total={train_loss:.4f}  seg={train_seg:.4f}  attn={train_attn:.4f}  div={train_div:.4f}  acc={train_acc:.4f} | "
            f"val    total={val_loss:.4f}  seg={val_seg:.4f}  attn={val_attn:.4f}  div={val_div:.4f}  acc={val_acc:.4f}"
        )
        csv_writer.writerow([
            epoch,
            train_loss, train_seg, train_attn, train_div, train_acc,
            val_loss,   val_seg,   val_attn,   val_div,   val_acc,
        ])
        csv_file.flush()

        try:
            improved = v_seen and val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "num_channels": args.num_channels,
                "focus_size": args.focus_size,
                "encoder_stride": args.encoder_stride,
            }
            torch.save(ckpt, last_model_path)
            if improved:
                torch.save(ckpt, best_model_path)
                logging.info(f"New best model saved: {best_model_path} (val_loss={val_loss:.6f})")
        except Exception as e:
            logging.warning(f"Failed to save checkpoint: {e}")

        try:
            save_validation_sample(model, val_ds, device, out_samples, epoch)
        except Exception as e:
            logging.warning(f"Failed to save validation sample: {e}")

    csv_file.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="path to json config file")
    p.add_argument("--data-root", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--num-channels", type=int, default=None, help="number of channels in sf_seg attention block")
    p.add_argument("--kernel-size", type=int, default=None, help="conv kernel size in attention block (default 7)")
    p.add_argument("--focus-size", type=int, default=None, help="attention budget = focus_size^2 pixels per channel (default 16)")
    p.add_argument("--encoder-stride", type=int, default=None, choices=[1, 2],
                   help="stride of first encoder conv (1=full res, 2=half res ~3x faster)")
    p.add_argument("--attn-guide-weight", type=float, default=None, help="weight for attention guidance loss (default 0.3, 0 = disabled)")
    p.add_argument("--diversity-weight", type=float, default=None, help="weight for attention diversity loss (default 0.1, 0 = disabled)")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--log-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--loss-type", default=None, choices=["iou", "bce", "bce_iou", "combine", "mse"])
    p.add_argument("--resume", default=None, help="checkpoint path or 'last'")
    p.add_argument("--image-size", type=int, default=None)
    return p.parse_args()


def merge_config(args):
    cfg = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        cfg = json.load(open(cfg_path))
    else:
        default_cfg = Path("config.json")
        if default_cfg.exists():
            cfg = json.load(open(default_cfg))

    defaults = {
        "data_root": "data",
        "epochs": 100,
        "batch_size": 64,
        "lr": 1e-4,
        "num_workers": 0,
        "num_channels": 128,
        "kernel_size": 7,
        "focus_size": 16,
        "encoder_stride": 1,
        "attn_guide_weight": 0.3,
        "diversity_weight": 0.1,
        "log_dir": "logs",
        "output_dir": "outputs",
        "checkpoint_dir": "checkpoints",
        "loss_type": "iou",
        "resume": None,
        "image_size": 128,
    }

    for key, default_val in defaults.items():
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            final = cli_val
        else:
            final = cfg.get(key, default_val)
        setattr(args, key, final)

    return args


if __name__ == "__main__":
    args = parse_args()
    args = merge_config(args)
    print("Resolved hyperparameters:")
    for k in ["data_root", "epochs", "batch_size", "lr", "num_workers", "num_channels", "kernel_size",
              "focus_size", "log_dir", "output_dir", "checkpoint_dir", "loss_type", "resume", "image_size"]:
        print(f"  {k}:", getattr(args, k))
    train(args)
