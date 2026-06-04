#!/usr/bin/env python3
"""Training script for sf_seg — multi-class segmentation on COCO 80 classes.

Binary segmentation (num_classes=1) is also supported for backward compatibility.
Set num_classes=81 in config.json for full COCO training.
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.transforms.functional as TF

from losses import (iou_loss, combine_losses, mse_loss, diversity_loss,
                    multiclass_iou_loss, ce_iou_loss)
from sf_seg import sf_seg


# ── Dataset ───────────────────────────────────────────────────────────────────

class SegmentationDataset(Dataset):
    """
    Loads image/mask pairs.

    Multi-class (num_classes>1): mask is a (H,W) long tensor with values 0..C-1.
    Binary       (num_classes=1): mask is a (1,H,W) float tensor with values {0,1}.
    """

    def __init__(self, images_dir: Path, masks_dir: Path,
                 image_size: int = 224, augment: bool = False, num_classes: int = 1):
        self.image_size  = int(image_size)
        self.augment     = augment
        self.num_classes = num_classes
        images_dir, masks_dir = Path(images_dir), Path(masks_dir)
        self.pairs = [
            (p, masks_dir / (p.stem + '.png'))
            for p in sorted(images_dir.iterdir())
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
            and (masks_dir / (p.stem + '.png')).exists()
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, mask_p = self.pairs[idx]
        img  = Image.open(img_p).convert('RGB')
        mask = Image.open(mask_p).convert('L')
        sz   = self.image_size
        img  = img.resize((sz, sz), Image.BILINEAR)
        mask = mask.resize((sz, sz), Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                angle = random.uniform(-15.0, 15.0)
                img  = TF.rotate(img,  angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST,  fill=0)
            if random.random() > 0.5:
                d = int(sz * 0.1)
                tx, ty = random.randint(-d, d), random.randint(-d, d)
                img  = TF.affine(img,  0, [tx, ty], 1.0, 0, TF.InterpolationMode.BILINEAR, fill=0)
                mask = TF.affine(mask, 0, [tx, ty], 1.0, 0, TF.InterpolationMode.NEAREST,  fill=0)
            if random.random() > 0.5:
                img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
            if random.random() > 0.5:
                img = TF.adjust_contrast(img,   random.uniform(0.7, 1.3))
            if random.random() > 0.5:
                img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))

        img_t  = TF.to_tensor(img)
        if self.num_classes > 1:
            mask_t = torch.from_numpy(np.array(mask)).long()   # (H, W)
        else:
            mask_t = (TF.to_tensor(mask) > 0.5).float()       # (1, H, W)
        return img_t, mask_t


# ── Metrics ───────────────────────────────────────────────────────────────────

def update_confusion_matrix(conf: torch.Tensor, pred: torch.Tensor,
                             target: torch.Tensor) -> None:
    """Accumulate (C,C) confusion matrix [true×pred] in-place, fully on GPU."""
    C    = conf.shape[0]
    p    = pred.view(-1).long().clamp(0, C - 1)
    t    = target.view(-1).long().clamp(0, C - 1)
    conf += torch.bincount(t * C + p, minlength=C * C).view(C, C)


def miou_from_confusion(conf: torch.Tensor) -> tuple:
    """Return (mean_iou, per_class_iou). Classes absent from GT are excluded."""
    inter     = conf.diagonal().float()
    union     = (conf.sum(1) + conf.sum(0) - inter).float()
    per_class = inter / union.clamp(min=1e-8)
    valid     = conf.sum(1) > 0
    return (per_class[valid].mean().item() if valid.any() else 0.0), per_class


# ── Class-frequency weights ───────────────────────────────────────────────────

def compute_class_weights(masks_dir: Path, num_classes: int,
                           cache_path: Path | None = None,
                           sample_size: int = 4000) -> torch.Tensor:
    """Median-frequency balancing (SegNet/DeepLab standard).

    w_c = median_freq / freq_c, clipped to [0.05, 20].
    Result is cached to avoid recomputation on restarts.
    """
    if cache_path and cache_path.exists():
        counts = np.array(json.load(open(cache_path)))
        logging.info(f"Loaded class frequencies from {cache_path}")
    else:
        files  = sorted(masks_dir.glob('*.png'))
        sample = random.sample(files, min(sample_size, len(files)))
        counts = np.zeros(num_classes, dtype=np.float64)
        for f in tqdm(sample, desc='Class freq scan', leave=False):
            vals, cnts = np.unique(np.array(Image.open(f), dtype=np.int32),
                                   return_counts=True)
            for v, c in zip(vals, cnts):
                if 0 <= v < num_classes:
                    counts[v] += c
        if cache_path:
            json.dump(counts.tolist(), open(cache_path, 'w'))
            logging.info(f"Saved class frequencies → {cache_path}")

    present     = counts > 0
    freq        = counts / counts.sum()
    median_freq = float(np.median(freq[present]))
    weights     = np.zeros(num_classes, dtype=np.float32)
    weights[present] = np.clip(median_freq / freq[present], 0.05, 20.0)
    logging.info(f"Class weights — bg={weights[0]:.2f}  person={weights[1]:.2f}  "
                 f"min={weights[present].min():.2f}  max={weights[present].max():.2f}")
    return torch.from_numpy(weights).float()


# ── Loss dispatcher ───────────────────────────────────────────────────────────

def seg_loss(logits, masks, loss_type, criterion, num_classes,
             no_obj_weight=0.1, class_weights=None):
    if num_classes > 1:
        if loss_type == "ce_iou":
            return ce_iou_loss(logits, masks,
                               class_weights=class_weights,
                               no_obj_weight=no_obj_weight)
        if loss_type == "iou":
            return multiclass_iou_loss(logits, masks, no_obj_weight=no_obj_weight)
        return F.cross_entropy(logits, masks, weight=class_weights)   # "ce"
    probs = torch.sigmoid(logits)
    if loss_type == "iou":     return iou_loss(probs, masks)
    if loss_type == "bce":     return criterion(probs, masks)
    if loss_type == "combine": return combine_losses(probs, masks)
    if loss_type == "mse":     return mse_loss(probs, masks)
    return criterion(probs, masks) + iou_loss(probs, masks)           # bce_iou


# ── Visualisation ─────────────────────────────────────────────────────────────

def _make_palette(n: int) -> np.ndarray:
    pal = np.zeros((n, 3), dtype=np.uint8)
    for i in range(1, n):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75 + 0.10 * (i % 3), 0.85 + 0.05 * (i % 2))
        pal[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return pal


def save_val_sample(model, dataset, device, out_dir, epoch, cat_names=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = random.randrange(len(dataset))
    img_t, mask_t = dataset[idx]
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(img_t.unsqueeze(0).to(device))
        logits = logits.cpu()

    img_pil = TF.to_pil_image(img_t)
    w, h    = img_pil.size

    if model.num_classes > 1:
        pal        = _make_palette(model.num_classes)
        gt_pil     = Image.fromarray(pal[mask_t.numpy().astype(np.int32)])
        pred_pil   = Image.fromarray(pal[logits[0].argmax(0).numpy().astype(np.int32)])
        pred_ids   = sorted(set(logits[0].argmax(0).numpy().flat) - {0})[:8]
        footer     = " ".join(
            cat_names.get(str(c), str(c)) if cat_names else str(c) for c in pred_ids)
    else:
        gt_pil   = TF.to_pil_image(mask_t.squeeze(0))
        pred_pil = TF.to_pil_image(torch.sigmoid(logits).squeeze())
        footer   = ""

    combined = Image.new('RGB', (w * 3, h))
    combined.paste(img_pil,                    (0,     0))
    combined.paste(gt_pil.resize((w,h), Image.NEAREST),   (w,     0))
    combined.paste(pred_pil.resize((w,h), Image.NEAREST), (w * 2, 0))
    try:
        draw = ImageDraw.Draw(combined)
        font = ImageFont.load_default()
        for i, lbl in enumerate(["RGB", "GT", "PRED"]):
            draw.text((i * w + w // 2 - 12, 4), lbl, fill=(255, 255, 255), font=font)
        if footer:
            draw.text((2 * w + 4, h - 14), footer[:60], fill=(255, 255, 0), font=font)
    except Exception:
        pass

    tw = int(combined.width * (480 / combined.height))
    combined.resize((tw, 480), Image.BILINEAR).save(
        out_dir / f"epoch_{epoch:04d}_sample_{idx}.png")


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    device      = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    data_root   = Path(args.data_root)
    num_classes = args.num_classes

    train_ds = SegmentationDataset(
        data_root / 'images' / 'train', data_root / 'masks' / 'train',
        image_size=args.image_size, augment=True, num_classes=num_classes)
    val_ds = SegmentationDataset(
        data_root / 'images' / 'val', data_root / 'masks' / 'val',
        image_size=args.image_size, augment=False, num_classes=num_classes)

    if len(train_ds) == 0:
        raise RuntimeError(f"No training samples in {data_root / 'images' / 'train'}")
    if len(val_ds) == 0:
        logging.warning("No val samples; using train split for validation")
        val_ds = train_ds

    kw = dict(num_workers=args.num_workers, pin_memory=device.type == 'cuda',
              prefetch_factor=2 if args.num_workers > 0 else None,
              persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **kw)

    model = sf_seg(num_channels=args.num_channels, focus_size=args.focus_size,
                   encoder_stride=args.encoder_stride, num_classes=num_classes).to(device)
    print(f"Model params: {model.get_num_parameters():,}  |  num_classes={num_classes}  |  device={device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = GradScaler('cuda', enabled=device.type == 'cuda')

    criterion = nn.BCELoss() if num_classes == 1 and args.loss_type in ("bce", "bce_iou") else None

    # ── Logging setup ──────────────────────────────────────────────────────────
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(log_dir / 'train.log'), logging.StreamHandler()])

    cat_names = None
    cat_map = data_root / "cat_to_idx.json"
    if cat_map.exists():
        cat_names = json.load(open(cat_map)).get("idx_to_name", {})

    class_weights = None
    if num_classes > 1:
        class_weights = compute_class_weights(
            data_root / 'masks' / 'train', num_classes,
            cache_path=data_root / 'class_freq.json').to(device)

    csv_path = log_dir / 'train_log.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_w    = csv.writer(csv_file)
    csv_w.writerow(['epoch',
                    'train_loss', 'train_seg', 'train_div', 'train_acc', 'train_miou',
                    'val_loss',   'val_seg',                'val_acc',   'val_miou'])

    # ── Checkpoint dirs ────────────────────────────────────────────────────────
    out_dir  = Path(args.output_dir);     out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    for p in ckpt_dir.glob("sf_seg_epoch_*.pt"):
        p.unlink(missing_ok=True)

    best_val  = float("inf")
    best_path = ckpt_dir / "sf_seg_best.pt"
    last_path = ckpt_dir / "sf_seg_last.pt"

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        fp = ckpt_dir / "sf_seg_last.pt" if str(args.resume).lower() in ("last","auto","true","1") \
             else Path(args.resume)
        if fp.exists():
            try:
                ckpt = torch.load(fp, map_location=device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    if ckpt.get("num_classes", 1) != num_classes:
                        logging.warning("Checkpoint num_classes mismatch — skipping resume")
                    else:
                        model.load_state_dict(ckpt["model_state_dict"])
                        try: optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                        except Exception: pass
                        try: scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                        except Exception: pass
                        start_epoch = ckpt.get("epoch", 0) + 1
                        best_val    = ckpt.get("best_val_loss", best_val)
                        logging.info(f"Resumed from {fp}, epoch {start_epoch}")
                else:
                    model.load_state_dict(ckpt)
            except Exception as e:
                logging.warning(f"Resume failed: {e}")
        else:
            logging.warning(f"Checkpoint not found: {fp}")

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    pin = device.type == 'cuda'
    for epoch in range(start_epoch, args.epochs + 1):

        # Train
        model.train()
        tr   = dict(loss=0., seg=0., div=0., acc=0.)
        seen = 0
        conf_tr = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for imgs, masks in bar:
            imgs, masks = imgs.to(device, non_blocking=pin), masks.to(device, non_blocking=pin)
            with autocast('cuda', enabled=device.type == 'cuda'):
                logits, _, attn = model(imgs)
                s = seg_loss(logits, masks, args.loss_type, criterion, num_classes,
                             args.no_obj_weight, class_weights)
                d = args.diversity_weight * diversity_loss(attn) if args.diversity_weight > 0 \
                    else torch.tensor(0., device=device)
                loss = s + d
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            b = imgs.size(0)
            with torch.no_grad():
                if num_classes > 1:
                    pred = logits.argmax(dim=1)
                    acc  = (pred == masks).float().mean().item()
                    update_confusion_matrix(conf_tr, pred, masks)
                else:
                    acc = ((torch.sigmoid(logits) > 0.5).float() == masks).float().mean().item()
            tr['loss'] += loss.item() * b; tr['seg'] += s.item() * b
            tr['div']  += d.item()    * b; tr['acc'] += acc * b; seen += b
            bar.set_postfix(loss=f"{loss.item():.4f}", seg=f"{s.item():.4f}",
                            div=f"{d.item():.4f}", acc=f"{acc:.4f}")

        tr = {k: v / seen for k, v in tr.items()}
        tr_miou, _ = miou_from_confusion(conf_tr.cpu()) if num_classes > 1 else (0., None)

        # Val
        model.eval()
        vl   = dict(loss=0., seg=0., acc=0.)
        vseen = 0
        conf_vl = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
        with torch.inference_mode():
            bar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]  ", leave=False)
            for imgs, masks in bar:
                imgs, masks = imgs.to(device, non_blocking=pin), masks.to(device, non_blocking=pin)
                with autocast('cuda', enabled=device.type == 'cuda'):
                    logits, _, _ = model(imgs)
                    s    = seg_loss(logits, masks, args.loss_type, criterion, num_classes,
                                    args.no_obj_weight, class_weights)
                b = imgs.size(0)
                if num_classes > 1:
                    pred = logits.argmax(dim=1)
                    acc  = (pred == masks).float().mean().item()
                    update_confusion_matrix(conf_vl, pred, masks)
                else:
                    acc = ((torch.sigmoid(logits) > 0.5).float() == masks).float().mean().item()
                vl['loss'] += s.item() * b; vl['seg'] += s.item() * b
                vl['acc']  += acc * b; vseen += b
                bar.set_postfix(loss=f"{s.item():.4f}", acc=f"{acc:.4f}")

        vl = {k: v / vseen for k, v in vl.items()} if vseen else vl
        vl_miou, vl_per = miou_from_confusion(conf_vl.cpu()) if num_classes > 1 else (0., None)

        # Per-class IoU top-10
        cls_info = ""
        if vl_per is not None and cat_names:
            ranked = sorted(
                [(vl_per[c].item(), c) for c in range(num_classes) if conf_vl.sum(1)[c] > 0],
                reverse=True)[:10]
            cls_info = "  " + "  ".join(
                f"{cat_names.get(str(c),str(c))}={v:.3f}" for v, c in ranked)

        lr = scheduler.get_last_lr()[0]; scheduler.step()
        logging.info(
            f"Epoch {epoch}/{args.epochs} | lr={lr:.2e} | "
            f"train loss={tr['loss']:.4f} seg={tr['seg']:.4f} div={tr['div']:.4f} "
            f"acc={tr['acc']:.4f} mIoU={tr_miou:.4f} | "
            f"val   loss={vl['loss']:.4f} seg={vl['seg']:.4f} "
            f"acc={vl['acc']:.4f} mIoU={vl_miou:.4f}" + cls_info)
        csv_w.writerow([epoch,
                        tr['loss'], tr['seg'], tr['div'], tr['acc'], tr_miou,
                        vl['loss'], vl['seg'],             vl['acc'], vl_miou])
        csv_file.flush()

        # Checkpoint
        try:
            improved = vseen and vl['loss'] < best_val
            if improved: best_val = vl['loss']
            ckpt = dict(epoch=epoch, model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(),
                        best_val_loss=best_val, num_channels=args.num_channels,
                        focus_size=args.focus_size, encoder_stride=args.encoder_stride,
                        num_classes=num_classes)
            torch.save(ckpt, last_path)
            if improved:
                torch.save(ckpt, best_path)
                logging.info(f"New best saved (val_loss={vl['loss']:.6f})")
        except Exception as e:
            logging.warning(f"Checkpoint save failed: {e}")

        try:
            save_val_sample(model, val_ds, device, out_dir, epoch, cat_names)
        except Exception as e:
            logging.warning(f"Sample save failed: {e}")

    csv_file.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",           default=None)
    p.add_argument("--data-root",        default=None)
    p.add_argument("--epochs",           type=int,   default=None)
    p.add_argument("--batch-size",       type=int,   default=None)
    p.add_argument("--lr",               type=float, default=None)
    p.add_argument("--num-workers",      type=int,   default=None)
    p.add_argument("--num-channels",     type=int,   default=None)
    p.add_argument("--focus-size",       type=int,   default=None)
    p.add_argument("--encoder-stride",   type=int,   default=None, choices=[1, 2])
    p.add_argument("--diversity-weight", type=float, default=None)
    p.add_argument("--num-classes",      type=int,   default=None)
    p.add_argument("--no-obj-weight",    type=float, default=None)
    p.add_argument("--loss-type",        default=None,
                   choices=["iou", "bce", "bce_iou", "combine", "mse", "ce", "ce_iou"])
    p.add_argument("--resume",           default=None)
    p.add_argument("--image-size",       type=int,   default=None)
    p.add_argument("--log-dir",          default=None)
    p.add_argument("--output-dir",       default=None)
    p.add_argument("--checkpoint-dir",   default=None)
    p.add_argument("--cpu",              action="store_true")
    return p.parse_args()


def merge_config(args):
    cfg = {}
    if args.config:
        cfg = json.load(open(args.config))
    elif Path("config.json").exists():
        cfg = json.load(open("config.json"))

    defaults = dict(
        data_root="data", epochs=200, batch_size=32, lr=1e-4, num_workers=4,
        num_channels=64, focus_size=32, encoder_stride=2,
        diversity_weight=0.1, num_classes=81, no_obj_weight=0.01,
        log_dir="logs", output_dir="outputs", checkpoint_dir="checkpoints",
        loss_type="ce_iou", resume=None, image_size=224,
    )
    for key, default in defaults.items():
        cli = getattr(args, key, None)
        setattr(args, key, cli if cli is not None else cfg.get(key, default))
    return args


if __name__ == "__main__":
    args = merge_config(parse_args())
    print("Config:")
    for k in ["data_root", "epochs", "batch_size", "lr", "num_workers",
              "num_channels", "focus_size", "encoder_stride",
              "num_classes", "no_obj_weight", "diversity_weight",
              "loss_type", "resume", "image_size"]:
        print(f"  {k}: {getattr(args, k)}")
    train(args)
