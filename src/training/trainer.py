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

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

from src.losses import (iou_loss, combine_losses, mse_loss, diversity_loss,
                         multiclass_iou_loss, ce_iou_loss,
                         focal_loss, focal_iou_loss, pure_focal_iou_loss,
                         attention_guide_loss, attention_exclusivity_loss,
                         edge_corner_loss)
from src.models import sf_seg as _sf_seg_custom
from src.models import sf_seg_r18 as _sf_seg_r18


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

        img  = img.resize((sz, sz),  Image.BILINEAR)
        mask = mask.resize((sz, sz), Image.NEAREST)

        if self.augment:
            # ── Spatial augmentation (image + mask together) ──────────────────
            # Random horizontal flip
            if random.random() > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)

            # Random resized crop: scale [0.5, 2.0] → crop → resize to sz
            # Simulates viewing the same scene at different distances
            scale  = random.uniform(0.5, 2.0)
            crop_w = int(sz / scale)
            crop_h = int(sz / scale)
            # Pad if crop size exceeds image size
            pad_w  = max(0, crop_w - sz)
            pad_h  = max(0, crop_h - sz)
            if pad_w > 0 or pad_h > 0:
                img  = TF.pad(img,  [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2],
                              fill=0)
                mask = TF.pad(mask, [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2],
                              fill=0)
            iw, ih = img.size
            x0 = random.randint(0, max(0, iw - crop_w))
            y0 = random.randint(0, max(0, ih - crop_h))
            img  = TF.resized_crop(img,  y0, x0, crop_h, crop_w, (sz, sz), Image.BILINEAR)
            mask = TF.resized_crop(mask, y0, x0, crop_h, crop_w, (sz, sz), Image.NEAREST)

            img_t = TF.to_tensor(img).float()          # (3, H, W) in [0, 1]

            # ── Photometric augmentation (image only) ─────────────────────────
            # Color jitter: brightness, contrast, saturation, hue
            if random.random() > 0.5:
                img_t = TF.adjust_brightness(img_t, random.uniform(0.6, 1.4))
            if random.random() > 0.5:
                img_t = TF.adjust_contrast(img_t, random.uniform(0.6, 1.4))
            if random.random() > 0.5:
                img_t = TF.adjust_saturation(img_t, random.uniform(0.6, 1.4))
            if random.random() > 0.5:
                img_t = TF.adjust_hue(img_t, random.uniform(-0.1, 0.1))
            img_t = img_t.clamp(0.0, 1.0)

            # Gaussian noise
            if random.random() > 0.5:
                sigma = random.uniform(0.005, 0.03)
                img_t = (img_t + torch.randn_like(img_t) * sigma).clamp(0.0, 1.0)

            # Random erasing (Cutout) — forces context-based reasoning
            if random.random() > 0.7:
                _, H, W = img_t.shape
                rh = random.randint(H // 8, H // 4)
                rw = random.randint(W // 8, W // 4)
                ry = random.randint(0, H - rh)
                rx = random.randint(0, W - rw)
                img_t[:, ry:ry+rh, rx:rx+rw] = 0.0

            if self.num_classes > 1:
                mask_t = torch.from_numpy(np.array(mask)).long()
            else:
                mask_t = (TF.to_tensor(mask) > 0.5).float()
            return img_t, mask_t

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
             no_obj_weight=0.1, class_weights=None, absent_weight=0.2):
    if num_classes > 1:
        if loss_type == "pure_focal_iou":
            return pure_focal_iou_loss(logits, masks, absent_weight=absent_weight)
        if loss_type == "focal_iou":
            return focal_iou_loss(logits, masks,
                                  class_weights=None,
                                  no_obj_weight=no_obj_weight)
        if loss_type == "focal":
            return focal_loss(logits, masks, weight=None)
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
        smooth  = F.avg_pool2d(logits, kernel_size=5, stride=1, padding=2)
        probs   = torch.softmax(smooth[0], dim=0)
        conf, pred_idx = probs.max(dim=0)
        pred_idx[conf < 0.1] = 0

        pal        = _make_palette(model.num_classes)
        gt_pil     = Image.fromarray(pal[mask_t.numpy().astype(np.int32)])
        pred_pil   = Image.fromarray(pal[pred_idx.numpy().astype(np.int32)])
        pred_ids   = sorted(set(pred_idx.numpy().flat) - {0})[:8]
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **kw)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kw)

    _model_cls = _sf_seg_r18.sf_seg if args.backbone == "resnet18" else _sf_seg_custom.sf_seg
    model = _model_cls(num_channels=args.num_channels, focus_size=args.focus_size,
                       encoder_stride=args.encoder_stride, num_classes=num_classes,
                       decoder_type=args.decoder_type,
                       encoder_pretrained=args.encoder_pretrained).to(device)
    print(f"Backbone: {args.backbone}  |  params: {model.get_num_parameters():,}  |  "
          f"num_classes={num_classes}  |  decoder={args.decoder_type}  |  device={device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    optimizer    = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_ep    = min(5, args.epochs // 10)          # 5-epoch linear warmup
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_ep)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_ep), eta_min=args.lr * 0.01)
    scheduler    = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_ep])
    scaler = GradScaler('cuda', enabled=device.type == 'cuda')

    criterion = nn.BCELoss() if num_classes == 1 and args.loss_type in ("bce", "bce_iou") else None

    # ── Logging setup ──────────────────────────────────────────────────────────
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(log_dir / 'train.log'), logging.StreamHandler()])

    tb_writer = None
    if _TB_AVAILABLE:
        tb_dir = log_dir / 'tensorboard' / args.backbone
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        logging.info(f"TensorBoard → tensorboard --logdir {log_dir / 'tensorboard'}")
        try:
            dummy = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
            tb_writer.add_graph(model, dummy)
            logging.info("Model graph pushed to TensorBoard")
        except Exception as e:
            logging.warning(f"add_graph failed: {e}")
    else:
        logging.warning("TensorBoard not available — pip install tensorboard")

    cat_names = None
    cat_map = data_root / "cat_to_idx.json"
    if cat_map.exists():
        cat_names = json.load(open(cat_map)).get("idx_to_name", {})

    class_weights = None
    if num_classes > 1 and args.loss_type not in ("focal", "focal_iou", "pure_focal_iou"):
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

    # ── TensorBoard image helper ───────────────────────────────────────────────
    # Grab one fixed val batch for consistent visualization across epochs
    _tb_imgs, _tb_masks = None, None
    for _b, _m in val_loader:
        _tb_imgs  = _b[:4].to(device)   # max 4 samples
        _tb_masks = _m[:4].to(device)
        break

    def _tb_log_images(writer, model, epoch, num_classes, device):
        """Log image / GT / pred / attention to TensorBoard."""
        if writer is None or _tb_imgs is None:
            return
        model.eval()
        with torch.no_grad():
            logits, _, attn = model(_tb_imgs)

        imgs_cpu = _tb_imgs.cpu().float()

        # Predicted class per pixel (argmax for multi-class, sigmoid>0.5 for binary)
        if num_classes > 1:
            pred_idx = logits.argmax(dim=1).cpu().float() / max(num_classes - 1, 1)
            gt_idx   = _tb_masks.cpu().float() / max(num_classes - 1, 1)
        else:
            pred_idx = (torch.sigmoid(logits) > 0.5).cpu().float().squeeze(1)
            gt_idx   = _tb_masks.cpu().float()

        # Clamp images to [0,1] (denormalize if needed)
        imgs_show = imgs_cpu.clamp(0, 1)

        writer.add_images("Val/image",    imgs_show,              epoch)
        writer.add_images("Val/gt_mask",  gt_idx.unsqueeze(1),    epoch)
        writer.add_images("Val/pred",     pred_idx.unsqueeze(1),  epoch)

        # Attention map: mean across channels, upsampled to input resolution
        H, W = _tb_imgs.shape[2], _tb_imgs.shape[3]
        attn_mean = attn.cpu().float().mean(dim=1, keepdim=True)
        attn_mean = F.interpolate(attn_mean, size=(H, W), mode='bilinear', align_corners=False)
        attn_norm = (attn_mean - attn_mean.amin(dim=[2,3], keepdim=True)) / \
                    (attn_mean.amax(dim=[2,3], keepdim=True) - attn_mean.amin(dim=[2,3], keepdim=True) + 1e-8)
        writer.add_images("Val/attn_head_large", attn_norm, epoch)

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
                             args.no_obj_weight, class_weights, args.absent_weight)
                zero = logits.new_tensor(0.0)
                d = args.diversity_weight * diversity_loss(attn) if args.diversity_weight > 0 \
                    else zero
                g = args.attn_guide_weight * attention_guide_loss(
                        attn, masks, num_classes) \
                    if args.attn_guide_weight > 0 and num_classes > 1 \
                    else zero
                e = args.attn_exclusive_weight * attention_exclusivity_loss(
                        attn, masks, num_classes) \
                    if args.attn_exclusive_weight > 0 and num_classes > 1 \
                    else zero
                sp_raw = model.routing_sparsity_loss()
                sp = args.sparse_weight * sp_raw \
                     if args.sparse_weight > 0 and sp_raw is not None \
                     else zero
                bd = args.boundary_weight * edge_corner_loss(
                        logits, masks,
                        edge_weight=args.edge_weight,
                        corner_weight=args.corner_weight) \
                     if args.boundary_weight > 0 and num_classes > 1 \
                     else zero
                loss = s + d + g + e + sp + bd

            if not torch.isfinite(loss):
                logging.warning(
                    f"Non-finite loss ({loss.item():.4f}) at epoch {epoch} "
                    f"seg={s.item():.4f} div={d.item():.4f} "
                    f"guide={g.item():.4f} excl={e.item():.4f} "
                    f"boundary={bd.item():.4f} — skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                                    args.no_obj_weight, class_weights, args.absent_weight)
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
        routing_info = ""
        if args.decoder_type == "sparse":
            rs = model.routing_weight_stats()
            routing_info = (f" | routing sparsity={rs['routing_sparsity']:.2%}"
                            f" mean={rs['routing_mean']:.4f}")
        logging.info(
            f"Epoch {epoch}/{args.epochs} | lr={lr:.2e} | "
            f"train loss={tr['loss']:.4f} seg={tr['seg']:.4f} div={tr['div']:.4f} "
            f"acc={tr['acc']:.4f} mIoU={tr_miou:.4f} | "
            f"val   loss={vl['loss']:.4f} seg={vl['seg']:.4f} "
            f"acc={vl['acc']:.4f} mIoU={vl_miou:.4f}" + cls_info + routing_info)
        csv_w.writerow([epoch,
                        tr['loss'], tr['seg'], tr['div'], tr['acc'], tr_miou,
                        vl['loss'], vl['seg'],             vl['acc'], vl_miou])
        csv_file.flush()

        # ── TensorBoard scalars ────────────────────────────────────────────────
        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train",      tr['loss'],   epoch)
            tb_writer.add_scalar("Loss/val",        vl['loss'],   epoch)
            tb_writer.add_scalar("SegLoss/train",   tr['seg'],    epoch)
            tb_writer.add_scalar("DivLoss/train",   tr['div'],    epoch)
            tb_writer.add_scalar("mIoU/train",      tr_miou,      epoch)
            tb_writer.add_scalar("mIoU/val",        vl_miou,      epoch)
            tb_writer.add_scalar("Accuracy/train",  tr['acc'],    epoch)
            tb_writer.add_scalar("Accuracy/val",    vl['acc'],    epoch)
            tb_writer.add_scalar("LR",              lr,           epoch)
            if epoch == 1 or epoch % 5 == 0:
                _tb_log_images(tb_writer, model, epoch, num_classes, device)

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
    if tb_writer is not None:
        tb_writer.close()


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
    p.add_argument("--absent-weight",    type=float, default=None)
    p.add_argument("--attn-guide-weight",     type=float, default=None)
    p.add_argument("--attn-exclusive-weight", type=float, default=None)
    p.add_argument("--decoder-type",   default=None, choices=["dense", "sparse"])
    p.add_argument("--sparse-weight",   type=float, default=None)
    p.add_argument("--boundary-weight", type=float, default=None)
    p.add_argument("--edge-weight",     type=float, default=None)
    p.add_argument("--corner-weight",   type=float, default=None)
    p.add_argument("--class-diverse",  action="store_true", default=None)
    p.add_argument("--loss-type",        default=None,
                   choices=["iou", "bce", "bce_iou", "combine", "mse", "ce", "ce_iou",
                            "focal", "focal_iou", "pure_focal_iou"])
    p.add_argument("--backbone",          default=None,
                   choices=["custom", "resnet18"],
                   help="Backbone type: custom (default) or resnet18 (ImageNet pretrained)")
    p.add_argument("--encoder-pretrained", default=None,
                   help="Path to enc_best.pt from pretrain_encoder.py")
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
        absent_weight=0.2, attn_guide_weight=0.0, attn_exclusive_weight=0.0,
        decoder_type="dense", sparse_weight=0.0, class_diverse=False,
        boundary_weight=0.3, edge_weight=4.0, corner_weight=6.0,
        log_dir="logs", output_dir="outputs", checkpoint_dir="checkpoints",
        loss_type="ce_iou", resume=None, image_size=224, encoder_pretrained=None,
        backbone="custom",
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
