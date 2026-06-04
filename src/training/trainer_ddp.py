#!/usr/bin/env python3
"""Training script for sf_seg — multi-GPU (DDP) variant for Google Colab dual T4.

Launch via train.sh which auto-detects GPU count.
For manual launch:
    torchrun --nproc_per_node=2 -m src.training.trainer_ddp [args]

DDP notes:
  - Each GPU processes batch_size samples → effective batch = batch_size × world_size
  - Confusion matrices are all-reduced across ranks for correct global mIoU
  - Only rank 0 saves checkpoints, writes CSV, prints per-epoch summary
  - GradScaler works with DDP out of the box (NCCL handles gradient sync)
"""
from __future__ import annotations

import argparse
import colorsys
import csv
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

from src.losses import (iou_loss, combine_losses, mse_loss, diversity_loss,
                         multiclass_iou_loss, ce_iou_loss)
from src.models import sf_seg


# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp() -> tuple[int, int, int]:
    """Init process group, return (local_rank, rank, world_size)."""
    dist.init_process_group(backend="nccl")
    local_rank  = int(os.environ["LOCAL_RANK"])
    rank        = dist.get_rank()
    world_size  = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

def is_rank0(rank: int) -> bool:
    return rank == 0

def all_reduce_mean(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size

def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


# ── Dataset ───────────────────────────────────────────────────────────────────

class SegmentationDataset(Dataset):
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

        img_t = TF.to_tensor(img)
        if self.num_classes > 1:
            mask_t = torch.from_numpy(np.array(mask)).long()
        else:
            mask_t = (TF.to_tensor(mask) > 0.5).float()
        return img_t, mask_t


# ── Metrics ───────────────────────────────────────────────────────────────────

def update_confusion_matrix(conf: torch.Tensor, pred: torch.Tensor,
                             target: torch.Tensor) -> None:
    C  = conf.shape[0]
    p  = pred.view(-1).long().clamp(0, C - 1)
    t  = target.view(-1).long().clamp(0, C - 1)
    conf += torch.bincount(t * C + p, minlength=C * C).view(C, C)


def miou_from_confusion(conf: torch.Tensor) -> tuple:
    inter     = conf.diagonal().float()
    union     = (conf.sum(1) + conf.sum(0) - inter).float()
    per_class = inter / union.clamp(min=1e-8)
    valid     = conf.sum(1) > 0
    return (per_class[valid].mean().item() if valid.any() else 0.0), per_class


# ── Class-frequency weights ───────────────────────────────────────────────────

def compute_class_weights(masks_dir: Path, num_classes: int,
                           cache_path: Path | None = None,
                           sample_size: int = 4000) -> torch.Tensor:
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
        return F.cross_entropy(logits, masks, weight=class_weights)
    probs = torch.sigmoid(logits)
    if loss_type == "iou":     return iou_loss(probs, masks)
    if loss_type == "bce":     return criterion(probs, masks)
    if loss_type == "combine": return combine_losses(probs, masks)
    if loss_type == "mse":     return mse_loss(probs, masks)
    return criterion(probs, masks) + iou_loss(probs, masks)


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
    raw = model.module if isinstance(model, DDP) else model
    idx = random.randrange(len(dataset))
    img_t, mask_t = dataset[idx]
    raw.eval()
    with torch.no_grad():
        logits, _, _ = raw(img_t.unsqueeze(0).to(device))
        logits = logits.cpu()

    img_pil = TF.to_pil_image(img_t)
    w, h    = img_pil.size

    if raw.num_classes > 1:
        pal      = _make_palette(raw.num_classes)
        gt_pil   = Image.fromarray(pal[mask_t.numpy().astype(np.int32)])
        pred_pil = Image.fromarray(pal[logits[0].argmax(0).numpy().astype(np.int32)])
        pred_ids = sorted(set(logits[0].argmax(0).numpy().flat) - {0})[:8]
        footer   = " ".join(
            cat_names.get(str(c), str(c)) if cat_names else str(c) for c in pred_ids)
    else:
        gt_pil   = TF.to_pil_image(mask_t.squeeze(0))
        pred_pil = TF.to_pil_image(torch.sigmoid(logits).squeeze())
        footer   = ""

    combined = Image.new('RGB', (w * 3, h))
    combined.paste(img_pil,                                 (0,     0))
    combined.paste(gt_pil.resize((w, h), Image.NEAREST),   (w,     0))
    combined.paste(pred_pil.resize((w, h), Image.NEAREST), (w * 2, 0))
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
    local_rank, rank, world_size = setup_ddp()
    device      = torch.device(f"cuda:{local_rank}")
    rank0       = is_rank0(rank)
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
    if len(val_ds) == 0 and rank0:
        logging.warning("No val samples; using train split for validation")
        val_ds = train_ds

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size,
                                       rank=rank, shuffle=False)

    # Cap workers per process: DDP spawns world_size processes, each with num_workers
    nw = min(args.num_workers, max(1, 4 // world_size))
    kw = dict(num_workers=nw, pin_memory=True,
              prefetch_factor=2 if nw > 0 else None,
              persistent_workers=nw > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              sampler=val_sampler,   **kw)

    raw_model = sf_seg(num_channels=args.num_channels, focus_size=args.focus_size,
                       encoder_stride=args.encoder_stride,
                       num_classes=num_classes).to(device)
    model = DDP(raw_model, device_ids=[local_rank])

    if rank0:
        eff_bs = args.batch_size * world_size
        print(f"Model params: {raw_model.get_num_parameters():,}  |  "
              f"num_classes={num_classes}  |  "
              f"GPUs={world_size}  |  "
              f"effective batch={eff_bs}")
        for i in range(world_size):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler    = GradScaler('cuda')

    criterion = (nn.BCELoss() if num_classes == 1
                 and args.loss_type in ("bce", "bce_iou") else None)

    if rank0:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, handlers=[
            logging.FileHandler(log_dir / 'train.log'),
            logging.StreamHandler()])

        cat_names = None
        cat_map = data_root / "cat_to_idx.json"
        if cat_map.exists():
            cat_names = json.load(open(cat_map)).get("idx_to_name", {})
    else:
        cat_names = None
        logging.basicConfig(level=logging.WARNING)

    class_weights = None
    if num_classes > 1:
        # Compute on rank 0, move to CUDA, then broadcast (NCCL requires CUDA tensors)
        if rank0:
            cw = compute_class_weights(
                data_root / 'masks' / 'train', num_classes,
                cache_path=data_root / 'class_freq.json').to(device)
        else:
            cw = torch.zeros(num_classes, device=device)
        dist.broadcast(cw, src=0)
        class_weights = cw

    if rank0:
        out_dir  = Path(args.output_dir);     out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
        for p in ckpt_dir.glob("sf_seg_epoch_*.pt"):
            p.unlink(missing_ok=True)
        csv_path = Path(args.log_dir) / 'train_log.csv'
        csv_file = open(csv_path, 'w', newline='')
        csv_w    = csv.writer(csv_file)
        csv_w.writerow(['epoch',
                        'train_loss', 'train_seg', 'train_div', 'train_acc', 'train_miou',
                        'val_loss',   'val_seg',                'val_acc',   'val_miou'])
        best_val  = float("inf")
        best_path = ckpt_dir / "sf_seg_best.pt"
        last_path = ckpt_dir / "sf_seg_last.pt"
    else:
        best_val = float("inf")

    # ── Resume (rank 0 loads, broadcasts to others) ────────────────────────────
    start_epoch = 1
    if args.resume:
        fp = (Path(args.checkpoint_dir) / "sf_seg_last.pt"
              if str(args.resume).lower() in ("last", "auto", "true", "1")
              else Path(args.resume))
        if fp.exists() and rank0:
            try:
                ckpt = torch.load(fp, map_location="cpu")
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    if ckpt.get("num_classes", 1) != num_classes:
                        logging.warning("Checkpoint num_classes mismatch — skipping resume")
                    else:
                        raw_model.load_state_dict(ckpt["model_state_dict"])
                        try: optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                        except Exception: pass
                        try: scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                        except Exception: pass
                        start_epoch = ckpt.get("epoch", 0) + 1
                        best_val    = ckpt.get("best_val_loss", best_val)
                        logging.info(f"Resumed from {fp}, epoch {start_epoch}")
                else:
                    raw_model.load_state_dict(ckpt)
            except Exception as e:
                logging.warning(f"Resume failed: {e}")
        elif not fp.exists() and rank0:
            logging.warning(f"Checkpoint not found: {fp}")
    dist.barrier()

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)   # different shuffle per epoch

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        # Accumulators: 5 values — [loss, seg, div, acc, count]
        stats = torch.zeros(5, device=device)

        bar = (tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
               if rank0 else train_loader)
        conf_tr = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

        for imgs, masks in bar:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with autocast('cuda'):
                logits, _, attn = model(imgs)
                s    = seg_loss(logits, masks, args.loss_type, criterion, num_classes,
                                args.no_obj_weight, class_weights)
                d    = args.diversity_weight * diversity_loss(attn) \
                       if args.diversity_weight > 0 else torch.zeros(1, device=device)
                loss = s + d

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                b = imgs.size(0)
                if num_classes > 1:
                    pred = logits.argmax(dim=1)
                    acc  = (pred == masks).float().mean()
                    update_confusion_matrix(conf_tr, pred, masks)
                else:
                    acc = ((torch.sigmoid(logits) > 0.5).float() == masks).float().mean()
                stats += torch.tensor([loss.item() * b, s.item() * b,
                                       d.item() * b,   acc.item() * b, b],
                                      device=device)
            if rank0:
                bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.4f}")

        # All-reduce stats and confusion matrix across GPUs
        all_reduce_sum(stats)
        all_reduce_sum(conf_tr)
        n = stats[4].item()
        tr_loss, tr_seg, tr_div, tr_acc = (stats[:4] / n).tolist()
        tr_miou, _ = miou_from_confusion(conf_tr.cpu()) if num_classes > 1 else (0., None)

        # ── Val ────────────────────────────────────────────────────────────────
        model.eval()
        vstats  = torch.zeros(4, device=device)   # [loss, seg, acc, count]
        conf_vl = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

        with torch.inference_mode():
            bar = (tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]  ", leave=False)
                   if rank0 else val_loader)
            for imgs, masks in bar:
                imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                with autocast('cuda'):
                    logits, _, _ = model(imgs)
                    s = seg_loss(logits, masks, args.loss_type, criterion, num_classes,
                                 args.no_obj_weight, class_weights)
                b = imgs.size(0)
                if num_classes > 1:
                    pred = logits.argmax(dim=1)
                    acc  = (pred == masks).float().mean()
                    update_confusion_matrix(conf_vl, pred, masks)
                else:
                    acc = ((torch.sigmoid(logits) > 0.5).float() == masks).float().mean()
                vstats += torch.tensor([s.item() * b, s.item() * b,
                                        acc.item() * b, b], device=device)
                if rank0:
                    bar.set_postfix(loss=f"{s.item():.4f}", acc=f"{acc.item():.4f}")

        all_reduce_sum(vstats)
        all_reduce_sum(conf_vl)
        vn = vstats[3].item()
        vl_loss, vl_seg, vl_acc = (vstats[:3] / max(vn, 1)).tolist()
        vl_miou, vl_per = miou_from_confusion(conf_vl.cpu()) if num_classes > 1 else (0., None)

        scheduler.step()

        if rank0:
            lr = scheduler.get_last_lr()[0]
            cls_info = ""
            if vl_per is not None and cat_names:
                ranked = sorted(
                    [(vl_per[c].item(), c) for c in range(num_classes)
                     if conf_vl.sum(1)[c] > 0],
                    reverse=True)[:10]
                cls_info = "  " + "  ".join(
                    f"{cat_names.get(str(c), str(c))}={v:.3f}" for v, c in ranked)

            logging.info(
                f"Epoch {epoch}/{args.epochs} | lr={lr:.2e} | "
                f"train loss={tr_loss:.4f} seg={tr_seg:.4f} div={tr_div:.4f} "
                f"acc={tr_acc:.4f} mIoU={tr_miou:.4f} | "
                f"val   loss={vl_loss:.4f} seg={vl_seg:.4f} "
                f"acc={vl_acc:.4f} mIoU={vl_miou:.4f}" + cls_info)
            csv_w.writerow([epoch,
                            tr_loss, tr_seg, tr_div, tr_acc, tr_miou,
                            vl_loss, vl_seg,           vl_acc, vl_miou])
            csv_file.flush()

            improved = vn > 0 and vl_loss < best_val
            if improved: best_val = vl_loss
            ckpt = dict(epoch=epoch,
                        model_state_dict=raw_model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(),
                        best_val_loss=best_val, num_channels=args.num_channels,
                        focus_size=args.focus_size, encoder_stride=args.encoder_stride,
                        num_classes=num_classes)
            try:
                torch.save(ckpt, last_path)
                if improved:
                    torch.save(ckpt, best_path)
                    logging.info(f"New best saved (val_loss={vl_loss:.6f})")
            except Exception as e:
                logging.warning(f"Checkpoint save failed: {e}")

            try:
                save_val_sample(model, val_ds, device, out_dir, epoch, cat_names)
            except Exception as e:
                logging.warning(f"Sample save failed: {e}")

        dist.barrier()

    if rank0:
        csv_file.close()
    cleanup_ddp()


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
    return p.parse_args()


def merge_config(args):
    cfg = {}
    if args.config:
        cfg = json.load(open(args.config))
    elif Path("config.json").exists():
        cfg = json.load(open("config.json"))

    defaults = dict(
        data_root="data", epochs=200, batch_size=16, lr=1e-3, num_workers=4,
        num_channels=64, focus_size=32, encoder_stride=2,
        diversity_weight=0.1, num_classes=151, no_obj_weight=0.01,
        log_dir="logs", output_dir="outputs", checkpoint_dir="checkpoints",
        loss_type="ce_iou", resume=None, image_size=224,
    )
    for key, default in defaults.items():
        cli = getattr(args, key, None)
        setattr(args, key, cli if cli is not None else cfg.get(key, default))
    return args


if __name__ == "__main__":
    args = merge_config(parse_args())
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("Config:")
        for k in ["data_root", "epochs", "batch_size", "lr", "num_workers",
                  "num_channels", "focus_size", "encoder_stride",
                  "num_classes", "no_obj_weight", "diversity_weight",
                  "loss_type", "resume", "image_size"]:
            print(f"  {k}: {getattr(args, k)}")
    train(args)
