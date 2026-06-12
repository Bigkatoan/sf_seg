#!/usr/bin/env python3
"""ImageNet-1k pretraining cho SFBackbone (stage-0, trước presence/seg).

Classification 1000 class, recipe nhẹ kiểu ConvNeXt: AdamW + wd 0.05,
label smoothing 0.1, augment chỉ RandomResizedCrop + hflip (theo policy
project: không cutout/color-jitter). Key names backbone.* khớp sf_seg_v2
→ nạp thẳng: ./train.sh --resume checkpoints/backbone_in1k.pt --restart

Data layout (ImageFolder):
    data/imagenet/train/<wnid>/*.JPEG
    data/imagenet/val/<wnid>/*.JPEG

Nguồn tải (cần tài khoản, ~155GB):
  - Kaggle: kaggle competitions download -c imagenet-object-localization-challenge
    (val cần restructure theo wnid — xem LOC_val_solution.csv)
  - hoặc image-net.org: ILSVRC2012_img_train.tar + val + devkit

Usage:
    ./train_imagenet.sh [--epochs 100] [--batch-size 256] [--resume last]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.models.sf_seg_v2 import SFBackbone


class ImageNetNet(nn.Module):
    """SFBackbone + classifier — attribute name 'backbone' khớp sf_seg_v2."""

    def __init__(self, num_channels: int = 64, num_classes: int = 1000,
                 variant: str = 'small', dw_kernel: int = 3):
        super().__init__()
        C = num_channels
        self.backbone = SFBackbone((C, 2 * C, 4 * C, 8 * C), variant=variant,
                                   dw_kernel=dw_kernel, grad_checkpoint=False)
        # GAP+GMP giống cách sf_seg_v2 tính g — feature statistics khớp khi transfer
        self.cls_head = nn.Linear(8 * C, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, _, f4 = self.backbone(x)
        g = f4.mean(dim=(2, 3)) + f4.amax(dim=(2, 3))
        return self.cls_head(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',     default='config.json')
    ap.add_argument('--data-root',  default='data/imagenet')
    ap.add_argument('--epochs',     type=int,   default=100)
    ap.add_argument('--batch-size', type=int,   default=256)
    ap.add_argument('--lr',         type=float, default=1e-3)
    ap.add_argument('--image-size', type=int,   default=176)   # 176 nhanh ~1.6× vs 224, transfer OK
    ap.add_argument('--workers',    type=int,   default=16)
    ap.add_argument('--lr-schedule', choices=['cosine', 'constant'], default='cosine')
    ap.add_argument('--compile', action='store_true', help='torch.compile backbone (~1.3-1.5×)')
    ap.add_argument('--val-every', type=int, default=5, help='val mỗi N epoch (val 50K ảnh tốn ~30s)')
    ap.add_argument('--out',        default='checkpoints/backbone_in1k.pt')
    ap.add_argument('--resume',     default=None)
    args = ap.parse_args()

    cfg    = json.loads(Path(args.config).read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_dir = Path(cfg.get('log_dir', 'logs'))
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / 'pretrain_imagenet.log')])

    sz = args.image_size
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(sz, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm])
    val_tf = transforms.Compose([
        transforms.Resize(int(sz * 256 / 224)),
        transforms.CenterCrop(sz),
        transforms.ToTensor(), norm])

    root = Path(args.data_root)
    train_ds = datasets.ImageFolder(root / 'train', train_tf)
    val_ds   = datasets.ImageFolder(root / 'val',   val_tf)
    n_cls    = len(train_ds.classes)
    logging.info(f"ImageNet: {len(train_ds)} train / {len(val_ds)} val / {n_cls} classes")

    kw = dict(num_workers=args.workers, pin_memory=True,
              prefetch_factor=4, persistent_workers=True, drop_last=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **kw)
    kw['drop_last'] = False
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kw)

    model = ImageNetNet(num_channels=cfg['num_channels'], num_classes=n_cls,
                        variant=cfg['backbone_variant'],
                        dw_kernel=cfg['dw_kernel']).to(device)
    model = model.to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model)
        logging.info("torch.compile bật — epoch đầu chậm (warmup compile)")
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"ImageNetNet: {n_params/1e6:.2f}M  bs={args.batch_size}  "
                 f"lr={args.lr}  sched={args.lr_schedule}  epochs={args.epochs}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    warmup = min(5, args.epochs // 10)
    if args.lr_schedule == 'constant':
        sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=warmup)
    else:
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            [torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, total_iters=warmup),
             torch.optim.lr_scheduler.CosineAnnealingLR(
                 opt, T_max=max(1, args.epochs - warmup), eta_min=args.lr * 0.01)],
            milestones=[warmup])
    scaler = GradScaler()

    csv_f = open(log_dir / 'pretrain_imagenet.csv', 'a', newline='')
    csv_w = csv.writer(csv_f)
    if csv_f.tell() == 0:
        csv_w.writerow(['epoch', 'train_loss', 'train_top1', 'val_loss',
                        'val_top1', 'val_top5'])

    best_top1, start_epoch = 0.0, 1
    if args.resume:
        fp = (Path(args.out).with_suffix('.last.pt')
              if str(args.resume).lower() == 'last' else Path(args.resume))
        if fp.exists():
            ckpt = torch.load(fp, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            try: opt.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception: pass
            try: sched.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception: pass
            start_epoch = ckpt.get('epoch', 0) + 1
            best_top1   = ckpt.get('best_top1', 0.0)
            logging.info(f"Resumed {fp}: epoch {start_epoch}, best top1={best_top1:.3f}")
        else:
            logging.warning(f"Checkpoint not found: {fp}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        tr_loss = tr_corr = seen = 0
        bar = tqdm(train_loader, desc=f"IN1k {epoch}/{args.epochs}", leave=False)
        for imgs, labels in bar:
            imgs   = imgs.to(device, non_blocking=True,
                             memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            with autocast('cuda', enabled=device.type == 'cuda'):
                logits = model(imgs)
                loss   = F.cross_entropy(logits.float(), labels, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            b = imgs.size(0)
            tr_loss += loss.item() * b
            tr_corr += (logits.argmax(1) == labels).sum().item()
            seen    += b
            if seen % (100 * b) == 0:
                bar.set_postfix(loss=f"{tr_loss/seen:.3f}", top1=f"{tr_corr/seen:.3f}")
        tr_loss /= max(seen, 1)
        tr_top1  = tr_corr / max(seen, 1)

        model.eval()
        # Val mỗi N epoch (và epoch cuối) — tiết kiệm ~30s/epoch bỏ qua
        do_val = (epoch % args.val_every == 0) or (epoch == args.epochs)
        if not do_val:
            sched.step()
            logging.info(f"IN1k {epoch}/{args.epochs} | "
                         f"train loss={tr_loss:.4f} top1={tr_top1:.3f} | (skip val)")
            csv_w.writerow([epoch, tr_loss, tr_top1, '', '', ''])
            csv_f.flush()
            torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch,
                            optimizer_state_dict=opt.state_dict(),
                            scheduler_state_dict=sched.state_dict(),
                            val_top1=best_top1, best_top1=best_top1),
                       Path(args.out).with_suffix('.last.pt'))
            continue

        vl_loss = vl_c1 = vl_c5 = vseen = 0
        with torch.inference_mode():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device, non_blocking=True,
                                 memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=True)
                with autocast('cuda', enabled=device.type == 'cuda'):
                    logits = model(imgs)
                    loss   = F.cross_entropy(logits.float(), labels)
                top5 = logits.topk(5, dim=1).indices
                vl_c1   += (logits.argmax(1) == labels).sum().item()
                vl_c5   += (top5 == labels.unsqueeze(1)).any(1).sum().item()
                vl_loss += loss.item() * imgs.size(0)
                vseen   += imgs.size(0)
        vl_loss /= max(vseen, 1)
        v1, v5   = vl_c1 / max(vseen, 1), vl_c5 / max(vseen, 1)
        sched.step()

        logging.info(f"IN1k {epoch}/{args.epochs} | "
                     f"train loss={tr_loss:.4f} top1={tr_top1:.3f} | "
                     f"val loss={vl_loss:.4f} top1={v1:.3f} top5={v5:.3f}")
        csv_w.writerow([epoch, tr_loss, tr_top1, vl_loss, v1, v5])
        csv_f.flush()

        if v1 > best_top1:
            best_top1 = v1
        ckpt = dict(model_state_dict=model.state_dict(), epoch=epoch,
                    optimizer_state_dict=opt.state_dict(),
                    scheduler_state_dict=sched.state_dict(),
                    val_top1=v1, best_top1=best_top1)
        torch.save(ckpt, Path(args.out).with_suffix('.last.pt'))
        if v1 >= best_top1:
            torch.save(ckpt, args.out)
            logging.info(f"  ↳ best saved (top1={v1:.3f}) → {args.out}")

    logging.info(f"Done. Best val top-1 = {best_top1:.3f}")
    logging.info(f"Tiếp theo: ./train.sh --resume {args.out} --restart "
                 f"(hoặc qua presence pretrain trước)")


if __name__ == '__main__':
    main()
