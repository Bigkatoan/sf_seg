#!/usr/bin/env python3
"""Stage-1 pretraining: SFBackbone + presence head trên chính ADE20K.

Bài toán: multi-label classification "class nào có mặt trong ảnh" (BCE,
label lấy free từ mask). Dễ hơn segmentation nhiều → backbone học cách
"nhìn tổng quan" nhanh, sau đó stage-2 train full model với backbone
slow-update (--backbone-lr-factor 0.1).

Key names (backbone.*, presence_head.*) khớp sf_seg_v2 → trainer load
trực tiếp qua --resume <ckpt> --restart (strict=False, phần còn lại fresh).

Usage:
    ./train_backbone.sh [--epochs 40] [--batch-size 56] [--lr 1e-3]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.sf_seg_v2 import SFBackbone
from src.training.trainer import (ADE20KDataset, build_class_aware_sampler,
                                  presence_target, update_presence_counts,
                                  presence_f1)


class PresencePretrainNet(nn.Module):
    """SFBackbone + GAP/GMP presence head — attribute names khớp sf_seg_v2."""

    def __init__(self, num_channels: int = 32, num_classes: int = 150,
                 variant: str = 'micro', dw_kernel: int = 3):
        super().__init__()
        C = num_channels
        self.backbone = SFBackbone((C, 2 * C, 4 * C, 8 * C), variant=variant,
                                   dw_kernel=dw_kernel, grad_checkpoint=False)
        self.presence_head = nn.Linear(8 * C, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, _, f4 = self.backbone(x)
        g = f4.mean(dim=(2, 3)) + f4.amax(dim=(2, 3))
        return self.presence_head(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',     default='config.json')
    ap.add_argument('--epochs',     type=int,   default=40)
    ap.add_argument('--batch-size', type=int,   default=56)
    ap.add_argument('--lr',         type=float, default=1e-3)
    ap.add_argument('--out',        default='checkpoints/backbone_pres.pt')
    ap.add_argument('--resume',     default=None,
                    help="'last' hoặc đường dẫn ckpt — train tiếp từ epoch đã dừng")
    args = ap.parse_args()

    cfg    = json.loads(Path(args.config).read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    C      = cfg['num_classes']
    pos_w  = float(cfg.get('presence_pos_weight', 4.0))

    log_dir = Path(cfg.get('log_dir', 'logs'))
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_dir / 'pretrain_presence.log')])

    data_root = Path(cfg['data_root'])
    train_ds = ADE20KDataset(
        data_root / 'images' / 'train', data_root / 'masks' / 'train',
        image_size=cfg['image_size'], augment=True,
        aug_hflip=cfg.get('aug_hflip', True),
        aug_resized_crop=cfg.get('aug_resized_crop', True),
        aug_color_jitter=cfg.get('aug_color_jitter', True),
        aug_cutout=cfg.get('aug_cutout', True),
        aug_shift=cfg.get('aug_shift', True),
        aug_hue=cfg.get('aug_hue', False))
    val_ds = ADE20KDataset(
        data_root / 'images' / 'val', data_root / 'masks' / 'val',
        image_size=cfg['image_size'], augment=False)

    kw = dict(num_workers=cfg.get('num_workers', 4), pin_memory=True,
              prefetch_factor=2, persistent_workers=True)
    # Class-aware sampler (accum=1): mỗi batch đủ 150 class → presence của
    # class hiếm xuất hiện trong mọi gradient. batch_size >= 51 để cover đủ.
    sampler = build_class_aware_sampler(
        train_ds, C, log_dir / 'class_index.pt', args.batch_size, accum_steps=1)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, **kw)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kw)

    model = PresencePretrainNet(
        num_channels=cfg['num_channels'], num_classes=C,
        variant=cfg['backbone_variant'], dw_kernel=cfg['dw_kernel']).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"PresencePretrainNet: {n_params/1e6:.2f}M params  "
                 f"bs={args.batch_size}  lr={args.lr}  pos_weight={pos_w}  "
                 f"epochs={args.epochs}")

    opt    = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.PolynomialLR(opt, total_iters=args.epochs, power=0.9)
    scaler = GradScaler()
    pw_t   = torch.full((C,), pos_w, device=device)

    csv_f = open(log_dir / 'pretrain_log.csv', 'a', newline='')
    csv_w = csv.writer(csv_f)
    if csv_f.tell() == 0:
        csv_w.writerow(['epoch', 'train_loss', 'train_f1', 'val_loss', 'val_f1'])

    # ── Resume ────────────────────────────────────────────────────────────────
    best_f1, start_epoch = 0.0, 1
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
            best_f1     = ckpt.get('best_val_f1', ckpt.get('val_pres_f1', 0.0))
            logging.info(f"Resumed from {fp}: epoch {start_epoch}, best F1={best_f1:.3f}")
        else:
            logging.warning(f"Checkpoint not found: {fp} — bắt đầu từ epoch 1")

    for epoch in range(start_epoch, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        tr_loss, seen = 0.0, 0
        cnt_tr = dict(tp=0., fp=0., fn=0.)
        for imgs, masks in tqdm(train_loader, desc=f"Pretrain {epoch}/{args.epochs}",
                                leave=False):
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with autocast('cuda', enabled=device.type == 'cuda'):
                logits = model(imgs)
                tgt    = presence_target(masks, C)
                loss   = F.binary_cross_entropy_with_logits(
                    logits.float(), tgt, pos_weight=pw_t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            update_presence_counts(cnt_tr, logits, tgt)
            b = imgs.size(0)
            tr_loss += loss.item() * b
            seen    += b
        tr_loss /= max(seen, 1)
        tr_f1    = presence_f1(cnt_tr)

        model.eval()
        vl_loss, vseen = 0.0, 0
        cnt_vl = dict(tp=0., fp=0., fn=0.)
        with torch.inference_mode():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with autocast('cuda', enabled=device.type == 'cuda'):
                    logits = model(imgs)
                    tgt    = presence_target(masks, C)
                    loss   = F.binary_cross_entropy_with_logits(
                        logits.float(), tgt, pos_weight=pw_t)
                update_presence_counts(cnt_vl, logits, tgt)
                vl_loss += loss.item() * imgs.size(0)
                vseen   += imgs.size(0)
        vl_loss /= max(vseen, 1)
        vl_f1    = presence_f1(cnt_vl)
        sched.step()

        logging.info(f"Pretrain {epoch}/{args.epochs} | "
                     f"train loss={tr_loss:.4f} F1={tr_f1:.3f} | "
                     f"val loss={vl_loss:.4f} F1={vl_f1:.3f}")
        csv_w.writerow([epoch, tr_loss, tr_f1, vl_loss, vl_f1])
        csv_f.flush()

        if vl_f1 > best_f1:
            best_f1 = vl_f1
        ckpt = dict(model_state_dict=model.state_dict(), epoch=epoch,
                    optimizer_state_dict=opt.state_dict(),
                    scheduler_state_dict=sched.state_dict(),
                    val_pres_f1=vl_f1, best_val_f1=best_f1)
        torch.save(ckpt, Path(args.out).with_suffix('.last.pt'))
        if vl_f1 >= best_f1:
            torch.save(ckpt, args.out)
            logging.info(f"  ↳ best saved (val_F1={vl_f1:.3f}) → {args.out}")

    logging.info(f"Done. Best val presence F1 = {best_f1:.3f}")
    logging.info("Stage 2: ./train.sh --resume "
                 f"{args.out} --restart --backbone-lr-factor 0.1")


if __name__ == '__main__':
    main()
