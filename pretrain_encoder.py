"""pretrain_encoder.py — Pretrain the attention_head encoder on an image classification task.

The encoder (enc1 + enc2) from attention_head is extracted, a classification
head is attached, and the whole thing is trained with focal CE on any
ImageFolder-compatible dataset (ImageNet, Places365, iNaturalist, etc.).

After pretraining, only the encoder weights are saved. Load them into sf_seg
via the --encoder-pretrained argument to trainer.py.

Usage:
    # ImageNet (folder structure: train/classname/img.jpg)
    python pretrain_encoder.py --data /path/to/imagenet --num-classes 1000

    # Any ImageFolder dataset
    python pretrain_encoder.py --data /path/to/dataset --num-classes 100 --epochs 30

    # Resume
    python pretrain_encoder.py --data /path/to/imagenet --resume checkpoints/enc_last.pt

Output:
    checkpoints/enc_best.pt  — best val-acc encoder weights
    checkpoints/enc_last.pt  — latest checkpoint (for resume)

The saved file contains:
    {
        "enc1": state_dict,    # Conv(3→C, 3×3, stride=2) + ReLU
        "enc2": state_dict,    # Conv(C→2C, 3×3)
        "num_channels": C,
        "epoch": N,
        "val_acc": float,
    }
Load into sf_seg with:
    model = sf_seg(num_channels=128, ..., encoder_pretrained="checkpoints/enc_best.pt")
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


# ── Encoder (same as attention_head's enc1 + enc2) ────────────────────────────

class Encoder(nn.Module):
    def __init__(self, num_channels: int = 128):
        super().__init__()
        C = num_channels
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Conv2d(C, C * 2, 3, padding=1)

    def forward(self, x):
        return self.enc2(self.enc1(x))   # (B, 2C, H/2, W/2)


class EncoderClassifier(nn.Module):
    def __init__(self, num_channels: int = 128, num_classes: int = 1000):
        super().__init__()
        self.encoder = Encoder(num_channels)
        C = num_channels
        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C * 2, C),
            nn.ReLU(inplace=True),
            nn.Linear(C, num_classes),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


# ── Focal CE loss ──────────────────────────────────────────────────────────────

def focal_ce(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0):
    ce = F.cross_entropy(logits, target, reduction='none')
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # ── Data ──────────────────────────────────────────────────────────────────
    sz = args.image_size
    train_tf = T.Compose([
        T.RandomResizedCrop(sz, scale=(0.4, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.ToTensor(),
    ])
    val_tf = T.Compose([
        T.Resize(int(sz * 1.14)),
        T.CenterCrop(sz),
        T.ToTensor(),
    ])

    train_dir = Path(args.data) / 'train'
    val_dir   = Path(args.data) / 'val'
    if not train_dir.exists():
        train_dir = Path(args.data)           # flat ImageFolder at root
        val_dir   = None

    train_ds = ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = ImageFolder(str(val_dir),   transform=val_tf) if (val_dir and val_dir.exists()) else None
    num_classes = args.num_classes or len(train_ds.classes)

    kw = dict(num_workers=args.num_workers, pin_memory=device.type == 'cuda',
              persistent_workers=args.num_workers > 0,
              prefetch_factor=2 if args.num_workers > 0 else None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **kw) \
                   if val_ds else None

    logging.info(f"Dataset: {len(train_ds)} train  "
                 f"{'/ ' + str(len(val_ds)) + ' val' if val_ds else '(no val split)'}  "
                 f"| {num_classes} classes  |  device={device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = EncoderClassifier(num_channels=args.num_channels,
                              num_classes=num_classes).to(device)
    total = sum(p.numel() for p in model.parameters())
    enc   = sum(p.numel() for p in model.encoder.parameters())
    logging.info(f"Params: total={total:,}  encoder={enc:,}  head={total-enc:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_ep = min(5, args.epochs // 10)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, warmup_ep),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - warmup_ep), eta_min=args.lr * 0.01),
        ],
        milestones=[warmup_ep])
    scaler = GradScaler('cuda', enabled=device.type == 'cuda')

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_acc  = 0.0
    start_ep  = 1

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume:
        fp = Path(args.resume)
        if fp.exists():
            ckpt = torch.load(fp, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_ep = ckpt.get('epoch', 0) + 1
            best_acc = ckpt.get('best_val_acc', 0.0)
            logging.info(f"Resumed from {fp}  epoch={start_ep}  best_acc={best_acc:.4f}")

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_ep, args.epochs + 1):
        model.train()
        total_loss = correct = seen = 0
        t0 = time.time()
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for imgs, labels in bar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with autocast('cuda', enabled=device.type == 'cuda'):
                logits = model(imgs)
                loss   = focal_ce(logits, labels, gamma=args.gamma)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            b = imgs.size(0)
            correct     += (logits.argmax(1) == labels).sum().item()
            total_loss  += loss.item() * b
            seen        += b
            bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/seen:.4f}")

        train_acc  = correct / seen
        train_loss = total_loss / seen
        elapsed    = time.time() - t0

        # ── Val ───────────────────────────────────────────────────────────────
        val_acc = 0.0
        if val_loader:
            model.eval()
            vc = vs = 0
            with torch.inference_mode():
                for imgs, labels in tqdm(val_loader, desc="[val]", leave=False):
                    imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    with autocast('cuda', enabled=device.type == 'cuda'):
                        logits = model(imgs)
                    vc += (logits.argmax(1) == labels).sum().item()
                    vs += imgs.size(0)
            val_acc = vc / vs

        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        logging.info(
            f"Epoch {epoch}/{args.epochs}  lr={lr:.2e}  "
            f"train loss={train_loss:.4f} acc={train_acc:.4f}  "
            f"val acc={val_acc:.4f}  ({elapsed:.0f}s)")

        # ── Save ──────────────────────────────────────────────────────────────
        enc_state = {
            'enc1': model.encoder.enc1.state_dict(),
            'enc2': model.encoder.enc2.state_dict(),
            'num_channels': args.num_channels,
            'epoch': epoch,
            'val_acc': val_acc,
        }
        full_ckpt = dict(
            **enc_state,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            best_val_acc=best_acc,
        )
        torch.save(full_ckpt, ckpt_dir / 'enc_last.pt')

        if val_acc > best_acc or (not val_loader and epoch == args.epochs):
            best_acc = val_acc
            torch.save(enc_state, ckpt_dir / 'enc_best.pt')
            logging.info(f"  → new best encoder saved (val_acc={val_acc:.4f})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data',            required=True,
                   help='Root of ImageFolder dataset (must have train/ subdir or be flat)')
    p.add_argument('--num-classes',     type=int, default=None,
                   help='Override number of classes (default: inferred from folder names)')
    p.add_argument('--num-channels',    type=int, default=128,
                   help='Must match num_channels in sf_seg config (default: 128)')
    p.add_argument('--epochs',          type=int, default=50)
    p.add_argument('--batch-size',      type=int, default=256)
    p.add_argument('--lr',              type=float, default=1e-3)
    p.add_argument('--gamma',           type=float, default=2.0,
                   help='Focal loss gamma (default: 2.0)')
    p.add_argument('--image-size',      type=int, default=128,
                   help='Training crop size — match sf_seg image_size (default: 128)')
    p.add_argument('--num-workers',     type=int, default=8)
    p.add_argument('--checkpoint-dir',  default='checkpoints')
    p.add_argument('--resume',          default=None,
                   help='Path to enc_last.pt to resume')
    p.add_argument('--cpu',             action='store_true')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler('checkpoints/pretrain.log'),
                        ])
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
