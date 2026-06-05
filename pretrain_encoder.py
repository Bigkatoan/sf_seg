"""pretrain_encoder.py — Pretrain the full sf_seg backbone on ImageNet classification.

Architecture:
    sf_seg.extract_features(x)         →  (B, C//2, H, W)   [backbone — will be kept]
        ↓
    CNN reduction + GAP + MLP          →  (B, num_classes)   [cls head — discarded after]

After training, the backbone state dict (everything except self.masks) is saved.
Load into sf_seg via --encoder-pretrained / "encoder_pretrained" in config.json.

Usage:
    python pretrain_encoder.py --data ~/data/imagenet --num-classes 1000
    python pretrain_encoder.py --data ~/data/imagenet --resume checkpoints/enc_last.pt
"""
from __future__ import annotations

import argparse
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

from src.models.sf_seg import sf_seg


# ── Classification head (discarded after pretraining) ─────────────────────────

class ClsHead(nn.Module):
    """Small CNN + MLP attached to sf_seg.extract_features output (B, C//2, H, W)."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        C = in_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(C, C, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(C, C),
            nn.ReLU(inplace=True),
            nn.Linear(C, num_classes),
        )

    def forward(self, x):
        return self.mlp(self.cnn(x))


class SfSegClassifier(nn.Module):
    def __init__(self, backbone: sf_seg, num_classes: int):
        super().__init__()
        self.backbone = backbone
        C_out = backbone.pre_masks[0].out_channels   # C//2
        self.head = ClsHead(C_out, num_classes)

    def forward(self, x):
        features, _ = self.backbone.extract_features(x)
        return self.head(features)


# ── Loss ──────────────────────────────────────────────────────────────────────

def focal_ce(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0):
    ce = F.cross_entropy(logits, target, reduction='none')
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    sz = args.image_size

    # ── Data ──────────────────────────────────────────────────────────────────
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

    data_root = Path(args.data).expanduser()
    train_dir = data_root / 'train'
    val_dir   = data_root / 'val'
    if not train_dir.exists():
        train_dir = data_root

    train_ds = ImageFolder(str(train_dir), transform=train_tf)
    val_ds   = ImageFolder(str(val_dir), transform=val_tf) if val_dir.exists() else None
    num_classes = args.num_classes or len(train_ds.classes)

    kw = dict(num_workers=args.num_workers, pin_memory=device.type == 'cuda',
              persistent_workers=args.num_workers > 0,
              prefetch_factor=2 if args.num_workers > 0 else None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **kw) \
                   if val_ds else None

    # ── Model ─────────────────────────────────────────────────────────────────
    backbone = sf_seg(num_channels=args.num_channels,
                      focus_size=args.focus_size,
                      num_classes=1)          # num_classes=1 dummy — masks layer unused
    model = SfSegClassifier(backbone, num_classes).to(device)

    backbone_params = sum(p.numel() for p in backbone.parameters())
    head_params     = sum(p.numel() for p in model.head.parameters())
    logging.info(f"Backbone params: {backbone_params:,}  |  "
                 f"Cls head params: {head_params:,}  |  device={device}")
    if device.type == 'cuda':
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup_ep = min(5, args.epochs // 10)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, warmup_ep),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs - warmup_ep),
                eta_min=args.lr * 0.01),
        ],
        milestones=[warmup_ep])
    scaler   = GradScaler('cuda', enabled=device.type == 'cuda')
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    start_ep = 1

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
        tot_loss = correct = seen = 0
        t0  = time.time()
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

            b        = imgs.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            tot_loss += loss.item() * b
            seen     += b
            bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/seen:.4f}")

        train_acc = correct / seen
        elapsed   = time.time() - t0

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
            f"train loss={tot_loss/seen:.4f} acc={train_acc:.4f}  "
            f"val acc={val_acc:.4f}  ({elapsed:.0f}s)")

        # ── Save backbone (masks layer excluded) ──────────────────────────────
        backbone_sd = {k: v for k, v in backbone.state_dict().items()
                       if not k.startswith('masks.')}
        enc_ckpt = {
            'backbone':      backbone_sd,
            'num_channels':  args.num_channels,
            'focus_size':    args.focus_size,
            'epoch':         epoch,
            'val_acc':       val_acc,
        }
        full_ckpt = {
            **enc_ckpt,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc':         best_acc,
        }
        torch.save(full_ckpt, ckpt_dir / 'enc_last.pt')

        if val_acc > best_acc or (not val_loader and epoch == args.epochs):
            best_acc = val_acc
            torch.save(enc_ckpt, ckpt_dir / 'enc_best.pt')
            logging.info(f"  → best backbone saved  val_acc={val_acc:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data',           required=True,
                   help='ImageFolder root (must contain train/ subdir or be flat)')
    p.add_argument('--num-classes',    type=int, default=None,
                   help='Override num classes (default: inferred from folder count)')
    p.add_argument('--num-channels',   type=int, default=128,
                   help='Must match num_channels in config.json (default: 128)')
    p.add_argument('--focus-size',     type=int, default=32,
                   help='Must match focus_size in config.json (default: 32)')
    p.add_argument('--image-size',     type=int, default=128,
                   help='Training crop size — match sf_seg image_size (default: 128)')
    p.add_argument('--epochs',         type=int, default=50)
    p.add_argument('--batch-size',     type=int, default=256)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--gamma',          type=float, default=2.0)
    p.add_argument('--num-workers',    type=int, default=8)
    p.add_argument('--checkpoint-dir', default='checkpoints')
    p.add_argument('--resume',         default=None)
    p.add_argument('--cpu',            action='store_true')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s', datefmt='%H:%M:%S',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(
                                      Path(args.checkpoint_dir) / 'pretrain.log')])
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
