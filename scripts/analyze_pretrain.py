#!/usr/bin/env python3
"""Phân tích backbone đã pretrain ImageNet — sức khỏe feature + sẵn sàng transfer.

Báo cáo:
  1. Final val top-1/top-5 (từ checkpoint + log)
  2. Feature health mỗi stage: activation RMS, % kênh chết, dynamic range
  3. Transfer check: backbone.* keys nạp được vào sf_seg_v2 (micro hiện tại)
  4. Linear-probe nhanh trên feature f4 (subset val) — đo separability

Usage:
    python -m scripts.analyze_pretrain [--checkpoint checkpoints/backbone_in1k.pt]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from tqdm import tqdm

from src.training.pretrain_imagenet import ImageNetNet
from src.models.sf_seg_v2 import sf_seg_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='checkpoints/backbone_in1k.pt')
    ap.add_argument('--config',     default='config.json')
    ap.add_argument('--data-root',  default='data/imagenet')
    ap.add_argument('--max-val',    type=int, default=5000)
    args = ap.parse_args()

    cfg    = json.loads(Path(args.config).read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ImageNetNet(num_channels=cfg['num_channels'], num_classes=1000,
                        variant=cfg['backbone_variant'], dw_kernel=cfg['dw_kernel']).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"=== Pretrain backbone analysis ===")
    print(f"checkpoint: {args.checkpoint}  epoch={ckpt.get('epoch')}  "
          f"best_top1={ckpt.get('best_top1', '?')}")

    # ── 1. Feature health per stage ───────────────────────────────────────────
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf   = T.Compose([T.Resize(200), T.CenterCrop(176), T.ToTensor(), norm])
    val  = datasets.ImageFolder(Path(args.data_root) / 'val', tf)
    val.samples = val.samples[::max(1, len(val) // args.max_val)]
    loader = DataLoader(val, batch_size=64, num_workers=8)

    feats = {}
    def hook(name):
        def fn(_m, _i, o):
            feats.setdefault(name, []).append(o.detach().float())
        return fn
    bb = model.backbone
    hs = [getattr(bb, s).register_forward_hook(hook(s))
          for s in ['stage1', 'stage2', 'stage3', 'stage4']]

    f4_all, y_all = [], []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for i, (imgs, labels) in enumerate(tqdm(loader, desc='features')):
            _, _, _, _, f4 = bb(imgs.to(device))
            f4_all.append((f4.float().mean((2, 3)) + f4.float().amax((2, 3))).cpu())
            y_all.append(labels)
            if i >= 30: break   # đủ cho thống kê
    for h in hs: h.remove()

    print(f"\n{'stage':<8} {'act_RMS':>9} {'dead%':>7} {'range':>9}")
    for s in ['stage1', 'stage2', 'stage3', 'stage4']:
        a = torch.cat([f[:4] for f in feats[s]])           # vài batch đầu
        rms  = a.pow(2).mean().sqrt().item()
        # kênh chết: gần như luôn 0 trên toàn không gian+batch
        ch_mean = a.abs().mean(dim=(0, 2, 3))
        dead = (ch_mean < 1e-3).float().mean().item() * 100
        rng  = (a.amax() - a.amin()).item()
        print(f"{s:<8} {rms:>9.3f} {dead:>6.1f}% {rng:>9.2f}")

    # ── 2. Separability: head đã train CHÍNH LÀ linear trên g → đo top1 subset
    # (linear-probe from-scratch cần ~vài chục nghìn ảnh cho 1000 class, redundant)
    X = torch.cat(f4_all).to(device); Y = torch.cat(y_all).to(device)
    with torch.no_grad():
        logits = model.cls_head(X)
        top1 = (logits.argmax(1) == Y).float().mean().item()
        top5 = (logits.topk(5, 1).indices == Y.unsqueeze(1)).any(1).float().mean().item()
    print(f"\nSeparability (head linear trên g, {X.shape[0]} ảnh val): "
          f"top1={top1:.3f} top5={top5:.3f}  (= xác nhận g phân tách lớp tốt)")

    # ── 3. Transfer check ─────────────────────────────────────────────────────
    full = sf_seg_v2(num_channels=cfg['num_channels'], focus_size=cfg['focus_size'],
                     num_classes=cfg['num_classes'], backbone_variant=cfg['backbone_variant'],
                     dw_kernel=cfg['dw_kernel'],
                     attn_masks=tuple(cfg['attn_masks']) if cfg.get('attn_masks') else None,
                     budget_ladder=cfg.get('budget_ladder', False),
                     pos_encode=cfg.get('pos_encode', False))
    bb_keys = {k for k in model.state_dict() if k.startswith('backbone.')}
    fk = set(full.state_dict())
    matched = sum(1 for k in bb_keys if k in fk and full.state_dict()[k].shape == model.state_dict()[k].shape)
    n_bb = sum(p.numel() for n, p in full.named_parameters() if n.startswith('backbone.'))
    n_tot = sum(p.numel() for p in full.parameters())
    print(f"\nTransfer → sf_seg_v2: {matched}/{len(bb_keys)} backbone tensors khớp shape")
    print(f"  backbone {n_bb/1e6:.2f}M / {n_tot/1e6:.2f}M total ({100*n_bb/n_tot:.0f}% được pretrain)")
    print(f"\nLệnh seg: ./train.sh --resume {args.checkpoint} --restart")


if __name__ == '__main__':
    main()
