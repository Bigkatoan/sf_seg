#!/usr/bin/env python3
"""Training script — sf_seg on ADE20K-150 (151 classes)."""
from __future__ import annotations

import argparse
import colorsys
import copy
import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision.transforms.functional as TF

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:
    _TB = False

from torch.utils.data import WeightedRandomSampler
from src.losses import sf_loss, SFLossConfig
from src.training.visualize import save_epoch_outputs


# ── Dataset ───────────────────────────────────────────────────────────────────

class ADE20KDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path,
                 image_size: int = 512, augment: bool = False,
                 aug_hflip: bool = True, aug_resized_crop: bool = True,
                 aug_color_jitter: bool = True, aug_cutout: bool = True,
                 aug_shift: bool = True, aug_hue: bool = False):
        self.image_size       = int(image_size)
        self.augment          = augment
        self.aug_hflip        = aug_hflip
        self.aug_resized_crop = aug_resized_crop
        self.aug_color_jitter = aug_color_jitter
        self.aug_cutout       = aug_cutout
        self.aug_shift        = aug_shift
        self.aug_hue          = aug_hue
        self.pairs = [
            (p, masks_dir / (p.stem + '.png'))
            for p in sorted(Path(images_dir).iterdir())
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
            if self.aug_hflip and random.random() > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)

            if self.aug_shift and random.random() > 0.5:
                ms  = int(sz * 0.1)
                dx  = random.randint(-ms, ms)
                dy  = random.randint(-ms, ms)
                pad = [max(0, dx), max(0, dy), max(0, -dx), max(0, -dy)]
                img  = TF.pad(img,  pad, fill=0)
                mask = TF.pad(mask, pad, fill=0)
                img  = TF.crop(img,  max(0, -dy), max(0, -dx), sz, sz)
                mask = TF.crop(mask, max(0, -dy), max(0, -dx), sz, sz)

            if self.aug_resized_crop:
                scale  = random.uniform(0.7, 1.4)
                cw, ch = int(sz / scale), int(sz / scale)
                pw, ph = max(0, cw - sz), max(0, ch - sz)
                if pw > 0 or ph > 0:
                    img  = TF.pad(img,  [pw//2, ph//2, pw-pw//2, ph-ph//2], fill=0)
                    mask = TF.pad(mask, [pw//2, ph//2, pw-pw//2, ph-ph//2], fill=0)
                iw, ih = img.size
                x0 = random.randint(0, max(0, iw - cw))
                y0 = random.randint(0, max(0, ih - ch))
                img  = TF.resized_crop(img,  y0, x0, ch, cw, (sz, sz), Image.BILINEAR)
                mask = TF.resized_crop(mask, y0, x0, ch, cw, (sz, sz), Image.NEAREST)

            img_t = TF.to_tensor(img).float()

            if self.aug_color_jitter:
                if random.random() > 0.5:
                    img_t = TF.adjust_brightness(img_t, random.uniform(0.75, 1.25))
                if random.random() > 0.5:
                    img_t = TF.adjust_contrast(img_t, random.uniform(0.75, 1.25))
                if random.random() > 0.5:
                    img_t = TF.adjust_saturation(img_t, random.uniform(0.75, 1.25))
                if self.aug_hue and random.random() > 0.5:
                    img_t = TF.adjust_hue(img_t, random.uniform(-0.05, 0.05))
                img_t = img_t.clamp(0.0, 1.0)

            if self.aug_cutout and random.random() > 0.5:
                _, H, W = img_t.shape
                rh = random.randint(H // 10, H // 6)
                rw = random.randint(W // 10, W // 6)
                y0 = random.randint(0, H - rh)
                x0 = random.randint(0, W - rw)
                img_t[:, y0:y0+rh, x0:x0+rw] = 0.0

            return img_t, torch.from_numpy(np.array(mask)).long()

        return TF.to_tensor(img), torch.from_numpy(np.array(mask)).long()


# ── Class-aware sampler ───────────────────────────────────────────────────────

def build_class_aware_sampler(dataset: ADE20KDataset,
                               num_classes: int = 151,
                               cache_path: Path | None = None) -> WeightedRandomSampler:
    """WeightedRandomSampler that upsamples images containing rare classes.

    Each image is scored by Σ(1/class_freq) over all classes present in it.
    Images containing rare classes get a high weight and are sampled more often.
    The weights are cached to disk so the scan only runs once.

    Effect: a batch of 16 images covers ~82 unique classes on average vs ~58
    with plain random shuffling — substantially more gradient signal per step
    for the long tail of rare classes.
    """
    if cache_path is not None and cache_path.exists():
        weights = torch.load(cache_path, weights_only=True)
        logging.info(f"Class-aware sampler: loaded weights from {cache_path}  "
                     f"(n={len(weights)}, mean={weights.mean():.2f})")
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    logging.info(f"Class-aware sampler: scanning {len(dataset.pairs)} masks "
                 f"(runs once, cached afterwards)...")

    # ── Pass 1: pixel count per class across entire training set ──────────────
    class_pixel_count = np.zeros(num_classes, dtype=np.int64)
    image_class_sets: list[set[int]] = []

    for _, mask_path in tqdm(dataset.pairs, desc="Scanning masks", leave=False):
        mask = np.array(Image.open(mask_path))
        valid = mask.ravel()
        valid = valid[(valid != 255) & (valid < num_classes)]
        classes, counts = np.unique(valid, return_counts=True)
        for c, n in zip(classes, counts):
            class_pixel_count[c] += n
        image_class_sets.append(set(int(c) for c in classes))

    # ── Per-class inverse frequency ────────────────────────────────────────────
    total = max(class_pixel_count.sum(), 1)
    class_freq    = class_pixel_count / total
    inv_freq      = np.where(class_freq > 0, 1.0 / np.maximum(class_freq, 1e-9), 0.0)

    # Log frequency info
    nonzero_freq = class_freq[class_freq > 0]
    logging.info(f"  pixel freq skew: {nonzero_freq.max()/nonzero_freq.min():.0f}x  "
                 f"  classes with >1% pixels: {(class_freq > 0.01).sum()}")

    # ── Per-image weight = Σ inv_freq over classes present ────────────────────
    raw_w = np.array(
        [sum(inv_freq[c] for c in s) for s in image_class_sets],
        dtype=np.float32,
    )
    # Clip extreme outliers (images with a single ultra-rare class) to 20×mean
    # so they don't dominate every batch.
    clip_val = raw_w.mean() * 20.0
    weights  = np.clip(raw_w, 0, clip_val).astype(np.float32)
    weights /= weights.mean()   # normalize so mean weight ≈ 1

    weights_t = torch.from_numpy(weights)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(weights_t, cache_path)
        logging.info(f"  saved to {cache_path}")

    return WeightedRandomSampler(weights_t, num_samples=len(weights_t), replacement=True)


# ── Metrics ───────────────────────────────────────────────────────────────────

def update_confusion(conf: torch.Tensor, pred: torch.Tensor, target: torch.Tensor):
    C = conf.shape[0]
    p = pred.view(-1).long()
    t = target.view(-1).long()
    valid = (t >= 0) & (t < C)   # excludes ignore pixels (e.g. 255)
    p = p[valid]
    t = t[valid]
    conf += torch.bincount(t * C + p, minlength=C * C).view(C, C)


def miou_from_confusion(conf: torch.Tensor) -> tuple[float, torch.Tensor]:
    inter     = conf.diagonal().float()
    union     = (conf.sum(1) + conf.sum(0) - inter).float()
    per_class = inter / union.clamp(min=1e-8)
    valid     = conf.sum(1) > 0
    return (per_class[valid].mean().item() if valid.any() else 0.0), per_class


# ── Visualisation ─────────────────────────────────────────────────────────────

def _palette(n: int) -> np.ndarray:
    pal = np.zeros((n, 3), dtype=np.uint8)
    for i in range(1, n):
        h = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75 + 0.10 * (i % 3), 0.85 + 0.05 * (i % 2))
        pal[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return pal


def save_sample(model, dataset, device, out_dir: Path, epoch: int, cat_names=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    idx      = random.randrange(len(dataset))
    img_t, mask_t = dataset[idx]
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(img_t.unsqueeze(0).to(device))
    smooth    = F.avg_pool2d(logits.cpu(), 5, 1, 2)
    probs     = torch.softmax(smooth[0], dim=0)
    conf, pred = probs.max(dim=0)
    pred[conf < 0.1] = 0
    pal  = _palette(model.num_classes)
    w, h = img_t.shape[2], img_t.shape[1]
    gt_pil   = Image.fromarray(pal[mask_t.numpy().astype(np.int32)])
    pred_pil = Image.fromarray(pal[pred.numpy().astype(np.int32)])
    img_pil  = TF.to_pil_image(img_t)
    out = Image.new('RGB', (w * 3, h))
    out.paste(img_pil,                              (0,     0))
    out.paste(gt_pil.resize((w, h), Image.NEAREST), (w,     0))
    out.paste(pred_pil.resize((w, h), Image.NEAREST),(w * 2, 0))
    try:
        draw = ImageDraw.Draw(out)
        font = ImageFont.load_default()
        for i, lbl in enumerate(["RGB", "GT", "PRED"]):
            draw.text((i * w + w // 2 - 12, 4), lbl, fill=(255, 255, 255), font=font)
    except Exception:
        pass
    tw = int(out.width * (480 / out.height))
    out.resize((tw, 480), Image.BILINEAR).save(out_dir / f"epoch_{epoch:04d}_{idx}.png")


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    data_root = Path(args.data_root)

    train_ds = ADE20KDataset(
        data_root / 'images' / 'train', data_root / 'masks' / 'train',
        image_size=args.image_size, augment=True,
        aug_hflip=args.aug_hflip, aug_resized_crop=args.aug_resized_crop,
        aug_color_jitter=args.aug_color_jitter,
        aug_cutout=args.aug_cutout, aug_shift=args.aug_shift,
        aug_hue=getattr(args, 'aug_hue', False))
    val_ds = ADE20KDataset(
        data_root / 'images' / 'val', data_root / 'masks' / 'val',
        image_size=args.image_size, augment=False)

    if len(train_ds) == 0:
        raise RuntimeError(f"No training data in {data_root/'images'/'train'}")

    kw = dict(num_workers=args.num_workers, pin_memory=device.type == 'cuda',
              prefetch_factor=2 if args.num_workers > 0 else None,
              persistent_workers=args.num_workers > 0)

    # Class-aware sampler: upsample images with rare classes so each batch
    # covers ~82 unique classes instead of ~58 (random baseline).
    use_cas = getattr(args, 'class_aware_sampler', True)
    if use_cas:
        cache_p = Path(args.log_dir) / 'sampler_weights.pt'
        sampler = build_class_aware_sampler(train_ds, args.num_classes, cache_p)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, **kw)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, **kw)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kw)

    model_type = getattr(args, 'model_type', 'v1')
    if model_type == 'v2':
        from src.models.sf_seg_v2 import sf_seg_v2
        model = sf_seg_v2(
            num_channels=args.num_channels, focus_size=args.focus_size,
            num_classes=args.num_classes,
            backbone_variant=getattr(args, 'backbone_variant', 'micro'),
            dw_kernel=getattr(args, 'dw_kernel', 3),
        ).to(device)
        _bb_ids = {id(p) for p in model.backbone.parameters()}
    else:
        from src.models.sf_seg_r18 import sf_seg
        model = sf_seg(
            num_channels=args.num_channels, focus_size=args.focus_size,
            num_classes=args.num_classes, decoder_type=args.decoder_type,
        ).to(device)
        _bb = {model.r18_stem_conv, model.r18_stem_pool,
               model.r18_layer1, model.r18_layer2, model.r18_layer3, model.r18_layer4}
        _bb_ids = {id(p) for m in _bb for p in m.parameters()}
    param_groups = [
        {'params': [p for p in model.parameters() if id(p) in    _bb_ids], 'lr': args.lr * args.backbone_lr_factor},
        {'params': [p for p in model.parameters() if id(p) not in _bb_ids], 'lr': args.lr},
    ]

    total_p = model.get_num_parameters()
    print(f"Backbone: resnet18  |  params: {total_p:,}  |  num_classes={args.num_classes}"
          f"  |  device={device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    optimizer    = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=1e-4)
    warmup_ep    = min(5, args.epochs // 10)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_ep), eta_min=args.lr * 0.05)
    scheduler    = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_ep),
                    cosine_sched],
        milestones=[warmup_ep])
    scaler = GradScaler('cuda', enabled=device.type == 'cuda')

    # ── Logging ───────────────────────────────────────────────────────────────
    log_dir = Path(args.log_dir);   log_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoint_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, force=True, handlers=[
        logging.FileHandler(log_dir / 'train.log'), logging.StreamHandler()])

    tb = SummaryWriter(log_dir=str(log_dir / 'tensorboard')) if _TB else None
    if not _TB:
        logging.warning("TensorBoard not available — pip install tensorboard")

    cat_names = None
    if (data_root / "cat_to_idx.json").exists():
        with open(data_root / "cat_to_idx.json") as _f:
            cat_names = json.load(_f).get("idx_to_name", {})

    csv_path = log_dir / 'train_log.csv'
    csv_new  = not csv_path.exists()
    csv_f    = open(csv_path, 'a', newline='')
    csv_w    = csv.writer(csv_f)
    if csv_new:
        csv_w.writerow(['epoch',
                        'train_loss','train_seg','train_div','train_edge','train_aux',
                        'train_acc','train_miou',
                        'val_loss','val_seg','val_acc','val_miou'])

    best_miou = 0.0
    best_path = ckpt_dir / 'sf_seg_best.pt'
    last_path = ckpt_dir / 'sf_seg_last.pt'

    # Fixed val batch for TensorBoard visualisation
    _tb_imgs, _tb_masks = None, None
    for b, m in val_loader:
        _tb_imgs, _tb_masks = b[:4].to(device), m[:4].to(device)
        break

    # Fixed val indices for per-epoch visualization (reproducible across epochs)
    _rng         = np.random.default_rng(seed=42)
    _vis_indices = _rng.choice(len(val_ds), size=min(args.vis_samples, len(val_ds)),
                               replace=False).tolist()

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        fp = last_path if str(args.resume).lower() in ('last', 'true', '1') else Path(args.resume)
        if fp.exists():
            try:
                ckpt = torch.load(fp, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
                if not args.restart:
                    try: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    except Exception: pass
                    try: scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                    except Exception: pass
                start_epoch = ckpt.get('epoch', 0) + 1
                best_miou   = ckpt.get('best_val_miou', best_miou)
                mode = 'restarted' if args.restart else 'resumed'
                logging.info(f"Weights {mode} from {fp}, epoch {start_epoch}")
            except Exception as e:
                logging.warning(f"Resume failed: {e}")
        else:
            logging.warning(f"Checkpoint not found: {fp}")

    # ── Epoch loop ────────────────────────────────────────────────────────────
    sf_cfg = SFLossConfig.from_args(args)

    for epoch in range(start_epoch, args.epochs + 1):

        # Progressive resolution
        prog = getattr(args, 'prog_res', None)
        if prog:
            cur_res = args.image_size
            for until, res in sorted(prog, key=lambda x: x[0]):
                if epoch < until:
                    cur_res = res
                    break
            if train_ds.image_size != cur_res:
                train_ds.image_size = cur_res
                if use_cas:
                    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                              sampler=sampler, **kw)
                else:
                    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                              shuffle=True, **kw)
                logging.info(f"[epoch {epoch}] resolution → {cur_res}×{cur_res}")

        # Ramp IoU loss after warm-up epochs
        warm = getattr(args, 'iou_warm_epochs', 0)
        if warm > 0 and epoch <= warm:
            sf_cfg = copy.copy(SFLossConfig.from_args(args))
            sf_cfg.iou_w = 0.0
        else:
            sf_cfg = SFLossConfig.from_args(args)

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        tr   = dict(loss=0., seg=0., div=0., edge=0., aux=0., acc=0.)
        seen = 0
        conf_tr = torch.zeros(args.num_classes, args.num_classes, dtype=torch.long, device=device)

        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for imgs, masks in bar:
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast('cuda', enabled=device.type == 'cuda'):
                logits, _, attn = model(imgs)
                loss, parts = sf_loss(logits, attn, masks, sf_cfg)

                # Deep supervision: auxiliary CE at each head scale
                aux = getattr(model, '_aux', None)
                if aux:
                    aux_ce = logits.new_tensor(0.)
                    for ax in aux:
                        H, W   = ax.shape[2:]
                        tgt_ax = F.interpolate(masks.float().unsqueeze(1), (H, W),
                                               mode='nearest').squeeze(1).long()
                        aux_ce = aux_ce + F.cross_entropy(ax.float(), tgt_ax, ignore_index=255)
                    aux_ce = aux_ce / len(aux)
                    loss   = loss + args.aux_weight * aux_ce
                    parts['aux'] = (args.aux_weight * aux_ce).detach()
                else:
                    parts['aux'] = logits.new_tensor(0.)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            b = imgs.size(0)
            with torch.no_grad():
                pred     = logits.argmax(dim=1)
                valid_px = (masks != 255)
                acc      = ((pred[valid_px] == masks[valid_px]).float().mean().item()
                            if valid_px.any() else 0.0)
                update_confusion(conf_tr, pred, masks)
            for k in ('loss', 'seg', 'div', 'edge', 'aux'):
                tr[k] += (loss if k == 'loss' else parts[k]).item() * b
            tr['acc'] += acc * b
            seen      += b
            bar.set_postfix(loss=f"{loss.item():.4f}", seg=f"{parts['seg'].item():.4f}",
                            aux=f"{parts['aux'].item():.4f}", acc=f"{acc:.4f}")

        tr = {k: v / seen for k, v in tr.items()}
        tr_miou, _ = miou_from_confusion(conf_tr.cpu())

        # ── Val ───────────────────────────────────────────────────────────────
        model.eval()
        vl    = dict(loss=0., seg=0., acc=0.)
        vseen = 0
        conf_vl = torch.zeros(args.num_classes, args.num_classes, dtype=torch.long, device=device)

        with torch.inference_mode():
            for imgs, masks in tqdm(val_loader,
                                    desc=f"Epoch {epoch}/{args.epochs} [val]  ", leave=False):
                imgs  = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with autocast('cuda', enabled=device.type == 'cuda'):
                    logits, _, _ = model(imgs)
                    s = F.cross_entropy(logits.float(), masks.long(), ignore_index=255)
                pred     = logits.argmax(dim=1)
                valid_px = (masks != 255)
                acc      = ((pred[valid_px] == masks[valid_px]).float().mean().item()
                            if valid_px.any() else 0.0)
                update_confusion(conf_vl, pred, masks)
                b = imgs.size(0)
                vl['loss'] += s.item() * b
                vl['seg']  += s.item() * b
                vl['acc']  += acc * b
                vseen      += b

        vl = {k: v / vseen for k, v in vl.items()}
        vl_miou, vl_per = miou_from_confusion(conf_vl.cpu())

        # Top-10 class IoU
        cls_info = ""
        if vl_per is not None and cat_names:
            ranked = sorted(
                [(vl_per[c].item(), c)
                 for c in range(args.num_classes) if conf_vl.sum(1)[c] > 0],
                reverse=True)[:10]
            cls_info = "  " + "  ".join(
                f"{cat_names.get(str(c), str(c))}={v:.3f}" for v, c in ranked)

        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        logging.info(
            f"Ep {epoch}/{args.epochs} | lr={lr:.2e} | "
            f"train loss={tr['loss']:.4f} seg={tr['seg']:.4f} "
            f"div={tr['div']:.4f} aux={tr['aux']:.4f} "
            f"acc={tr['acc']:.4f} mIoU={tr_miou:.4f} | "
            f"val loss={vl['loss']:.4f} acc={vl['acc']:.4f} mIoU={vl_miou:.4f}"
            + cls_info)

        csv_w.writerow([epoch,
                        tr['loss'], tr['seg'], tr['div'], tr['edge'], tr['aux'],
                        tr['acc'], tr_miou,
                        vl['loss'], vl['seg'], vl['acc'], vl_miou])
        csv_f.flush()

        # TensorBoard
        if tb:
            for tag, val in [("Loss/train", tr['loss']), ("Loss/val", vl['loss']),
                              ("mIoU/train", tr_miou),   ("mIoU/val",  vl_miou),
                              ("Acc/train",  tr['acc']),  ("Acc/val",   vl['acc']),
                              ("Seg/train",  tr['seg']),  ("Div/train", tr['div']),
                              ("Edge/train", tr['edge']), ("Aux/train", tr['aux']),
                              ("LR", lr)]:
                tb.add_scalar(tag, val, epoch)
            if epoch == 1 or epoch % 5 == 0:
                _tb_images(tb, model, _tb_imgs, _tb_masks, epoch, args.num_classes, device)

        # Checkpoint
        improved = vl_miou > best_miou
        if improved:
            best_miou = vl_miou
        ckpt = dict(epoch=epoch, model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    best_val_miou=best_miou,
                    num_channels=args.num_channels, focus_size=args.focus_size,
                    num_classes=args.num_classes,
                    model_type=model_type,
                    backbone_variant=getattr(args, 'backbone_variant', 'micro'),
                    dw_kernel=getattr(args, 'dw_kernel', 3))
        torch.save(ckpt, last_path)
        if improved:
            torch.save(ckpt, best_path)
            logging.info(f"  ↳ best saved (val_miou={vl_miou:.4f})")

        if epoch % args.vis_interval == 0 or epoch == 1:
            try:
                ep_dir = save_epoch_outputs(
                    model       = model,
                    val_dataset = val_ds,
                    device      = device,
                    out_root    = out_dir,
                    epoch       = epoch,
                    num_classes = args.num_classes,
                    n_samples   = args.vis_samples,
                    fixed_indices = _vis_indices,
                    cat_names   = cat_names,
                )
                logging.info(f"  ↳ visualizations saved → {ep_dir}")
            except Exception as e:
                logging.warning(f"Visualization failed: {e}")

    csv_f.close()
    if tb:
        tb.close()


def _tb_images(writer, model, imgs, masks, epoch, num_classes, device):
    if writer is None or imgs is None:
        return
    model.eval()
    with torch.no_grad():
        logits, _, attn = model(imgs)
    H, W      = imgs.shape[2:]
    pred_norm = logits.argmax(dim=1).cpu().float().unsqueeze(1) / max(num_classes - 1, 1)
    gt_norm   = masks.cpu().float().unsqueeze(1) / max(num_classes - 1, 1)
    attn_mean = attn.cpu().float().mean(dim=1, keepdim=True)
    attn_mean = F.interpolate(attn_mean, (H, W), mode='bilinear', align_corners=False)
    lo, hi    = attn_mean.amin([2,3], keepdim=True), attn_mean.amax([2,3], keepdim=True)
    attn_norm = (attn_mean - lo) / (hi - lo + 1e-8)
    writer.add_images("Val/image",      imgs.cpu().clamp(0, 1), epoch)
    writer.add_images("Val/gt",         gt_norm,                epoch)
    writer.add_images("Val/pred",       pred_norm,              epoch)
    writer.add_images("Val/attn_large", attn_norm,              epoch)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",               default=None)
    p.add_argument("--data-root",            default=None)
    p.add_argument("--epochs",               type=int,   default=None)
    p.add_argument("--batch-size",           type=int,   default=None)
    p.add_argument("--lr",                   type=float, default=None)
    p.add_argument("--num-workers",          type=int,   default=None)
    p.add_argument("--num-channels",         type=int,   default=None)
    p.add_argument("--focus-size",           type=int,   default=None)
    p.add_argument("--num-classes",          type=int,   default=None)
    p.add_argument("--image-size",           type=int,   default=None)
    p.add_argument("--diversity-weight",     type=float, default=None)
    p.add_argument("--boundary-weight",      type=float, default=None)
    p.add_argument("--aux-weight",           type=float, default=None)
    p.add_argument("--grad-clip",            type=float, default=None)
    p.add_argument("--iou-warm-epochs",      type=int,   default=None)
    p.add_argument("--backbone-lr-factor",   type=float, default=None)
    p.add_argument("--decoder-type",         default=None, choices=["dense"])
    p.add_argument("--resume",               default=None)
    p.add_argument("--restart",              action="store_true", default=None)
    p.add_argument("--log-dir",              default=None)
    p.add_argument("--output-dir",           default=None)
    p.add_argument("--checkpoint-dir",       default=None)
    p.add_argument("--cpu",                  action="store_true")
    p.add_argument("--aug-hflip",            type=lambda x: x.lower() != 'false', default=None)
    p.add_argument("--aug-resized-crop",     type=lambda x: x.lower() != 'false', default=None)
    p.add_argument("--aug-color-jitter",     type=lambda x: x.lower() != 'false', default=None)
    p.add_argument("--aug-cutout",           type=lambda x: x.lower() != 'false', default=None)
    p.add_argument("--aug-shift",            type=lambda x: x.lower() != 'false', default=None)
    p.add_argument("--vis-interval",         type=int,   default=None)
    p.add_argument("--vis-samples",          type=int,   default=None)
    p.add_argument("--model-type",           default=None, choices=["v1", "v2"])
    p.add_argument("--backbone-variant",     default=None, choices=["nano", "micro"])
    return p.parse_args()


def merge_config(args):
    cfg = {}
    if args.config:
        with open(args.config) as _f:
            cfg = json.load(_f)
    elif Path("config.json").exists():
        with open("config.json") as _f:
            cfg = json.load(_f)

    defaults = dict(
        data_root="data", epochs=500, batch_size=8, lr=1e-4, num_workers=8,
        num_channels=64, focus_size=64, num_classes=151,
        image_size=512, decoder_type="dense",
        iou_w=0.5, iou_downsample=4, no_obj_weight=0.1,
        diversity_weight=0.3, edge_weight=0.0,
        attn_guide_weight=0.0, attn_exclusive_weight=0.0,
        aux_weight=0.4,
        grad_clip=5.0, iou_warm_epochs=20,
        backbone_lr_factor=0.1, boundary_weight=3.0,
        prog_res=None,
        log_dir="logs", output_dir="outputs", checkpoint_dir="checkpoints",
        resume=None, restart=False,
        aug_hflip=True, aug_resized_crop=True, aug_color_jitter=True,
        aug_cutout=True, aug_shift=True, aug_hue=False,
        vis_interval=5, vis_samples=6,
        model_type='v1', backbone_variant='micro', dw_kernel=3,
        class_aware_sampler=True,
    )
    for key, default in defaults.items():
        cli = getattr(args, key, None)
        setattr(args, key, cli if cli is not None else cfg.get(key, default))
    return args


if __name__ == "__main__":
    args = merge_config(parse_args())
    print("Config:")
    for k in ["data_root", "epochs", "batch_size", "lr", "num_workers",
              "num_channels", "focus_size", "num_classes", "image_size",
              "backbone_lr_factor", "boundary_weight", "diversity_weight",
              "iou_warm_epochs", "prog_res", "resume"]:
        print(f"  {k}: {getattr(args, k)}")
    train(args)
