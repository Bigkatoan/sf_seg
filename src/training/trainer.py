#!/usr/bin/env python3
"""Training script — sf_seg on ADE20K, standard 150-class protocol."""
from __future__ import annotations

import argparse
import colorsys
import copy
import csv
import json
import logging
import random
import time
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

from src.losses import sf_loss, SFLossConfig
from src.training.visualize import save_epoch_outputs


# ── Dataset ───────────────────────────────────────────────────────────────────

def _remap_ade_mask(mask_arr: np.ndarray) -> np.ndarray:
    """Standard ADE20K protocol: label 0 (unlabeled 'other') → ignore 255,
    classes 1..150 → 0..149. Padded pixels (already 255) stay ignored."""
    return np.where((mask_arr == 0) | (mask_arr == 255), 255, mask_arr - 1)


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
        self.aug_copy_paste = False   # bật qua configure_copy_paste()
        self.cp_prob        = 0.5
        self.cp_rare_classes: list = []

    def configure_copy_paste(self, class_to_images: dict, rare_thresh: int = 300,
                             prob: float = 0.5):
        """Bật copy-paste: dán region class HIẾM (<rare_thresh ảnh) từ ảnh nguồn
        vào ảnh đích → tăng pixel class hiếm, đánh vào đuôi dài kéo sập mIoU."""
        self.cp_rare = {c: imgs for c, imgs in class_to_images.items()
                        if imgs and len(imgs) < rare_thresh}
        self.cp_rare_classes = list(self.cp_rare.keys())
        self.cp_prob        = prob
        self.aug_copy_paste = len(self.cp_rare_classes) > 0
        logging.info(f"Copy-paste bật: {len(self.cp_rare_classes)} class hiếm "
                     f"(<{rare_thresh} ảnh), prob={prob}")

    def __len__(self):
        return len(self.pairs)

    def _copy_paste(self, img_t: torch.Tensor, mask_t: torch.Tensor):
        """Dán 1-2 region class hiếm từ ảnh nguồn (chứa class đó) vào ảnh đích."""
        sz = self.image_size
        for _ in range(random.randint(1, 2)):
            c       = random.choice(self.cp_rare_classes)
            src_idx = random.choice(self.cp_rare[c])
            sip, smp = self.pairs[src_idx]
            si = Image.open(sip).convert('RGB').resize((sz, sz), Image.BILINEAR)
            sm = Image.open(smp).convert('L').resize((sz, sz), Image.NEAREST)
            si_t = TF.to_tensor(si)
            sm_t = torch.from_numpy(_remap_ade_mask(np.array(sm, dtype=np.int64))).long()
            if random.random() > 0.5:                 # hflip nguồn cho đa dạng
                si_t = TF.hflip(si_t); sm_t = TF.hflip(sm_t)
            region = (sm_t == c)                       # pixel của class hiếm
            if region.any():
                img_t[:, region] = si_t[:, region]
                mask_t[region]   = c
        return img_t, mask_t

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
                mask = TF.pad(mask, pad, fill=255)   # padded pixels = ignore, not a class
                img  = TF.crop(img,  max(0, -dy), max(0, -dx), sz, sz)
                mask = TF.crop(mask, max(0, -dy), max(0, -dx), sz, sz)

            if self.aug_resized_crop:
                # Zoom-out (scale<1) pad viền ignore → giới hạn 0.75; zoom-in rộng hơn
                scale  = random.uniform(0.75, 1.7)
                cw, ch = int(sz / scale), int(sz / scale)
                pw, ph = max(0, cw - sz), max(0, ch - sz)
                if pw > 0 or ph > 0:
                    img  = TF.pad(img,  [pw//2, ph//2, pw-pw//2, ph-ph//2], fill=0)
                    mask = TF.pad(mask, [pw//2, ph//2, pw-pw//2, ph-ph//2], fill=255)
                iw, ih = img.size
                x0 = random.randint(0, max(0, iw - cw))
                y0 = random.randint(0, max(0, ih - ch))
                img  = TF.resized_crop(img,  y0, x0, ch, cw, (sz, sz), Image.BILINEAR)
                mask = TF.resized_crop(mask, y0, x0, ch, cw, (sz, sz), Image.NEAREST)

            img_t = TF.to_tensor(img).float()

            if self.aug_color_jitter:
                if random.random() > 0.5:
                    img_t = TF.adjust_brightness(img_t, random.uniform(0.6, 1.4))
                if random.random() > 0.5:
                    img_t = TF.adjust_contrast(img_t, random.uniform(0.6, 1.4))
                if random.random() > 0.5:
                    img_t = TF.adjust_saturation(img_t, random.uniform(0.6, 1.4))
                if self.aug_hue and random.random() > 0.5:
                    img_t = TF.adjust_hue(img_t, random.uniform(-0.05, 0.05))
                img_t = img_t.clamp(0.0, 1.0)

            mask_t = torch.from_numpy(_remap_ade_mask(np.array(mask, dtype=np.int64))).long()

            # Copy-paste class hiếm (trước cutout để cutout có thể che cả vùng dán)
            if self.aug_copy_paste and random.random() < self.cp_prob:
                img_t, mask_t = self._copy_paste(img_t, mask_t)

            # Cutout MASK-AWARE: vùng cut → ignore 255 (không phạt presence/seg
            # trên pixel đã bị che) → bật lại an toàn. 1-3 hố, to hơn cũ.
            if self.aug_cutout:
                _, H, W = img_t.shape
                for _ in range(random.randint(1, 2)):
                    if random.random() > 0.5:
                        continue
                    rh = random.randint(H // 8, H // 5)
                    rw = random.randint(W // 8, W // 5)
                    y0 = random.randint(0, H - rh)
                    x0 = random.randint(0, W - rw)
                    img_t[:, y0:y0+rh, x0:x0+rw] = 0.0
                    mask_t[y0:y0+rh, x0:x0+rw]   = 255

            return img_t, mask_t

        return TF.to_tensor(img), torch.from_numpy(
            _remap_ade_mask(np.array(mask, dtype=np.int64))).long()


# ── Class-aware sampler ───────────────────────────────────────────────────────

def _build_class_index(dataset: ADE20KDataset,
                        num_classes: int,
                        cache_path: Path | None) -> tuple[dict, dict]:
    """Return (class_to_images, img_classes). Scans masks once, then caches."""
    if cache_path is not None and cache_path.exists():
        d = torch.load(cache_path, weights_only=False)
        logging.info(f"Class index loaded from {cache_path}  "
                     f"({len(d['img_classes'])} images, {num_classes} classes)")
        return d['class_to_images'], d['img_classes']

    logging.info(f"Building class index — scanning {len(dataset.pairs)} masks "
                 f"(runs once, result cached)...")
    class_to_images: dict[int, list[int]] = {c: [] for c in range(num_classes)}
    img_classes:     dict[int, set[int]]  = {}

    for idx, (_, mask_path) in enumerate(
            tqdm(dataset.pairs, desc="Scanning masks", leave=False)):
        mask  = _remap_ade_mask(np.array(Image.open(mask_path), dtype=np.int64))
        valid = mask.ravel()
        classes = np.unique(valid[(valid != 255) & (valid < num_classes)])
        for c in classes:
            class_to_images[c].append(idx)
        img_classes[idx] = set(int(c) for c in classes)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'class_to_images': class_to_images, 'img_classes': img_classes},
                   cache_path)
        logging.info(f"  saved to {cache_path}")

    return class_to_images, img_classes


class AllClassBatchSampler(torch.utils.data.Sampler):
    """Builds every optimizer step so that ALL num_classes appear at least once,
    while spreading usage evenly across the dataset.

    Vấn đề của greedy thuần: cùng ~29 hub images được chọn lại MỌI step
    (max usage = 481/epoch, chỉ 34% dataset được dùng) → model overfit vào
    hub images, val mIoU plateau trong khi train mIoU tăng.

    Algorithm (per optimizer step = batch_size × accum_steps ảnh)
    -------------------------------------------------------------
    Phase A — rotation cho class hiếm (< rare_thresh ảnh):
      Mỗi class hiếm round-robin qua danh sách ảnh đã shuffle của nó →
      usage hoàn toàn đều (ảnh của class 41-img được dùng ~9 lần/epoch
      thay vì 295). Duyệt từ hiếm nhất; class đã covered do co-occurrence
      thì bỏ qua.

    Phase B — greedy set cover + usage tie-break cho class còn lại:
      score[i] = số class chưa cover trong ảnh i. Trong các ảnh có score
      >= max·(1-rel_margin) (floor: >= 1 class mới), chọn ảnh ÍT DÙNG nhất
      → vẫn tối ưu cover, nhưng xoay vòng giữa các ảnh tương đương.

    Phase C — fill từ global permutation stream:
      Phần còn lại lấy tuần tự từ 1 permutation của toàn dataset (không
      lặp trong epoch) → phần dataset không chứa class hiếm vẫn được train.
    """

    def __init__(self, class_to_images: dict[int, list[int]],
                 img_classes: dict[int, set[int]],
                 dataset_size: int, batch_size: int,
                 accum_steps: int = 1, seed: int = 42,
                 rare_thresh: int = 200, rel_margin: float = 0.4):
        self.n           = dataset_size
        self.batch_size  = batch_size
        self.accum_steps = accum_steps
        self.seed        = seed
        self.epoch       = 0
        self.rel_margin  = rel_margin
        self.all_classes = sorted(c for c, imgs in class_to_images.items() if imgs)
        C    = len(self.all_classes)
        cmap = {c: j for j, c in enumerate(self.all_classes)}

        # Mỗi optimizer step = accum_steps mini-batches = batch_size × accum_steps ảnh
        eff_bs           = batch_size * accum_steps
        opt_steps        = max(1, dataset_size // eff_bs)
        self.num_batches = opt_steps * accum_steps   # tổng mini-batches mỗi epoch

        # M[i, c] = True nếu image i chứa class c  (N×C bool, ~3 MB)
        self.M = np.zeros((dataset_size, C), dtype=bool)
        for c, imgs in class_to_images.items():
            if imgs:
                self.M[imgs, cmap[c]] = True

        # Class hiếm (ít ảnh) — sắp từ hiếm nhất, rotation riêng từng class
        self.rare_classes = sorted(
            (c for c in self.all_classes if len(class_to_images[c]) < rare_thresh),
            key=lambda c: len(class_to_images[c]))
        self.rare_col   = {c: cmap[c] for c in self.rare_classes}
        self._rare_imgs = {c: np.asarray(class_to_images[c]) for c in self.rare_classes}

        min_imgs = min(len(v) for v in class_to_images.values() if v)
        logging.info(
            f"AllClassBatchSampler: {C} classes  "
            f"batch_size={batch_size}  accum_steps={accum_steps}  "
            f"effective_bs={eff_bs}  num_batches={self.num_batches}  "
            f"rare_classes={len(self.rare_classes)} (<{rare_thresh} imgs)  "
            f"rarest_class={min_imgs} imgs"
        )

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng     = np.random.default_rng(self.seed + self.epoch)
        N       = self.n
        C       = len(self.all_classes)
        accum   = self.accum_steps
        eff_bs  = self.batch_size * accum
        opt_steps = self.num_batches // accum

        usage  = np.zeros(N, dtype=np.float32)            # số lần dùng trong epoch
        rot    = {c: rng.permutation(v) for c, v in self._rare_imgs.items()}
        ptr    = {c: 0 for c in self.rare_classes}
        stream = rng.permutation(N)                       # fill stream toàn epoch
        sp     = 0

        for _ in range(opt_steps):
            all_imgs = []
            in_set   = np.zeros(N, dtype=bool)
            covered  = np.zeros(C, dtype=bool)

            # ── Phase A: rotation class hiếm — usage đều tuyệt đối ──────────
            for c in self.rare_classes:
                if covered[self.rare_col[c]] or len(all_imgs) >= eff_bs:
                    continue
                lst = rot[c]
                pick = None
                for _ in range(len(lst)):
                    i = int(lst[ptr[c] % len(lst)])
                    ptr[c] += 1
                    if not in_set[i]:
                        pick = i
                        break
                if pick is None:
                    continue
                all_imgs.append(pick)
                in_set[pick] = True
                covered |= self.M[pick]

            # ── Phase B: greedy + usage tie-break cho class còn lại ─────────
            scores = self.M[:, ~covered].sum(axis=1).astype(np.float32)
            while not covered.all() and len(all_imgs) < eff_bs:
                s = scores.copy()
                s[in_set] = -1.0
                mx = s.max()
                if mx < 1.0:
                    break   # không còn ảnh nào cover class mới
                # Trong các ảnh gần-tối-ưu (floor: cover >= 1 class mới),
                # chọn ảnh ít dùng nhất trong epoch
                cand = np.flatnonzero(s >= max(mx * (1.0 - self.rel_margin), 1.0))
                u    = usage[cand] + rng.uniform(0.0, 1.0, len(cand))
                best = int(cand[u.argmin()])
                all_imgs.append(best)
                in_set[best] = True
                newly = self.M[best] & ~covered
                if newly.any():
                    scores -= self.M[:, newly].sum(axis=1)
                    scores  = np.maximum(scores, 0.0)
                    covered |= self.M[best]

            # ── Phase C: fill từ stream — mỗi ảnh tối đa 1 lần fill/epoch ───
            while len(all_imgs) < eff_bs:
                i = int(stream[sp % N])
                sp += 1
                if not in_set[i]:
                    all_imgs.append(i)
                    in_set[i] = True

            usage[all_imgs] += 1.0

            # Shuffle rồi chia thành accum mini-batches, yield từng cái
            rng.shuffle(all_imgs)
            bs = self.batch_size
            for a in range(accum):
                yield all_imgs[a * bs : (a + 1) * bs]


def build_class_aware_sampler(dataset: ADE20KDataset,
                               num_classes: int = 151,
                               cache_path: Path | None = None,
                               batch_size: int = 14,
                               accum_steps: int = 3) -> AllClassBatchSampler:
    """Build AllClassBatchSampler from cached class index."""
    c2i, img_cls = _build_class_index(dataset, num_classes, cache_path)
    return AllClassBatchSampler(c2i, img_cls, len(dataset), batch_size, accum_steps)


# ── Metrics ───────────────────────────────────────────────────────────────────

def presence_target(masks: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(B,H,W) labels → (B,C) multi-label 0/1: class c có mặt trong ảnh."""
    B     = masks.shape[0]
    valid = (masks != 255)
    idx   = torch.where(valid, masks, torch.zeros_like(masks))
    tgt   = torch.zeros(B, num_classes, device=masks.device)
    tgt.scatter_add_(1, idx.view(B, -1), valid.view(B, -1).float())
    return tgt.clamp_(max=1.0)


def update_presence_counts(cnt: dict, pres_logits: torch.Tensor,
                            tgt: torch.Tensor, thresh: float = 0.5):
    """Accumulate micro tp/fp/fn của presence prediction (sigmoid > thresh)."""
    with torch.no_grad():
        pb = (pres_logits.sigmoid() > thresh).float()
        cnt['tp'] += (pb * tgt).sum().item()
        cnt['fp'] += (pb * (1.0 - tgt)).sum().item()
        cnt['fn'] += ((1.0 - pb) * tgt).sum().item()


def presence_f1(cnt: dict) -> float:
    denom = 2 * cnt['tp'] + cnt['fp'] + cnt['fn']
    return (2 * cnt['tp'] / denom) if denom > 0 else 0.0


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

    # Class-aware batch sampler: every batch guaranteed to contain all 151
    # classes via greedy set cover (needs batch_size ≥ 29; min cover = 29 imgs).
    use_cas = getattr(args, 'class_aware_sampler', True)
    accum_steps = getattr(args, 'accum_steps', 1)
    cache_p = Path(args.log_dir) / 'class_index.pt'
    if getattr(args, 'aug_copy_paste', False):
        _c2i, _ = _build_class_index(train_ds, args.num_classes, cache_p)
        train_ds.configure_copy_paste(_c2i, getattr(args, 'cp_rare_thresh', 300),
                                      getattr(args, 'cp_prob', 0.5))
    if use_cas:
        _sampler = build_class_aware_sampler(
            train_ds, args.num_classes, cache_p, args.batch_size, accum_steps)
        # batch_sampler takes over — cannot combine with batch_size/shuffle
        train_loader = DataLoader(train_ds, batch_sampler=_sampler, **kw)
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
            decoder_dim=getattr(args, 'decoder_dim', 256),
            hr_dim=getattr(args, 'hr_dim', 96),
            grad_checkpoint=getattr(args, 'grad_checkpoint', True),
            attn_masks=(tuple(args.attn_masks) if getattr(args, 'attn_masks', None) else None),
            budget_ladder=getattr(args, 'budget_ladder', False),
            pos_encode=getattr(args, 'pos_encode', False),
            dropout=getattr(args, 'dropout', 0.0),
            enable_ensemble=getattr(args, 'enable_ensemble', False),
            attn_temperature=getattr(args, 'attn_temperature', 1.0),
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
    _bb_name = 'SFBackbone-' + getattr(args, 'backbone_variant', 'micro') \
               if model_type == 'v2' else 'resnet18'
    print(f"Backbone: {_bb_name}  |  params: {total_p:,}  |  num_classes={args.num_classes}"
          f"  |  device={device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=1e-4)
    warmup_ep = min(5, args.epochs // 10)
    if getattr(args, 'lr_schedule', 'cosine') == 'constant':
        # Constant lr (chỉ giữ warmup ngắn): không decay — backbone factor 0.1
        # từng đóng băng backbone vì decay chồng lên factor
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_ep)
    else:
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - warmup_ep), eta_min=args.lr * 0.05)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
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
            _raw = json.load(_f).get("idx_to_name", {})
        # File indexes raw ADE labels (0=background, 1=wall...); model classes
        # follow the 150-class protocol (0=wall) → shift by -1, drop background.
        cat_names = {str(int(k) - 1): v for k, v in _raw.items() if int(k) >= 1}

    CSV_HEADER = ['epoch',
                  'train_loss','train_seg','train_div','train_edge','train_aux',
                  'train_acc','train_miou','train_pres_f1',
                  'val_loss','val_seg','val_acc','val_miou','val_pres_f1']
    csv_path = log_dir / 'train_log.csv'
    # Header cũ khác (run đời trước) → rotate sang file backup, mở file mới
    if csv_path.exists():
        with open(csv_path) as _f:
            _old_header = _f.readline().strip().split(',')
        if _old_header != CSV_HEADER:
            _bak = csv_path.with_name(
                f"train_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            csv_path.rename(_bak)
            logging.info(f"CSV schema đổi — log cũ chuyển sang {_bak.name}")
    csv_new  = not csv_path.exists()
    csv_f    = open(csv_path, 'a', newline='')
    csv_w    = csv.writer(csv_f)
    if csv_new:
        csv_w.writerow(CSV_HEADER)

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
                _missing, _unexpected = model.load_state_dict(
                    ckpt['model_state_dict'], strict=False)
                if _missing:
                    logging.info(f"  new layers (fresh init): {_missing}")
                if _unexpected:
                    logging.warning(f"  unexpected keys ignored: {_unexpected}")
                if args.restart:
                    # Restart: chỉ lấy weights — epoch, optimizer, scheduler,
                    # best_miou đều fresh (ckpt nguồn có thể từ pretrain stage-1)
                    start_epoch = 1
                else:
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
                    train_loader = DataLoader(train_ds, batch_sampler=_sampler, **kw)
                else:
                    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                              shuffle=True, **kw)
                logging.info(f"[epoch {epoch}] resolution → {cur_res}×{cur_res}")

        # Ramp IoU loss after warm-up epochs — CHỈ khi có CE (focal_w>0) gánh
        # segmentation lúc warmup. Nếu focal_w=0 (IoU-only tuning), tắt IoU sẽ
        # để seg loss = 0 hoàn toàn → segmentation head không có gradient → sập.
        warm = getattr(args, 'iou_warm_epochs', 0)
        if warm > 0 and epoch <= warm and getattr(args, 'focal_w', 1.0) > 0:
            sf_cfg = copy.copy(SFLossConfig.from_args(args))
            sf_cfg.iou_w = 0.0
        else:
            sf_cfg = SFLossConfig.from_args(args)

        # ── Train ─────────────────────────────────────────────────────────────
        if use_cas:
            _sampler.set_epoch(epoch)   # re-seed so class order differs each epoch
        model.train()
        tr   = dict(loss=0., seg=0., div=0., edge=0., aux=0., acc=0.)
        seen = 0
        conf_tr = torch.zeros(args.num_classes, args.num_classes, dtype=torch.long, device=device)
        pres_tr = dict(tp=0., fp=0., fn=0.)   # presence micro-F1 counters

        bar        = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        step_loss  = 0.0   # accumulated loss for display
        step_parts: dict = {}
        micro_step = 0     # counts mini-batches within current optimizer step

        optimizer.zero_grad(set_to_none=True)

        for imgs, masks in bar:
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast('cuda', enabled=device.type == 'cuda'):
                if model_type == 'v2':
                    # Loss tại H/2: nhanh 1.5×, VRAM −7.7GB; upsample cuối là
                    # bilinear không tham số nên không mất tín hiệu học
                    logits, _, attn = model(imgs, upsample=False)
                    masks = F.interpolate(masks.float().unsqueeze(1), logits.shape[2:],
                                          mode='nearest').squeeze(1).long()
                else:
                    logits, _, attn = model(imgs)
                loss, parts = sf_loss(logits, attn, masks, sf_cfg)

                # Deep supervision: auxiliary CE at each head scale (skip nếu weight=0)
                aux = getattr(model, '_aux', None)
                if aux and args.aux_weight > 0:
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

                # Class-presence guide: BCE multi-label "class nào có trong ảnh"
                # — supervise global vector g (label free từ mask). Logged in 'aux'.
                pres = getattr(model, '_presence', None)
                pw   = getattr(args, 'presence_weight', 0.0)
                if pres is not None and pw > 0:
                    pres_tgt = presence_target(masks, pres.shape[1])
                    # pos_weight: ~10 class có mặt / 140 vắng — BCE thuần bias về
                    # "vắng" (đo được: precision 0.67, recall 0.22). Đẩy recall lên.
                    ppw = getattr(args, 'presence_pos_weight', 1.0)
                    p_loss = F.binary_cross_entropy_with_logits(
                        pres.float(), pres_tgt,
                        pos_weight=pres.new_full((pres.shape[1],), ppw))
                    loss   = loss + pw * p_loss
                    parts['aux'] = parts['aux'] + (pw * p_loss).detach()
                    update_presence_counts(pres_tr, pres, pres_tgt)

                # Anti-collapse: diversity loss trên sparse attention heads
                adiv = getattr(model, '_attn_div', None)
                adw  = getattr(args, 'attn_div_weight', 0.0)
                if adiv is not None and adw > 0:
                    loss = loss + adw * adiv
                    parts['div'] = parts['div'] + (adw * adiv).detach()

                # Per-mask region-gated CE: mỗi sparse mask là weak predictor,
                # supervise CHỈ ở vùng nó attend (CE_pixel × gate / Σgate). Mask
                # chuyên một vùng → ensemble. Logged in 'aux'.
                pmp = getattr(model, '_per_mask_preds', None)
                msw = getattr(args, 'mask_sup_weight', 0.0)
                if pmp is not None and msw > 0:
                    m_loss = logits.new_tensor(0.); nmask = 0
                    for pred, gate in pmp:            # pred (B,M,C,h,w), gate (B,M,h,w)
                        B_, M_, C_, h_, w_ = pred.shape
                        tgt = F.interpolate(masks.float().unsqueeze(1), (h_, w_),
                                            mode='nearest').squeeze(1).long()   # (B,h,w)
                        tgt = tgt.unsqueeze(1).expand(B_, M_, h_, w_).reshape(-1, h_, w_)
                        ce  = F.cross_entropy(pred.reshape(-1, C_, h_, w_).float(), tgt,
                                              ignore_index=255, reduction='none')  # (B*M,h,w)
                        g   = gate.reshape(-1, h_, w_).float()
                        m_loss = m_loss + (ce * g).sum() / (g.sum().clamp(min=1.0))
                        nmask += 1
                    m_loss = m_loss / max(nmask, 1)
                    loss = loss + msw * m_loss
                    parts['aux'] = parts['aux'] + (msw * m_loss).detach()

            if not torch.isfinite(loss):
                micro_step = 0
                optimizer.zero_grad(set_to_none=True)
                continue

            # Scale loss for accumulation so sum ≡ mean over accum_steps
            scaler.scale(loss / accum_steps).backward()
            micro_step += 1

            # Metrics — track every mini-batch regardless of accumulation
            b = imgs.size(0)
            with torch.no_grad():
                pred     = logits.argmax(dim=1)
                valid_px = (masks != 255)
                acc      = ((pred[valid_px] == masks[valid_px]).float().mean().item()
                            if valid_px.any() else 0.0)
                update_confusion(conf_tr, pred, masks)
            step_loss += loss.item()
            for k, v in parts.items():
                step_parts[k] = step_parts.get(k, 0.0) + v.item()
            tr['acc'] += acc * b
            seen      += b

            # ── Optimizer step every accum_steps mini-batches ──────────────
            if micro_step == accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # Log averaged over the accumulated steps
                disp_loss = step_loss / accum_steps
                disp_seg  = step_parts.get('seg', 0.0) / accum_steps
                disp_aux  = step_parts.get('aux', 0.0) / accum_steps
                bar.set_postfix(loss=f"{disp_loss:.4f}", seg=f"{disp_seg:.4f}",
                                aux=f"{disp_aux:.4f}", acc=f"{acc:.4f}")

                tr['loss'] += step_loss * b
                for k in ('seg', 'div', 'edge', 'aux'):
                    tr[k] += step_parts.get(k, 0.0) * b

                step_loss  = 0.0
                step_parts = {}
                micro_step = 0

        tr = {k: v / seen for k, v in tr.items()}
        tr_miou, _ = miou_from_confusion(conf_tr.cpu())

        # ── Val ───────────────────────────────────────────────────────────────
        model.eval()
        vl    = dict(loss=0., seg=0., acc=0.)
        vseen = 0
        conf_vl = torch.zeros(args.num_classes, args.num_classes, dtype=torch.long, device=device)
        pres_vl = dict(tp=0., fp=0., fn=0.)

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
                _pres = getattr(model, '_presence', None)
                if _pres is not None:
                    update_presence_counts(
                        pres_vl, _pres, presence_target(masks, _pres.shape[1]))
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

        tr_pf1 = presence_f1(pres_tr)
        vl_pf1 = presence_f1(pres_vl)

        logging.info(
            f"Ep {epoch}/{args.epochs} | lr={lr:.2e} | "
            f"train loss={tr['loss']:.4f} seg={tr['seg']:.4f} "
            f"div={tr['div']:.4f} aux={tr['aux']:.4f} "
            f"acc={tr['acc']:.4f} mIoU={tr_miou:.4f} presF1={tr_pf1:.3f} | "
            f"val loss={vl['loss']:.4f} acc={vl['acc']:.4f} mIoU={vl_miou:.4f} "
            f"presF1={vl_pf1:.3f}"
            + cls_info)

        csv_w.writerow([epoch,
                        tr['loss'], tr['seg'], tr['div'], tr['edge'], tr['aux'],
                        tr['acc'], tr_miou, tr_pf1,
                        vl['loss'], vl['seg'], vl['acc'], vl_miou, vl_pf1])
        csv_f.flush()

        # TensorBoard
        if tb:
            for tag, val in [("Loss/train", tr['loss']), ("Loss/val", vl['loss']),
                              ("mIoU/train", tr_miou),   ("mIoU/val",  vl_miou),
                              ("Acc/train",  tr['acc']),  ("Acc/val",   vl['acc']),
                              ("Seg/train",  tr['seg']),  ("Div/train", tr['div']),
                              ("Edge/train", tr['edge']), ("Aux/train", tr['aux']),
                              ("PresenceF1/train", tr_pf1), ("PresenceF1/val", vl_pf1),
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
                    dw_kernel=getattr(args, 'dw_kernel', 3),
                    # Runtime-behavior flags (không có params) — phải lưu để
                    # probe/eval dựng lại đúng, nếu không model chạy sai lặng lẽ
                    decoder_dim=getattr(args, 'decoder_dim', 256),
                    hr_dim=getattr(args, 'hr_dim', 96),
                    attn_masks=getattr(args, 'attn_masks', None),
                    budget_ladder=getattr(args, 'budget_ladder', False),
                    pos_encode=getattr(args, 'pos_encode', False),
                    enable_ensemble=getattr(args, 'enable_ensemble', False),
                    attn_temperature=getattr(args, 'attn_temperature', 1.0))
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
    p.add_argument("--accum-steps",          type=int,   default=None)
    p.add_argument("--lr",                   type=float, default=None)
    p.add_argument("--num-workers",          type=int,   default=None)
    p.add_argument("--num-channels",         type=int,   default=None)
    p.add_argument("--focus-size",           type=int,   default=None)
    p.add_argument("--num-classes",          type=int,   default=None)
    p.add_argument("--image-size",           type=int,   default=None)
    p.add_argument("--diversity-weight",     type=float, default=None)
    p.add_argument("--boundary-weight",      type=float, default=None)
    p.add_argument("--aux-weight",           type=float, default=None)
    p.add_argument("--focal-w",              type=float, default=None)
    p.add_argument("--iou-w",                type=float, default=None)
    p.add_argument("--iou-form",             choices=['linear', 'log'], default=None)
    p.add_argument("--presence-weight",      type=float, default=None)
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
        focal_w=1.0, iou_w=0.5, iou_downsample=4, iou_form='linear', no_obj_weight=0.1,
        diversity_weight=0.3, edge_weight=0.0,
        attn_guide_weight=0.0, attn_exclusive_weight=0.0,
        aux_weight=0.4, presence_weight=0.2, presence_pos_weight=4.0,
        decoder_dim=256, hr_dim=96, grad_checkpoint=True, lr_schedule='cosine',
        attn_masks=None, budget_ladder=False, pos_encode=False, dropout=0.0,
        attn_div_weight=0.5, aug_copy_paste=False, cp_rare_thresh=300, cp_prob=0.5,
        enable_ensemble=False, attn_temperature=1.0, mask_sup_weight=0.3,
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
        accum_steps=1,
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
