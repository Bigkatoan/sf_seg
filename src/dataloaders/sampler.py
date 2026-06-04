"""ClassDiverseBatchSampler — maximise class coverage per batch.

Motivation:
  focal_iou_loss gradient for class c is non-zero ONLY in batches where
  class c appears. With random sampling and batch_size=16 over 20K images,
  a class present in 1% of images (~200 images) appears in only ~0.8 batches
  per epoch on average. Rare classes receive almost no gradient signal.

  This sampler greedily builds each batch so the images collectively cover
  as many distinct classes as possible, giving every class a fair chance
  to contribute to each epoch's gradient.

Usage:
  class_idx   = build_class_index(masks_dir, num_classes,
                                   cache_path=data_root/'class_index.json')
  sampler     = ClassDiverseBatchSampler(class_idx, len(dataset), batch_size)
  loader      = DataLoader(dataset, batch_sampler=sampler, num_workers=N, ...)
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Sampler
from tqdm import tqdm


# ── Index builder ─────────────────────────────────────────────────────────────

def build_class_index(masks_dir: Path, num_classes: int,
                       cache_path: Path | None = None) -> dict[int, list[int]]:
    """Return {class_idx: [sample_idx, ...]} by scanning all mask PNG files.

    Result is cached to cache_path (JSON) so subsequent runs are instant.
    Scanning 20K masks takes ~30s; cached load takes <1s.
    """
    if cache_path and Path(cache_path).exists():
        raw = json.load(open(cache_path))
        return {int(k): v for k, v in raw.items()}

    masks = sorted(Path(masks_dir).glob('*.png'))
    idx: dict[int, list[int]] = defaultdict(list)

    for i, mp in enumerate(tqdm(masks, desc='Building class index', leave=False)):
        for c in np.unique(np.array(Image.open(mp), dtype=np.int32)):
            if 0 <= int(c) < num_classes:
                idx[int(c)].append(i)

    result = {k: v for k, v in idx.items()}
    if cache_path:
        json.dump({str(k): v for k, v in result.items()}, open(cache_path, 'w'))

    n_covered = sum(1 for v in result.values() if v)
    print(f"Class index: {n_covered}/{num_classes} classes have training images  "
          f"(saved → {cache_path})")
    return result


# ── Sampler ───────────────────────────────────────────────────────────────────

class ClassDiverseBatchSampler(Sampler):
    """Yields batches that maximise the number of distinct classes per batch.

    Algorithm (per epoch):
      1. Shuffle each class's image list independently.
      2. For each batch:
         a. Iterate classes in a random order.
         b. For each class, pick the next unused image containing that class.
         c. Stop once batch_size images are collected.
         d. Fill any remaining slots from a global shuffled pool of unused images.
      3. Repeat until all images have been yielded (or dataset is exhausted).

    Complexity: O(batch_size × num_classes) per batch — fast in practice.
    Fills leftover slots in O(1) amortised via a pre-shuffled pointer.
    """

    def __init__(self, class_to_imgs: dict[int, list[int]],
                 n_samples: int, batch_size: int,
                 drop_last: bool = False):
        self.class_to_imgs = {int(c): list(imgs)
                              for c, imgs in class_to_imgs.items() if imgs}
        self.classes   = sorted(self.class_to_imgs.keys())
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.drop_last  = drop_last

        # Quick coverage stat
        covered = len(self.classes)
        print(f"ClassDiverseBatchSampler: {covered} classes  "
              f"batch_size={batch_size}  "
              f"~{n_samples // batch_size} batches/epoch")

    def __iter__(self):
        # Per-epoch: shuffle each class's pool independently
        pools = {c: random.sample(imgs, len(imgs))
                 for c, imgs in self.class_to_imgs.items()}
        ptrs  = {c: 0 for c in self.classes}
        used  = set()

        # Pre-shuffled global filler (O(1) amortised fill for remainder slots)
        filler   = random.sample(range(self.n_samples), self.n_samples)
        fill_ptr = 0

        while len(used) < self.n_samples:
            batch: list[int] = []

            # ── Greedy diversity pass ─────────────────────────────────────────
            for c in random.sample(self.classes, len(self.classes)):
                if len(batch) == self.batch_size:
                    break
                p   = pools[c]
                ptr = ptrs[c]
                # Skip images already used in a previous batch
                while ptr < len(p) and p[ptr] in used:
                    ptr += 1
                ptrs[c] = ptr
                if ptr < len(p):
                    idx = p[ptr]
                    batch.append(idx)
                    used.add(idx)
                    ptrs[c] = ptr + 1

            # ── Fill remaining slots from global filler ────────────────────
            while len(batch) < self.batch_size and fill_ptr < len(filler):
                idx = filler[fill_ptr]; fill_ptr += 1
                if idx not in used:
                    batch.append(idx)
                    used.add(idx)

            if len(batch) == self.batch_size:
                yield batch
            elif batch and not self.drop_last:
                yield batch
            elif not batch:
                break  # exhausted

    def __len__(self) -> int:
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size
