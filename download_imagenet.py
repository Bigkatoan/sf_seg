"""download_imagenet.py — Download ImageNet-1K via HuggingFace and convert to ImageFolder.

Requires:
    pip install datasets huggingface_hub pillow tqdm
    huggingface-cli login          # one-time, needs HF account with ILSVRC/imagenet-1k access
    # Accept terms at: https://huggingface.co/datasets/ILSVRC/imagenet-1k

Output structure (compatible with torchvision.datasets.ImageFolder):
    <out>/
      train/
        n01440764/   (tench)
          ILSVRC2012_train_00000001.JPEG
          ...
        n01443537/   (goldfish)
          ...
      val/
        n01440764/
          ILSVRC2012_val_00000001.JPEG
          ...

Usage:
    python download_imagenet.py --out /data/imagenet
    python download_imagenet.py --out /data/imagenet --val-only   # skip train (130 GB)
    python download_imagenet.py --out /data/imagenet --split train --workers 8
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


def save_sample(args):
    idx, item, split_dir, class_names = args
    label    = item['label']
    class_id = class_names[label]              # e.g. "n01440764"
    out_dir  = split_dir / class_id
    out_dir.mkdir(parents=True, exist_ok=True)
    fpath = out_dir / f"{split_dir.name}_{idx:08d}.JPEG"
    if not fpath.exists():
        img = item['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(fpath, 'JPEG', quality=95)
    return fpath


def download_split(split: str, out_root: Path, workers: int, max_samples: int | None):
    from datasets import load_dataset

    print(f"\n── Downloading {split} split ──────────────────────────────")
    ds = load_dataset('ILSVRC/imagenet-1k', split=split, trust_remote_code=True,
                      streaming=False)

    # Build class name list (wordnet synset IDs)
    features    = ds.features['label']
    class_names = features.names          # list of 1000 synset IDs ordered by label int

    split_dir = out_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    total  = len(ds) if max_samples is None else min(max_samples, len(ds))
    subset = ds.select(range(total)) if max_samples else ds

    tasks = [(i, item, split_dir, class_names) for i, item in enumerate(subset)]

    saved = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(save_sample, t): t[0] for t in tasks}
        bar  = tqdm(as_completed(futs), total=total,
                    desc=f"{split} → {split_dir}", unit='img')
        for f in bar:
            try:
                f.result()
                saved += 1
            except Exception as e:
                bar.write(f"[warn] sample {futs[f]}: {e}")

    print(f"  {split}: {saved}/{total} images saved → {split_dir}")
    return class_names


def write_class_map(class_names: list[str], out_root: Path):
    """Write synset_id → label_index mapping for reference."""
    import json
    mapping = {name: i for i, name in enumerate(class_names)}
    path = out_root / 'synset_to_idx.json'
    with open(path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Synset map saved → {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out',        default=str(Path.home() / 'data' / 'imagenet'),
                    help='Output directory for ImageFolder dataset '
                         '(default: ~/data/imagenet)')
    ap.add_argument('--split',      default='both',
                    choices=['train', 'val', 'both'],
                    help='Which split to download (default: both)')
    ap.add_argument('--workers',    type=int, default=8,
                    help='Parallel save threads (default: 8)')
    ap.add_argument('--max-train',  type=int, default=None,
                    help='Limit training samples (for quick tests, e.g. 50000)')
    ap.add_argument('--max-val',    type=int, default=None,
                    help='Limit val samples')
    ap.add_argument('--cache-dir',  default=None,
                    help='HuggingFace cache directory (default: ~/.cache/huggingface)')
    args = ap.parse_args()

    # Set HF cache dir if specified
    if args.cache_dir:
        os.environ['HF_DATASETS_CACHE'] = args.cache_dir
        os.environ['HF_HOME']           = args.cache_dir

    out_root = Path(args.out).expanduser()
    try:
        out_root.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"[error] No permission to create {out_root}")
        print(f"  Use a path inside your home dir, e.g.:")
        print(f"    python download_imagenet.py --out ~/data/imagenet")
        return

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("Install dependencies: pip install datasets huggingface_hub pillow tqdm")
        return

    class_names = None
    splits = ['train', 'val'] if args.split == 'both' else [args.split]

    for split in splits:
        max_s = args.max_train if split == 'train' else args.max_val
        class_names = download_split(split, out_root, args.workers, max_s)

    if class_names:
        write_class_map(class_names, out_root)

    print(f"\nDone. Dataset ready at: {out_root}")
    print(f"Run pretraining with:")
    print(f"  python pretrain_encoder.py --data {out_root} --num-classes 1000 "
          f"--num-channels 128 --image-size 128")


if __name__ == '__main__':
    main()
