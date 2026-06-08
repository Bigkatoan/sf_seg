"""download_imagenet.py — Download ImageNet-1K via HuggingFace and convert to ImageFolder.

Two-phase approach (robust, resumable):
  Phase 1 — snapshot_download: tải các file parquet thô về local cache,
             có retry + resume tự động, không bị timeout.
  Phase 2 — convert: đọc parquet từ disk, lưu ảnh vào ImageFolder,
             checkpoint theo batch → có thể restart bất cứ lúc nào.

Requires:
    pip install datasets huggingface_hub pillow tqdm pyarrow
    huggingface-cli login
    # Accept terms: https://huggingface.co/datasets/ILSVRC/imagenet-1k

Output (ImageFolder):
    <out>/train/<synset_id>/*.JPEG
    <out>/val/<synset_id>/*.JPEG

Usage:
    python download_imagenet.py --out ~/data/imagenet
    python download_imagenet.py --out ~/data/imagenet --split val
    python download_imagenet.py --out ~/data/imagenet --workers 4
"""
from __future__ import annotations

import argparse
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm

HF_REPO   = 'ILSVRC/imagenet-1k'
REPO_TYPE = 'dataset'


# ── Phase 1: download raw parquet files ───────────────────────────────────────

def phase1_download(cache_dir: Path, splits: list[str]):
    from huggingface_hub import snapshot_download

    # Only download parquet files for the requested splits
    patterns = []
    for s in splits:
        patterns.append(f'data/{s}-*.parquet')

    print(f"\n── Phase 1: downloading parquet files → {cache_dir}")
    print(f"   splits: {splits}  (retry + resume handled automatically)")
    local = snapshot_download(
        repo_id=HF_REPO,
        repo_type=REPO_TYPE,
        local_dir=str(cache_dir),
        allow_patterns=patterns + ['*.json', '*.py'],
        resume_download=True,
        ignore_patterns=['*.arrow', '*.lock'],
    )
    print(f"   parquet files ready at: {local}")
    return Path(local)


# ── Phase 2: convert parquet → ImageFolder ────────────────────────────────────

def _save_one(task):
    i, image, label_id, out_dir, fname = task
    fpath = out_dir / fname
    if fpath.exists():
        return i, True          # already done
    out_dir.mkdir(parents=True, exist_ok=True)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(fpath, 'JPEG', quality=95)
    return i, False             # newly saved


def _load_class_names(cache_dir: Path) -> list[str]:
    """Return list of 1000 synset IDs ordered by label int.

    Tries, in order:
      1. classes.py  (ILSVRC/imagenet-1k has this)
      2. dataset_info.json
      3. label column dictionary in first parquet shard
    """
    import importlib.util

    classes_py = cache_dir / 'classes.py'
    if classes_py.exists():
        spec = importlib.util.spec_from_file_location('_imagenet_classes', classes_py)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return list(mod.IMAGENET2012_CLASSES.keys())

    info_path = cache_dir / 'dataset_info.json'
    if info_path.exists():
        info = json.load(open(info_path))
        return info['features']['label']['names']

    # Last resort: read dictionary encoding from parquet metadata
    import pyarrow.parquet as pq
    shards = sorted(cache_dir.glob('data/train-*.parquet')) or \
             sorted(cache_dir.glob('data/val-*.parquet'))
    if not shards:
        raise RuntimeError("No parquet shards found — run without --skip-download first")
    schema = pq.read_schema(shards[0])
    label_field = schema.field('label')
    if hasattr(label_field.type, 'dictionary'):
        return label_field.type.dictionary.to_pylist()
    raise RuntimeError("Cannot determine class names from cache")


def phase2_convert(cache_dir: Path, out_root: Path,
                   split: str, workers: int, batch_size: int = 500):
    import pyarrow.parquet as pq

    # Load class names (synset IDs ordered by label int)
    class_names = _load_class_names(cache_dir)

    checkpoint_path = out_root / f'.done_{split}.json'
    done_shards: set[str] = set()
    if checkpoint_path.exists():
        done_shards = set(json.load(open(checkpoint_path)))

    shards    = sorted(cache_dir.glob(f'data/{split}-*.parquet'))
    split_dir = out_root / split
    total_saved = 0

    print(f"\n── Phase 2: converting {split} ({len(shards)} shards) → {split_dir}")

    for shard_path in shards:
        shard_name = shard_path.name
        if shard_name in done_shards:
            print(f"   skip  {shard_name} (already done)")
            continue

        table  = pq.read_table(shard_path, columns=['image', 'label'])
        n      = len(table)
        images = table['image'].to_pylist()
        labels = table['label'].to_pylist()

        batch = []
        for row_i, (img_data, label_int) in enumerate(zip(images, labels)):
            label_id = class_names[label_int]
            fname    = f"{shard_name[:-8]}_{row_i:06d}.JPEG"   # strip .parquet
            out_dir  = split_dir / label_id

            # img_data may be bytes or dict {'bytes': ..., 'path': ...}
            raw = img_data if isinstance(img_data, bytes) else img_data.get('bytes', b'')
            try:
                img = Image.open(io.BytesIO(raw))
            except Exception:
                continue
            batch.append((row_i, img, label_id, out_dir, fname))

        # Save batch with thread pool
        newly = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_save_one, t): t[0] for t in batch}
            bar  = tqdm(as_completed(futs), total=n,
                        desc=f"  {shard_name}", unit='img', leave=False)
            for f in bar:
                _, was_done = f.result()
                if not was_done:
                    newly += 1

        total_saved += newly

        # Mark shard done and delete parquet to free space
        done_shards.add(shard_name)
        with open(checkpoint_path, 'w') as fp:
            json.dump(sorted(done_shards), fp)

        shard_size_mb = shard_path.stat().st_size / 1024**2
        shard_path.unlink()
        print(f"   done  {shard_name}  ({newly}/{n} imgs)  deleted {shard_size_mb:.0f} MB")

    # Write class list for reference
    class_map = {name: idx for idx, name in enumerate(class_names)}
    with open(out_root / 'synset_to_idx.json', 'w') as f:
        json.dump(class_map, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out',        default=str(Path.home() / 'data' / 'imagenet'),
                    help='Output ImageFolder directory (default: ~/data/imagenet)')
    ap.add_argument('--cache-dir',  default=str(Path.home() / '.cache' / 'imagenet_hf'),
                    help='HF parquet cache directory (default: ~/.cache/imagenet_hf)')
    ap.add_argument('--split',      default='both',
                    choices=['train', 'val', 'both'],
                    help='Which split to process (default: both)')
    ap.add_argument('--workers',    type=int, default=8,
                    help='Parallel image-save threads (default: 8)')
    ap.add_argument('--batch-size', type=int, default=500,
                    help='Checkpoint every N images (default: 500)')
    ap.add_argument('--skip-download', action='store_true',
                    help='Skip Phase 1 (parquet already in --cache-dir)')
    args = ap.parse_args()

    out_root  = Path(args.out).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()
    splits    = ['train', 'val'] if args.split == 'both' else [args.split]

    try:
        out_root.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        print(f"[error] {e}")
        print("  Use a path inside your home dir, e.g. --out ~/data/imagenet")
        return

    try:
        import datasets, huggingface_hub  # noqa: F401
    except ImportError:
        print("pip install datasets huggingface_hub pillow tqdm pyarrow")
        return

    # Phase 1 — download parquet files
    if not args.skip_download:
        cache_dir = phase1_download(cache_dir, splits)
    else:
        print(f"Skipping download, using cache: {cache_dir}")

    # Phase 2 — convert to ImageFolder
    for split in splits:
        phase2_convert(cache_dir, out_root, split, args.workers, args.batch_size)

    print(f"\nDone → {out_root}")
    print(f"Run pretraining:")
    print(f"  python pretrain_encoder.py --data {out_root} --num-classes 1000 "
          f"--num-channels 128 --image-size 128")


if __name__ == '__main__':
    main()
