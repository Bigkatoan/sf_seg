"""pack_webdataset.py — Convert ImageFolder → WebDataset tar shards.

Sequential tar reads are 5-10x faster than random JPEG file access,
especially on datasets with 1M+ small files.

Usage:
    python pack_webdataset.py --data ~/data/imagenet --out ~/data/imagenet_wds
    python pack_webdataset.py --data ~/data/imagenet --out ~/data/imagenet_wds --split val

Output:
    ~/data/imagenet_wds/
      train-000000.tar   (~1000 images each, ~130 MB)
      train-000001.tar
      ...
      val-000000.tar
      ...
      classes.json       {class_name: label_int}
"""
from __future__ import annotations

import argparse
import json
import random
import tarfile
import io
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def pack_split(split_dir: Path, out_dir: Path, split: str,
               shard_size: int, image_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all (path, label_int) pairs
    classes   = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    pairs: list[tuple[Path, int]] = []
    for cls_name, label in cls_to_idx.items():
        for img_path in (split_dir / cls_name).iterdir():
            if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                pairs.append((img_path, label))

    random.shuffle(pairs)
    print(f"{split}: {len(pairs)} images → {len(classes)} classes → "
          f"{(len(pairs) + shard_size - 1) // shard_size} shards")

    shard_idx = 0
    shard_tar: tarfile.TarFile | None = None
    count_in_shard = 0

    def open_shard():
        nonlocal shard_tar, shard_idx, count_in_shard
        if shard_tar:
            shard_tar.close()
        name = out_dir / f"{split}-{shard_idx:06d}.tar"
        shard_tar = tarfile.open(name, 'w')
        shard_idx += 1
        count_in_shard = 0

    open_shard()
    bar = tqdm(pairs, desc=f"packing {split}", unit='img')

    for i, (img_path, label) in enumerate(bar):
        # Resize and re-encode as JPEG in memory
        try:
            img = Image.open(img_path).convert('RGB')
            if image_size:
                img = img.resize((image_size, image_size), Image.BILINEAR)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=90)
            jpg_bytes = buf.getvalue()
        except Exception as e:
            bar.write(f"skip {img_path}: {e}")
            continue

        key = f"{i:08d}"

        # Add image bytes
        info      = tarfile.TarInfo(name=f"{key}.jpg")
        info.size = len(jpg_bytes)
        shard_tar.addfile(info, io.BytesIO(jpg_bytes))

        # Add label as text
        lbl_bytes = str(label).encode()
        info2      = tarfile.TarInfo(name=f"{key}.cls")
        info2.size = len(lbl_bytes)
        shard_tar.addfile(info2, io.BytesIO(lbl_bytes))

        count_in_shard += 1
        if count_in_shard >= shard_size:
            open_shard()

    if shard_tar:
        shard_tar.close()

    # Save class map
    json.dump(cls_to_idx, open(out_dir / 'classes.json', 'w'), indent=2)
    print(f"Done: {shard_idx} shards → {out_dir}")
    return shard_idx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data',       required=True,
                    help='ImageFolder root (contains train/ val/)')
    ap.add_argument('--out',        default=None,
                    help='Output dir (default: <data>_wds)')
    ap.add_argument('--split',      default='both',
                    choices=['train', 'val', 'both'])
    ap.add_argument('--shard-size', type=int, default=1000,
                    help='Images per tar shard (default: 1000)')
    ap.add_argument('--image-size', type=int, default=0,
                    help='Resize images while packing (0 = keep original, default: 0)')
    args = ap.parse_args()

    data_root = Path(args.data).expanduser()
    out_root  = Path(args.out).expanduser() if args.out \
                else data_root.parent / (data_root.name + '_wds')

    splits = ['train', 'val'] if args.split == 'both' else [args.split]
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"[skip] {split_dir} not found")
            continue
        pack_split(split_dir, out_root, split, args.shard_size, args.image_size)


if __name__ == '__main__':
    main()
