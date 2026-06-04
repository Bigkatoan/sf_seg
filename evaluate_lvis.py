#!/usr/bin/env python3
"""Wrapper for src.visualization.evaluation — see that module for details."""
from src.visualization.evaluation import evaluate
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/sf_seg_best.pt")
    p.add_argument("--lvis-ann", default="data/lvis/lvis_v1_val.json")
    p.add_argument("--data-root", default="data")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument(
        "--max-images",
        type=int,
        default=1000,
        help="Max images to evaluate (0=all). Default 1000 for quick run.",
    )
    p.add_argument("--vis-samples", type=int, default=8,
                   help="Number of visualisation samples to save")
    p.add_argument("--output-dir", default="outputs/lvis_eval")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    evaluate(p.parse_args())
