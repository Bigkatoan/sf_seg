#!/usr/bin/env python3
"""Wrapper for src.visualization.attention — see that module for details."""
from src.visualization.attention import visualize

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/sf_seg_best.pt")
    p.add_argument("--data-root", default="data")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-images", type=int, default=4)
    p.add_argument(
        "--show-channels",
        type=int,
        default=8,
        help="Max individual channels to show from selected head",
    )
    p.add_argument(
        "--head",
        default="large",
        choices=["small", "medium", "large"],
        help="Which attention head to show individual channels for",
    )
    p.add_argument(
        "--min-range",
        type=float,
        default=0.05,
        help="Min spatial range to include a channel (0=all, 0.05=active only)",
    )
    p.add_argument("--output-dir", default="outputs/attention_vis")
    p.add_argument("--seed", type=int, default=42)
    visualize(p.parse_args())
