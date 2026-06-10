"""Shared building blocks used by both sf_seg_r18 and sf_seg_v2."""
import torch.nn as nn


def _gn(C: int) -> nn.GroupNorm:
    for g in [8, 4, 2, 1]:
        if C % g == 0:
            return nn.GroupNorm(g, C)
