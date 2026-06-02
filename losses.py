#!/usr/bin/env python3
"""Loss functions for segmentation tasks."""
from typing import Optional

import torch


def iou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3, reduction: str = "mean") -> torch.Tensor:
    """IoU loss (1 - IoU).

    Formula (per-sample):
        inter = gt * pred
        union = (gt + pred) - inter
        iou = (inter.sum + eps) / (union.sum + eps)
        loss = 1 - iou

    pred is expected to be probabilities in [0,1] (i.e., model with final sigmoid).
    target should be binary (0/1) of same shape as pred.
    Returns: scalar loss (averaged over batch when reduction='mean').
    """
    pred = pred.float()
    target = target.float()

    if pred.shape != target.shape:
        # allow target to be (B,H,W) when pred is (B,1,H,W)
        if pred.dim() == 4 and target.dim() == 3 and pred.size(0) == target.size(0):
            target = target.unsqueeze(1)
        else:
            raise ValueError(f"pred and target must have same shape, got {pred.shape} vs {target.shape}")

    b = pred.shape[0]
    pred_flat = pred.view(b, -1)
    target_flat = target.view(b, -1)

    inter = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    loss = 1.0 - iou

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss

def mse_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Mean Squared Error loss."""
    pred = pred.float()
    target = target.float()

    if pred.shape != target.shape:
        # allow target to be (B,H,W) when pred is (B,1,H,W)
        if pred.dim() == 4 and target.dim() == 3 and pred.size(0) == target.size(0):
            target = target.unsqueeze(1)
        else:
            raise ValueError(f"pred and target must have same shape, got {pred.shape} vs {target.shape}")

    loss = (pred - target) ** 2

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss

def combine_losses(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.1, reduction: str = "mean") -> torch.Tensor:
    """Combined loss: alpha * IoU + (1-alpha) * MSE."""
    iou = iou_loss(pred, target, reduction=reduction)
    mse = mse_loss(pred, target, reduction=reduction)
    return alpha * iou + (1 - alpha) * mse

if __name__ == "__main__":
    # quick smoke test
    import torch
    from models import UNet

    net = UNet(final_sigmoid=True)
    x = torch.randn(1, 3, 128, 128)
    y = (torch.rand(1, 1, 128, 128) > 0.5).float()
    p = net(x)
    print("pred shape", p.shape)
    print("loss", iou_loss(p, y).item())
    print("combined loss", combine_losses(p, y).item())