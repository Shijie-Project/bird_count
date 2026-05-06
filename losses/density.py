"""Auxiliary Gaussian-density loss for crowd counting.

The DM-Count terms (OT + count + normalized-L1) are silent on empty regions:
when the GT count is zero, the OT and the "TV" term both contribute nothing,
and only the global count loss constrains the prediction. This module provides
a per-pixel L1 against a Gaussian-smoothed count target so the network gets
dense supervision everywhere — including background — which stabilizes early
training and reduces false-positive density spikes in low-density frames.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel_2d(sigma, kernel_size=None):
    if kernel_size is None:
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32)
    g = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, kernel_size, kernel_size), kernel_size


class GaussianSmooth(nn.Module):
    """2-D Gaussian smoothing on (B, 1, H, W) maps via depthwise conv."""

    def __init__(self, sigma):
        super().__init__()
        kernel, k = _gaussian_kernel_2d(sigma)
        self.register_buffer("kernel", kernel)
        self.pad = k // 2

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=self.pad)


class DensityAuxLoss(nn.Module):
    """Per-pixel L1/MSE between the raw prediction and a Gaussian-smoothed GT.

    Aggregation is sum-over-pixels-per-image then mean-over-batch so the
    magnitude is comparable to the global count loss (each chicken contributes
    ~1.0 to both). `dense_weight_alpha > 0` upweights pixels with high target
    density, biasing the loss toward pile-up regions.
    """

    def __init__(self, sigma, dense_weight_alpha=0.0, mode="l1"):
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be > 0; pass aux_sigma=0 in DMCountLoss to disable")
        if mode not in ("l1", "mse"):
            raise ValueError(f"mode must be 'l1' or 'mse', got {mode!r}")
        self.smooth = GaussianSmooth(sigma)
        self.alpha = dense_weight_alpha
        self.mode = mode

    def forward(self, pred, gt_discrete):
        target = self.smooth(gt_discrete)
        diff = (pred - target).abs() if self.mode == "l1" else (pred - target).pow(2)
        if self.alpha > 0:
            diff = diff * (1.0 + self.alpha * target.detach())
        return diff.sum(dim=(1, 2, 3)).mean()
