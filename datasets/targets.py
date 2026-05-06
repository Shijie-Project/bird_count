"""Density target generation."""

import numpy as np


def gen_discrete_map(im_height, im_width, points):
    """Place one count per (rounded) keypoint into a (h, w) float32 map.

    Total mass equals number of points, even when multiple points round to the
    same pixel.
    """
    out = np.zeros((im_height, im_width), dtype=np.float32)
    if len(points) == 0:
        return out
    pts = np.asarray(points)
    px = np.clip(np.round(pts[:, 0]).astype(np.int64), 0, im_width - 1)
    py = np.clip(np.round(pts[:, 1]).astype(np.int64), 0, im_height - 1)
    np.add.at(out, (py, px), 1.0)
    return out


def downsample_count_map(density, ratio):
    """Sum-pool a (h, w) count map by `ratio` along each axis."""
    h, w = density.shape
    if h % ratio or w % ratio:
        raise ValueError(f"density shape {(h, w)} not divisible by {ratio}")
    return density.reshape(h // ratio, ratio, w // ratio, ratio).sum(axis=(1, 3))
