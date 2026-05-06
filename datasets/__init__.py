import random

import numpy as np
import torch

from .bird import Bird, build_train_transform, build_val_transform


__all__ = ["Bird", "build_train_transform", "build_val_transform", "collate", "seed_worker"]


_STACK_KEYS = ("image", "density")


def collate(batch):
    """Dict-aware DataLoader collate.

    Stacks fixed-shape tensors (`image`, `density`); keeps variable-length /
    string fields (`keypoints`, `path`, `name`) as plain lists. Train uses fixed
    crops so stacking always works; val runs at batch=1 so per-image size
    differences are fine.
    """
    out = {}
    for key in batch[0]:
        vals = [s[key] for s in batch]
        out[key] = torch.stack(vals, 0) if key in _STACK_KEYS else vals
    return out


def seed_worker(worker_id):
    """DataLoader worker init: derives a per-worker seed for python/numpy RNGs.

    Pass as `worker_init_fn=seed_worker` to ensure augmentations are independent
    across workers.
    """
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
