import random

import numpy as np
import torch

from .bird import Bird, build_train_transform, build_val_transform


__all__ = ["Bird", "build_train_transform", "build_val_transform", "seed_worker"]


def seed_worker(worker_id):
    """DataLoader worker init: derives a per-worker seed for python/numpy RNGs.

    Pass as `worker_init_fn=seed_worker` to ensure augmentations are independent
    across workers.
    """
    seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
