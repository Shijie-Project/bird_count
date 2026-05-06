"""Exponential Moving Average (EMA) of model parameters for stable evaluation."""

from copy import deepcopy

import torch
import torch.nn as nn


class ModelEMA:
    """Maintain a shadow copy of `model` whose state is the EMA of `model`'s.

    Float parameters and float buffers are blended with decay; integer buffers
    (e.g. BN's `num_batches_tracked`) are copied verbatim because float math
    on them is meaningless and silently lossy after the cast back to int.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(d).add_(model_v, alpha=1.0 - d)
            else:
                ema_v.copy_(model_v)
