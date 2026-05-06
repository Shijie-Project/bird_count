"""Generic helpers shared across model implementations."""

import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


logger = logging.getLogger(__name__)


STATE_KEYS_PREFERENCE = ("ema_state_dict", "model_state_dict", "state_dict")


def extract_state_dict(checkpoint) -> "OrderedDict[str, torch.Tensor]":
    """Pull a state_dict out of a checkpoint that may be wrapped in a metadata dict.

    Prefers EMA weights when available (best for inference). Strips any
    `module.` prefix left over from DataParallel.
    """
    if isinstance(checkpoint, dict) and not all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        for key in STATE_KEYS_PREFERENCE:
            if key in checkpoint and checkpoint[key] is not None:
                logger.info(f"Loading weights from checkpoint key '{key}'")
                state_dict = checkpoint[key]
                break
        else:
            raise KeyError(f"Checkpoint dict has none of {STATE_KEYS_PREFERENCE}; got keys {list(checkpoint.keys())}")
    else:
        state_dict = checkpoint

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = OrderedDict((k[len("module.") :], v) for k, v in state_dict.items())
    return state_dict


def count_conv_bn_pairs(module: nn.Module) -> int:
    """Count adjacent Conv2d+BatchNorm2d pairs inside any nn.Sequential descendant."""
    n = 0
    for child in module.modules():
        if isinstance(child, nn.Sequential):
            kids = list(child.children())
            for i in range(len(kids) - 1):
                if isinstance(kids[i], nn.Conv2d) and isinstance(kids[i + 1], nn.BatchNorm2d):
                    n += 1
    return n


def fuse_conv_bn_recursive(module: nn.Module) -> None:
    """Recursively replace adjacent Conv2d+BN2d pairs in nn.Sequential with the fused Conv2d.

    Eval-mode only: BN running stats must be frozen for fusion to be exact.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Sequential):
            kids = list(child.children())
            new_layers = []
            i = 0
            while i < len(kids):
                cur = kids[i]
                if i + 1 < len(kids) and isinstance(cur, nn.Conv2d) and isinstance(kids[i + 1], nn.BatchNorm2d):
                    new_layers.append(fuse_conv_bn_eval(cur, kids[i + 1]))
                    i += 2
                else:
                    fuse_conv_bn_recursive(cur)
                    new_layers.append(cur)
                    i += 1
            setattr(module, name, nn.Sequential(*new_layers))
        else:
            fuse_conv_bn_recursive(child)
