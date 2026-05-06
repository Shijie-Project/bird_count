"""ShuffleNetV2-backed density estimator with U-Net-style lateral skip.

Backbone: torchvision ShuffleNetV2-1.0x truncated at stage3. The regression
head fuses stage2 features (1/8) with upsampled stage3 features (1/16 -> 1/8)
before predicting the density map at **input/8** resolution, matching the
dataset's `downsample_ratio=8`.

The skip connection feeds high-resolution stage2 features directly into the
head so it can localize small chickens precisely, while still benefiting from
the deeper semantic features of stage3.
"""

import logging
import os
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ShuffleNet_V2_X1_0_Weights, shufflenet_v2_x1_0

from .utils import count_conv_bn_pairs, extract_state_dict, fuse_conv_bn_recursive


logger = logging.getLogger(__name__)


class _BackboneFeatures(nn.Module):
    """ShuffleNetV2 stem + stage2 + stage3, exposing both stage outputs."""

    def __init__(self):
        super().__init__()
        bb = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        self.conv1 = bb.conv1
        self.maxpool = bb.maxpool
        self.stage2 = bb.stage2
        self.stage3 = bb.stage3

    def forward(self, x):
        x = self.maxpool(self.conv1(x))  # 1/4
        f2 = self.stage2(x)  # 1/8
        f3 = self.stage3(f2)  # 1/16
        return f2, f3


class ShuffleNetDensityNet(nn.Module):
    """U-Net-style density head: concat(stage2, upsample(stage3)) -> head -> density.

    forward(x: [B, 3, H, W]) -> density: [B, 1, H/8, W/8]
    """

    def __init__(self, freeze_backbone_bn: bool = True):
        super().__init__()
        self.backbone = _BackboneFeatures()
        self.freeze_backbone_bn = freeze_backbone_bn

        c2, c3 = self._infer_stage_channels()
        head_in = c2 + c3  # 116 + 232 = 348

        self.reg_layer = nn.Sequential(
            nn.Conv2d(head_in, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(inplace=True),  # density must be non-negative
        )

    @torch.no_grad()
    def _infer_stage_channels(self):
        was_training = self.training
        self.eval()
        try:
            f2, f3 = self.backbone(torch.zeros(1, 3, 64, 64))
            return f2.shape[1], f3.shape[1]
        finally:
            self.train(was_training)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_backbone_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f2, f3 = self.backbone(x)
        f3_up = F.interpolate(f3, scale_factor=2, mode="bilinear", align_corners=False)
        fused = torch.cat([f2, f3_up], dim=1)
        x = self.reg_layer(fused)
        return self.density_layer(x)

    def fuse_model(self):
        """Fuse every Conv2d+BatchNorm2d pair in the model (backbone + head)."""
        self.eval()
        n_before = count_conv_bn_pairs(self)
        fuse_conv_bn_recursive(self)
        n_after = count_conv_bn_pairs(self)
        logger.info(f"Fused {n_before - n_after} Conv+BN pairs ({n_before} -> {n_after}).")


_LEGACY_BACKBONE_MAP = {
    "features.0.": "backbone.conv1.",
    "features.2.": "backbone.stage2.",
    "features.3.": "backbone.stage3.",
    # features.1 was maxpool (no parameters)
}


def _migrate_legacy_state_dict(state_dict):
    """Remap pre-skip-connection checkpoints to the current module layout.

    Old layout: a single `features` Sequential containing [conv1, maxpool,
    stage2, stage3]. Current layout: a `backbone` submodule with named
    children. The regression-head input-channel count changed (232 -> 348),
    so old `reg_layer.0.*` and `density_layer.*` weights are dropped — they
    need to be re-trained.
    """
    if not any(k.startswith("features.") for k in state_dict):
        return state_dict, False, []

    new_sd = OrderedDict()
    dropped = []
    for k, v in state_dict.items():
        new_k = k
        for old_prefix, new_prefix in _LEGACY_BACKBONE_MAP.items():
            if k.startswith(old_prefix):
                new_k = new_prefix + k[len(old_prefix) :]
                break
        if k.startswith(("reg_layer.0.", "density_layer.")):
            dropped.append(k)
            continue
        new_sd[new_k] = v
    return new_sd, True, dropped


def get_shufflenet_density_model(
    model_path: Optional[str] = None,
    device: "str | torch.device" = "cpu",
    fuse: bool = False,
    freeze_backbone_bn: bool = True,
) -> nn.Module:
    """Build ShuffleNetDensityNet, optionally load weights, optionally fuse."""
    model = ShuffleNetDensityNet(freeze_backbone_bn=freeze_backbone_bn)

    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        sd = extract_state_dict(checkpoint)
        sd, migrated, dropped = _migrate_legacy_state_dict(sd)
        missing, unexpected = model.load_state_dict(sd, strict=not migrated)
        if migrated:
            logger.warning(
                "Loaded legacy checkpoint: backbone remapped to new layout; "
                f"dropped {len(dropped)} old head keys (head will re-init from scratch). "
                "Re-train or fine-tune the regression head before deploying."
            )
        if missing:
            logger.warning(f"Missing keys (random-init): {len(missing)} entries, e.g. {missing[:3]}")
        if unexpected:
            logger.warning(f"Unexpected keys (ignored): {len(unexpected)} entries, e.g. {unexpected[:3]}")
        logger.info(f"Loaded weights from {model_path}")
    elif model_path:
        logger.warning(f"Checkpoint not found at {model_path}; using random init.")
    else:
        logger.info("No model path provided; using random init (dev mode).")

    model.eval()
    model.to(device)
    if fuse:
        model.fuse_model()
    return model
