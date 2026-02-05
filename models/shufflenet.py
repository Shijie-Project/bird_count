import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ShuffleNet_V2_X1_0_Weights, shufflenet_v2_x1_0


class ShuffleNetDensityNet(nn.Module):
    """
    Density Estimation Network using ShuffleNetV2 as backbone.
    Designed for efficient bird counting on edge devices.
    """

    def __init__(self, backbone_features: nn.Module):
        super().__init__()
        self.features = backbone_features

        # Automatically infer the number of output channels from backbone
        in_channels = self._get_backbone_out_channels()

        # Density Regression Head
        self.reg_layer = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # Final 1x1 conv to produce a single-channel density map
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1), nn.ReLU())

    @torch.no_grad()
    def _get_backbone_out_channels(self) -> int:
        """Helper to determine the feature map depth after backbone processing."""
        self.eval()
        dummy_input = torch.zeros(1, 3, 224, 224)
        output = self.features(dummy_input)
        return output.shape[1]

    def train(self, mode=True):
        """Override train mode to keep BatchNorm in eval mode if needed for stability."""
        super().train(mode)
        # Example: Keeping backbone BN fixed if fine-tuning
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        Returns the raw density map. Summing this map yields the total count.
        """
        # 1. Feature extraction
        x = self.features(x)

        # 2. Upsampling to recover some spatial resolution (8x downsample -> 4x downsample)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # 3. Density estimation
        x = self.reg_layer(x)
        density_map = self.density_layer(x)

        return density_map


def get_shufflenet_density_model(model_path: str = None) -> nn.Module:
    """
    Factory function to create and initialize the ShuffleNetDensityNet.

    Args:
        model_path: Path to the .pth checkpoint file.

    Returns:
        The initialized model in eval mode.
    """
    # 1. Load Pretrained Backbone (ShuffleNetV2 x1.0)
    # Using 'weights' instead of 'pretrained' to follow modern Torchvision API
    backbone = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)

    # 2. Extract feature layers (removing the fully connected classifier)
    # Typically takes layers up to the global pool
    features = nn.Sequential(*list(backbone.children())[:-3])

    # 3. Initialize the full Density Network
    model = ShuffleNetDensityNet(features)

    # 4. Load custom weights if path is provided
    if model_path and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_map="cpu")
            # Handle cases where model is saved as a full dict or just state_dict
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            model.load_state_dict(state_dict)
            print(f"[Model] Successfully loaded weights from {model_path}")
        except Exception as e:
            print(f"[Model] Warning: Failed to load weights: {e}")
    else:
        print("[Model] Warning: No checkpoint found. Using ImageNet initialized backbone.")

    model.eval()
    return model


if __name__ == "__main__":
    # Sanity Check Block
    model = get_shufflenet_density_model()
    dummy_input = torch.randn(2, 3, 224, 224)
    mu, mu_normed = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Density Map shape: {mu.shape}")  # Expected: [2, 1, 28, 28] (1/8 size)
    print(f"Check Non-negative: {mu.min() >= 0}")
    print("Model build successful.")
