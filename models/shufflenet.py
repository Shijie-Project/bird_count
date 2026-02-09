import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ShuffleNet_V2_X1_0_Weights, shufflenet_v2_x1_0


# Setup Module Logger
logger = logging.getLogger(__name__)


class ShuffleNetDensityNet(nn.Module):
    """
    Density Estimation Network using ShuffleNetV2 as backbone.
    Optimized for Inference on GTX 1080Ti.
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
        self.density_layer = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU(inplace=True),  # Ensure non-negative density
        )

    @torch.no_grad()
    def _get_backbone_out_channels(self) -> int:
        """Helper to determine the feature map depth."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns:
            density_map (Tensor): [B, 1, H/4, W/4] The pixel-wise density.
        """
        # 1. Feature extraction (downsample 32x usually)
        x = self.features(x)

        # 2. Upsampling (Recover spatial resolution)
        # Bilinear is fast on 1080Ti
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # 3. Density estimation
        x = self.reg_layer(x)
        density_map = self.density_layer(x)

        return density_map

    def fuse_model(self):
        """
        Industrial Optimization: Fuses Conv2d + BatchNorm2d layers.
        This reduces memory access overhead during inference.
        """
        from torch.nn.utils.fusion import fuse_conv_bn_eval

        logger.info("Fusing Conv+BN layers for faster inference...")

        # Fuse inside the regression head
        # We know the structure is Conv -> BN -> ReLU -> Conv -> BN -> ReLU
        # Note: Fusing usually requires the module to be in eval mode
        self.eval()

        try:
            # Fuse first block
            fused_0 = fuse_conv_bn_eval(self.reg_layer[0], self.reg_layer[1])
            # Fuse second block
            fused_1 = fuse_conv_bn_eval(self.reg_layer[3], self.reg_layer[4])

            # Rebuild the sequential model with fused layers
            # Note: We remove the BN layers (indices 1 and 4)
            self.reg_layer = nn.Sequential(
                fused_0,
                self.reg_layer[2],  # ReLU
                fused_1,
                self.reg_layer[5],  # ReLU
            )

            logger.info("Model fusion complete.")
        except Exception as e:
            logger.warning(f"Model fusion failed (safe to skip): {e}")


def get_shufflenet_density_model(
    model_path: Optional[str] = None, device: str | torch.device = "cpu", fuse: bool = False
) -> nn.Module:
    """
    Factory function to create, load, and optimize the ShuffleNetDensityNet.
    """
    logger.info("Initializing ShuffleNet Density Model...")

    # 1. Load Backbone
    # Using 'weights' instead of 'pretrained'
    backbone = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)

    # 2. Extract feature layers (remove classifier)
    # ShuffleNetV2: children() are [stage1, ..., conv5, fc]
    # We take everything except the last few layers to get spatial features
    # Check architecture: usually we want up to 'conv5'
    features = nn.Sequential(*list(backbone.children())[:-3])

    # 3. Initialize Density Network
    model = ShuffleNetDensityNet(features)

    # 4. Load Weights
    if model_path and os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)

            # Handle DataParallel prefix 'module.'
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict, strict=True)
            logger.info(f"Loaded weights from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            # Depending on policy, we might want to raise here
            raise
    else:
        if model_path:
            logger.warning(f"Checkpoint file not found at {model_path}. Using random init.")
        else:
            logger.info("No model path provided. Using random init (Dev Mode).")

    # 5. Optimization
    model.eval()
    model.to(device)

    if fuse:
        model.fuse_model()

    return model


def export_to_onnx(model_path, onnx_output_path, device="cuda"):
    # 1. Initialize and load the model
    # We set fuse=True to simplify the graph before exporting
    model = get_shufflenet_density_model(model_path=model_path, device=device, fuse=True)
    model.eval()

    # 2. Define dummy input
    # Standard size is usually 224x224, but ensure it matches your inference size
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 720, 1080).to(device)

    # 3. Export to ONNX
    logger.info(f"Exporting model to {onnx_output_path}...")

    try:
        torch.onnx.export(
            model,  # Model being run
            dummy_input,  # Model input (or a tuple for multiple inputs)
            onnx_output_path,  # Where to save the model
            export_params=True,  # Store the trained parameter weights inside the model file
            opset_version=11,  # The ONNX version to export the model to (11+ recommended)
            do_constant_folding=True,  # Whether to execute constant folding for optimization
            input_names=["input"],  # The model's input names
            output_names=["output"],  # The model's output names
            # Enable dynamic axes for variable batch size or resolutions
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size", 2: "out_height", 3: "out_width"},
            },
        )
        logger.info("ONNX export complete.")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")


if __name__ == "__main__":
    CHECKPOINT = "./ckpts/shufflenet_best_model_214800.pth"
    OUTPUT_ONNX = "./ckpts/shufflenet_best_model_214800.onnx"
    export_to_onnx(CHECKPOINT, OUTPUT_ONNX)

    import onnx

    model = onnx.load(OUTPUT_ONNX)
    onnx.checker.check_model(model)
    print("ONNX Model check passed!")
