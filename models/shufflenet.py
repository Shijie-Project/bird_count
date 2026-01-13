import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ShuffleNet_V2_X1_0_Weights


HEAD_IN_CHANNELS = 512


class ShuffleNetDensityNet(nn.Module):
    def __init__(self, features, head_in_channels: int = HEAD_IN_CHANNELS):
        super().__init__()
        self.features = features

        out_channels = self._infer_out_channels()

        self.conv_adapter = nn.Sequential(
            nn.Conv2d(out_channels, head_in_channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

        self.reg_layer = nn.Sequential(
            nn.Conv2d(head_in_channels, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, kernel_size=1), nn.ReLU(inplace=True))

    @torch.no_grad()
    def _infer_out_channels(self) -> int:
        self.eval()
        x = torch.zeros(1, 3, 224, 224)
        y = self.features(x)
        return y.shape[1]

    def forward(self, x):
        # Feature extraction via ShuffleNet
        x = self.features(x)
        x = self.conv_adapter(x)

        # Upsampling (If your ShuffleNet features are 1/16, this upsamples to 1/8)
        # Note: If VGG was 1/8 downsampling, ShuffleNet may be 1/16. Check your feature sizes.
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)

        # Density Estimation Head
        x = self.reg_layer(x)
        mu = self.density_layer(x)

        # Normalization (MUNIT block)
        b = mu.shape[0]
        mu_sum = mu.flatten(1).sum(dim=1, keepdim=True).view(b, 1, 1, 1)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed


def get_shufflenet_density_model():
    backbone = torchvision.models.shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    features = nn.Sequential(*list(backbone.children())[:-2])  # up to stage4 (/32)
    return ShuffleNetDensityNet(features)


if __name__ == "__main__":
    # Sanity Check Block
    model = get_shufflenet_density_model()
    dummy_input = torch.randn(2, 3, 224, 224)
    mu, mu_normed = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Density Map shape: {mu.shape}")  # Expected: [2, 1, 28, 28] (1/8 size)
    print(f"Check Non-negative: {mu.min() >= 0}")
    print("Model build successful.")
