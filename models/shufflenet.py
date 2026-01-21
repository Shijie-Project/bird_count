import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ShuffleNet_V2_X1_0_Weights


class ShuffleNetDensityNet(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features

        in_ch = self._infer_out_channels()

        self.reg_layer = nn.Sequential(
            nn.Conv2d(in_ch, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    @torch.no_grad()
    def _infer_out_channels(self) -> int:
        self.eval()
        x = torch.zeros(1, 3, 224, 224)
        y = self.features(x)
        return y.shape[1]

    def train(self, mode=True):
        super().train(mode)
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        # Feature extraction via ShuffleNet
        x = self.features(x)

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Density Estimation Head
        x = self.reg_layer(x)
        mu = self.density_layer(x)

        # Normalization (MUNIT block)
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).view(B, 1, 1, 1)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed


def get_shufflenet_density_model():
    backbone = torchvision.models.shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    features = nn.Sequential(*list(backbone.children())[:-3])  # up to stage3 (/16)
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
