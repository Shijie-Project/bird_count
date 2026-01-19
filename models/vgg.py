import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models


class VGG19DensityNet(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features

        in_ch = self._infer_out_channels()

        self.reg_layer = nn.Sequential(
            nn.Conv2d(in_ch, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    @torch.no_grad()
    def _infer_out_channels(self) -> int:
        self.eval()
        x = torch.zeros(1, 3, 224, 224)
        y = self.features(x)
        return int(y.shape[1])

    def forward(self, x):
        x = self.features(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = self.reg_layer(x)
        mu = self.density_layer(x)

        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed


def VGG19():
    """VGG 19-layer model (configuration "E") model pre-trained on ImageNet"""
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    return VGG19DensityNet(features=vgg.features)
