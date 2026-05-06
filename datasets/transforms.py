"""Keypoint-aware transforms.

Each transform's __call__ takes (img, keypoints) and returns (img, keypoints).
Image-only ops (color jitter, normalize, ...) pass keypoints through unchanged.
Keypoints are an (N, 2) array in (x, y) pixel coordinates of the current image.
"""

import random

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms as T


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, keypoints):
        for t in self.transforms:
            img, keypoints = t(img, keypoints)
        return img, keypoints


class RandomScale:
    """Resize image and rescale keypoints by a uniformly random factor."""

    def __init__(self, scale_range=(0.8, 1.25)):
        self.lo, self.hi = scale_range

    def __call__(self, img, keypoints):
        scale = random.uniform(self.lo, self.hi)
        wd, ht = img.size
        new_wd = max(int(round(wd * scale)), 1)
        new_ht = max(int(round(ht * scale)), 1)
        if (new_wd, new_ht) != (wd, ht):
            img = img.resize((new_wd, new_ht), Image.BICUBIC)
            if len(keypoints):
                keypoints = keypoints * np.array([new_wd / wd, new_ht / ht])
        return img, keypoints


class RandomSquareCrop:
    """Random square crop. Upscales first if image is smaller than `size`."""

    def __init__(self, size):
        self.size = size

    def __call__(self, img, keypoints):
        wd, ht = img.size
        st = min(wd, ht)
        if st < self.size:
            rr = self.size / st
            new_wd = int(round(wd * rr))
            new_ht = int(round(ht * rr))
            img = img.resize((new_wd, new_ht), Image.BICUBIC)
            if len(keypoints):
                keypoints = keypoints * np.array([new_wd / wd, new_ht / ht])
            wd, ht = new_wd, new_ht

        i = random.randint(0, ht - self.size)
        j = random.randint(0, wd - self.size)
        img = F.crop(img, i, j, self.size, self.size)
        if len(keypoints):
            keypoints = keypoints - np.array([j, i])
            mask = (
                (keypoints[:, 0] >= 0)
                & (keypoints[:, 0] < self.size)
                & (keypoints[:, 1] >= 0)
                & (keypoints[:, 1] < self.size)
            )
            keypoints = keypoints[mask]
        else:
            keypoints = np.empty((0, 2))
        return img, keypoints


class RandomHFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, keypoints):
        if random.random() < self.p:
            img = F.hflip(img)
            if len(keypoints):
                keypoints = keypoints.copy()
                keypoints[:, 0] = (img.size[0] - 1) - keypoints[:, 0]
        return img, keypoints


class RandomVFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, keypoints):
        if random.random() < self.p:
            img = F.vflip(img)
            if len(keypoints):
                keypoints = keypoints.copy()
                keypoints[:, 1] = (img.size[1] - 1) - keypoints[:, 1]
        return img, keypoints


class RandomRot90:
    """Random k * 90° rotation. Requires a square image (assert)."""

    def __call__(self, img, keypoints):
        k = random.randint(0, 3)
        if k == 0:
            return img, keypoints
        w, h = img.size
        assert w == h, "RandomRot90 requires square input"
        img = F.rotate(img, 90 * k)
        if len(keypoints):
            x = keypoints[:, 0].copy()
            y = keypoints[:, 1].copy()
            keypoints = keypoints.copy()
            # F.rotate is counter-clockwise.
            if k == 1:
                keypoints[:, 0] = y
                keypoints[:, 1] = (w - 1) - x
            elif k == 2:
                keypoints[:, 0] = (w - 1) - x
                keypoints[:, 1] = (h - 1) - y
            else:  # k == 3
                keypoints[:, 0] = (h - 1) - y
                keypoints[:, 1] = x
        return img, keypoints


class ColorJitter:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05):
        self._cj = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, keypoints):
        return self._cj(img), keypoints


class RandomGamma:
    def __init__(self, gamma_range=(0.7, 1.3), p=0.5):
        self.lo, self.hi = gamma_range
        self.p = p

    def __call__(self, img, keypoints):
        if random.random() < self.p:
            img = F.adjust_gamma(img, random.uniform(self.lo, self.hi))
        return img, keypoints


class ToTensor:
    def __call__(self, img, keypoints):
        return F.to_tensor(img), keypoints


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img, keypoints):
        return F.normalize(img, self.mean, self.std), keypoints


class RandomGaussianNoise:
    """Adds Gaussian noise in (already-normalized) tensor space."""

    def __init__(self, std=0.02, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, img, keypoints):
        if self.std > 0 and random.random() < self.p:
            img = img + torch.randn_like(img) * self.std
        return img, keypoints


class PadToMultiple:
    """Right/bottom-pad a (C, H, W) tensor so H and W are multiples of `m`.

    Padding is added with value 0 (≈ neutral gray after ImageNet normalization),
    so the model sees an unobtrusive border. Keypoints are unchanged because
    padding is appended after the original content.
    """

    def __init__(self, multiple: int):
        self.m = multiple

    def __call__(self, img, keypoints):
        h, w = img.shape[-2:]
        pad_w = (self.m - w % self.m) % self.m
        pad_h = (self.m - h % self.m) % self.m
        if pad_h or pad_w:
            # torch.nn.functional.pad: (left, right, top, bottom)
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h))
        return img, keypoints
