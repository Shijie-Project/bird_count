import os
from glob import glob

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from . import transforms as T
from .targets import downsample_count_map, gen_discrete_map


def build_train_transform(
    crop_size,
    scale_range=(0.8, 1.25),
    color_jitter=(0.4, 0.4, 0.3, 0.05),
    gamma_range=(0.7, 1.3),
    gamma_p=0.5,
    noise_std=0.02,
    noise_p=0.5,
    hflip_p=0.5,
    vflip_p=0.5,
):
    return T.Compose(
        [
            T.RandomScale(scale_range),
            T.RandomSquareCrop(crop_size),
            T.RandomHFlip(hflip_p),
            T.RandomVFlip(vflip_p),
            T.RandomRot90(),
            T.ColorJitter(*color_jitter),
            T.RandomGamma(gamma_range, gamma_p),
            T.ToTensor(),
            T.Normalize(),
            T.RandomGaussianNoise(noise_std, noise_p),
        ]
    )


def build_val_transform():
    return T.Compose([T.ToTensor(), T.Normalize()])


class Bird(data.Dataset):
    """Counting dataset returning (image, keypoints, density) tuples.

    Train sample: (image[3, c, c], keypoints[N, 2], gt[1, c/d, c/d])
    Val sample:   (img_path, image[3, h, w], gt[h/d, w/d], name)
    """

    def __init__(
        self,
        root_path,
        crop_size,
        downsample_ratio=8,
        split="train",
        transform=None,
        train_aug=None,
    ):
        if crop_size % downsample_ratio:
            raise ValueError(f"crop_size {crop_size} not divisible by downsample_ratio {downsample_ratio}")

        self.root_path = root_path
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        self.split = split
        self.is_train = "train" in split

        if transform is None:
            transform = (
                build_train_transform(crop_size, **(train_aug or {})) if self.is_train else build_val_transform()
            )
        self.transform = transform

        self.im_list = sorted(glob(os.path.join(root_path, "images", split, "*.jpg")))
        print(f"number of img: {len(self.im_list)}")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        img_path = self.im_list[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]

        img = Image.open(img_path).convert("RGB")
        ann_path = os.path.join(self.root_path, "annotations", self.split, name + ".txt")
        keypoints = self._load_keypoints(ann_path, img.size)

        img, keypoints = self.transform(img, keypoints)

        # img is now a normalized (3, H, W) tensor; keypoints are still numpy in image coords.
        h, w = img.shape[-2:]
        density = gen_discrete_map(h, w, keypoints)
        density = downsample_count_map(density, self.d_ratio)

        return {
            "path": img_path,
            "name": name,
            "image": img,  # (3, H, W)
            "keypoints": torch.from_numpy(np.ascontiguousarray(keypoints)).float(),  # (N, 2)
            "density": torch.from_numpy(density).float().unsqueeze(0),  # (1, H/d, W/d)
        }

    @staticmethod
    def _load_keypoints(path, img_size):
        w, h = img_size
        kps = np.loadtxt(path, ndmin=2)
        if kps.size == 0:
            return np.empty((0, 2))
        if kps.shape[1] == 3:
            # YOLO-style "class x_norm y_norm" → (x, y) in pixels.
            kps = kps[:, 1:] * np.array([w, h])
        return kps
