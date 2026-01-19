import os
import random
from glob import glob

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms

from .utils import gen_discrete_map, random_crop


class Bird(data.Dataset):
    def __init__(self, root_path, crop_size, downsample_ratio=8, split="train"):
        self.root_path = root_path
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
        self.trans = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )

        self.split = split
        assert split in ("train", "val", "train_legacy", "val_legacy")

        self.im_list = list(glob(os.path.join(self.root_path, "images", split, "*.jpg")))
        self.im_list.sort()
        print(f"number of img: {len(self.im_list)}")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]

        name = os.path.basename(img_path).split(".")[0]

        img = Image.open(img_path).convert("RGB")
        h, w = img.size

        keypoints = np.loadtxt(os.path.join(self.root_path, "annotations", self.split, name + ".txt"), ndmin=2)
        keypoints = np.array(keypoints)

        if keypoints.shape[1] == 3:
            keypoints = keypoints[:, 1:] * np.array([w, h])

        if "train" in self.split:
            return self.train_transform(img, keypoints)

        img = self.trans(img)
        return img_path, img, len(keypoints), name

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (
                (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            )
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)

        gt_discrete = np.expand_dims(gt_discrete, 0)

        return (
            self.trans(img),
            torch.from_numpy(keypoints.copy()).float(),
            torch.from_numpy(gt_discrete.copy()).float(),
        )
