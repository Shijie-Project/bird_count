import logging
import os
import random
import sys
from collections import deque

import cv2
import numpy as np
import torch


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For speed, we don't set deterministic to True unless debugging
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class SaveHandle:
    """
    Manage checkpoint saving.
    Keeps a maximum of `max_num` checkpoints, deleting the oldest one.
    Refactored to use `collections.deque` for cleaner FIFO logic.
    """

    def __init__(self, max_num):
        self.max_num = max_num
        self.save_queue = deque(maxlen=max_num)

    def append(self, save_path):
        # If queue is full, deque automatically handles 'maxlen',
        # but we need to physically delete the file of the item being pushed out.
        if len(self.save_queue) == self.max_num:
            # Get the oldest path (which is about to be removed from queue logic naturally)
            oldest_path = self.save_queue[0]
            self.remove_file(oldest_path)

        self.save_queue.append(save_path)

    def remove_file(self, path):
        try:
            if os.path.exists(path):
                os.remove(path)
                logging.info(f"Removed old checkpoint: {path}")
        except Exception as e:
            logging.warning(f"Failed to remove {path}: {e}")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._count = 0

    def update(self, val, n=1):
        self._val = val
        self._sum += val * n
        self._count += n
        self._avg = 1.0 * self._sum / self._count

    @property
    def val(self):
        return self._val

    @property
    def avg(self):
        return self._avg


class Logger:
    """
    Robust Logger wrapper.
    Prevents duplicate logs if instantiated multiple times.
    """

    def __init__(self, log_file):
        self.logger = logging.getLogger("PileUpProject")
        self.logger.setLevel(logging.DEBUG)

        # Check if handlers already exist to prevent duplicate printing
        if not self.logger.handlers:
            # File Handler
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)

            # Console Handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)

            # Formatter
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            ch.setFormatter(formatter)
            fh.setFormatter(formatter)

            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

    def print_config(self, config):
        """Print configuration dict in a readable table format"""
        self.logger.info("=" * 40)
        self.logger.info(f"{'Key':<20} | {'Value':<20}")
        self.logger.info("-" * 40)
        for k, v in config.items():
            self.logger.info(f"{str(k):<20} | {str(v)}")
        self.logger.info("=" * 40)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)


def visualize_save(img_tensor, pred_density, gt_density, save_path):
    """
    Visualize and save the comparison between:
    [Original Image] | [GT Density] | [Pred Density]

    Args:
        img_tensor: Tensor [3, H, W] (Normalized)
        pred_density: Tensor [1, H, W] or [H, W]
        gt_density: Tensor [1, H, W] or [H, W]
        save_path: path to save the .png file
    """

    # 1. Un-normalize Image (ImageNet stats)
    # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    img = img_tensor.cpu().numpy()
    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).transpose(1, 2, 0).astype(np.uint8)  # [H, W, 3]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

    h, w = img.shape[:2]

    def process_map(density_map, target_h, target_w):
        if torch.is_tensor(density_map):
            density_map = density_map.detach().cpu().numpy()

        # Remove channel dim if exists
        if density_map.ndim == 3:
            density_map = density_map.squeeze(0)

        # Normalize to 0-255 for visualization
        if density_map.max() > 0:
            norm_map = 255 * (density_map / density_map.max())
        else:
            norm_map = density_map

        norm_map = np.clip(norm_map, 0, 255).astype(np.uint8)

        # Apply JET Colormap (Red = High Density)
        color_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

        # Resize to match original image size
        color_map = cv2.resize(color_map, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        return color_map

    # 2. Process GT and Pred
    vis_gt = process_map(gt_density, h, w)
    vis_pred = process_map(pred_density, h, w)

    # 3. Concatenate: Image | GT | Pred
    sep_line = np.ones((h, 10, 3), dtype=np.uint8) * 255  # White separator
    final_img = np.hstack((img, sep_line, vis_gt, sep_line, vis_pred))

    # 4. Add Text info
    gt_count = gt_density.sum()
    pred_count = pred_density.sum()

    cv2.putText(final_img, f"GT: {gt_count:.1f}", (w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(
        final_img, f"Pred: {pred_count:.1f}", (2 * w + 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )

    cv2.imwrite(save_path, final_img)
