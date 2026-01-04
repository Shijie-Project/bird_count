import logging
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch


class SaveHandle:
    """handle the number of"""

    def __init__(self, max_num):
        self.save_list = []
        self.max_num = max_num

    def append(self, save_path):
        assert len(self.save_list) <= self.max_num

        if len(self.save_list) == self.max_num:
            remove_path, self.save_list = self.save_list[0], self.save_list[1:]
            if os.path.exists(remove_path):
                os.remove(remove_path)

        self.save_list.append(save_path)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count

    def get_avg(self):
        return self.avg

    def get_count(self):
        return self.count


class Logger:
    def __init__(self, log_file):
        self.logger = logging.getLogger(log_file)
        self.logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    def print_config(self, config):
        """Print configuration of the model"""
        for k, v in config.items():
            self.logger.info(f"{k.ljust(15)}:\t{v}")

    def info(self, msg):
        self.logger.info(msg)


class SimulatedCamera:
    """
    Simulates a real-time camera stream using a video file.
    It blocks in .grab() to strictly match the target FPS, preventing
    the "fast-forward" effect when reading files.
    """

    def __init__(self, video_path, target_fps=None, loop=True):
        self.cap = cv2.VideoCapture(video_path)
        self.video_path = video_path
        self.loop = loop

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Determine FPS
        if target_fps is None:
            self.target_fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.target_fps = target_fps

        self.frame_interval = 1.0 / self.target_fps

        # Timing control
        self.frame_counter = 0
        self.start_time = time.perf_counter()

    def grab(self):
        """
        Simulate the blocking behavior of a real camera.
        Waits until the correct time to fetch the next frame.
        """
        # --- 1. Timing Logic (Drift-Free) ---
        # Calculate when the next frame *should* be captured relative to start
        target_time = self.start_time + (self.frame_counter * self.frame_interval)
        now = time.perf_counter()

        # If we are ahead of schedule, sleep precisely the difference
        time_to_wait = target_time - now
        if time_to_wait > 0:
            time.sleep(time_to_wait)
        else:
            # Optional: If we represent a live stream, and we are falling way behind,
            # strict timing might reset the baseline to avoid "catching up" quickly.
            # But for simple simulation, just proceeding is fine.
            pass

        # --- 2. Grab Frame ---
        ret = self.cap.grab()

        # --- 3. Handle Loop (End of File) ---
        if not ret and self.loop:
            # Rewind to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.start_time = time.perf_counter()  # Reset clock to avoid huge jump
            self.frame_counter = 0
            ret = self.cap.grab()

        if ret:
            self.frame_counter += 1

        return ret

    def retrieve(self, *args, **kwargs):
        """
        Just wraps the underlying retrieve.
        """
        return self.cap.retrieve(*args, **kwargs)

    def read(self):
        """
        Standard read: grab + retrieve
        """
        if self.grab():
            return self.retrieve()
        return False, None

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.cap.release()

    def set(self, propId, value):
        # Allow setting properties, though buffering/timeouts won't affect files much
        self.cap.set(propId, value)


@dataclass
class GrabberItem:
    """Data sent from FrameGrabber to Scheduler"""

    sid: int
    frame: np.ndarray  # uint8 array
    timestamp: float  # time.time()


@dataclass
class TaskItem:
    """Data sent from Scheduler to GPU Worker"""

    sids: list[int]
    frames: torch.Tensor  # CPU Tensor (B, H, W, 3)
    timestamps: list[float]  # For latency calculation


@dataclass
class ResultItem:
    """Data sent from GPU Worker back to Scheduler/Monitor"""

    sids: list[int]
    density_maps: torch.Tensor  # Float tensor (CPU)
    original_frames: torch.Tensor  # Uint8 tensor (CPU)
