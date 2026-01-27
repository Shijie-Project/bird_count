import logging
from dataclasses import dataclass, field
from pprint import pformat
from typing import Literal, Optional

import numpy as np
import torch


logger = logging.getLogger(__name__)


@dataclass
class Config:
    # Model Settings
    MODEL_PATH: str = "./ckpts/shufflenet_best_model_214800.pth"
    INPUT_H: int = 512
    INPUT_W: int = 512

    # Hardware Settings
    AVAILABLE_GPUS: tuple[int, ...] = tuple(range(torch.cuda.device_count()))
    NUM_WORKERS_PER_GPU: int = 1

    # Sources
    SOURCE: Literal["video", "rtsp", "camera"] = field(default="camera")
    VIDEO_PATH: tuple[str, ...] = ("./data/bird_count_demo.mp4",)
    RTSP_URL: tuple[str, ...] = ("rtsp://127.0.0.1:8554/live/test",)
    CAMERA_ADD: tuple[str, ...] = (
        "http://root:root@138.25.209.105/mjpg/1/video.mjpg",
        "http://root:root@138.25.209.109/mjpg/1/video.mjpg",
        "http://root:root@138.25.209.111/mjpg/1/video.mjpg",
        "http://root:root@138.25.209.112/mjpg/1/video.mjpg",
        "http://root:root@138.25.209.113/mjpg/1/video.mjpg",
        "http://root:root@138.25.209.203/mjpg/1/video.mjpg",
    )

    # Monitor
    ENABLE_MONITOR: bool = True

    # Smart plugs
    TAPO_DEVICES = {
        "zone1": "192.168.0.185",
        "zone2": "192.168.0.198",
        "zone3": "192.168.0.102",
        "zone4": "192.168.0.195",
        "zone5": "192.168.0.130",
        "zone6": "192.168.0.164",
        "zone7": "192.168.0.110",
    }
    ENABLE_SMART_PLUG = True

    # Stream
    TARGET_FPS: float = 10.0
    NUM_STREAMS: int = 22
    RUNTIME_SECONDS: Optional[int] = None

    # Buffer
    NUM_BUFFER: int = 10  # Suggest using >= 5 for stability with locking

    # Auto-calculated fields
    frame_interval_s: float = field(init=False)
    stream_sources: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        # Clamp FPS
        if self.TARGET_FPS < 1 or self.TARGET_FPS > 60:
            logger.warning(f"Clamping Target FPS {self.TARGET_FPS} to [1, 60].")
        self.TARGET_FPS = max(1.0, min(60.0, self.TARGET_FPS))
        self.frame_interval_s = 1.0 / self.TARGET_FPS

        # Setup Sources
        source_maps = {
            "video": self.VIDEO_PATH,
            "rtsp": self.RTSP_URL,
            "camera": self.CAMERA_ADD,
        }
        sources = source_maps[self.SOURCE]
        # Round-robin fill
        if len(sources) < self.NUM_STREAMS:
            sources = sources * (self.NUM_STREAMS // len(sources) + 1)
        self.stream_sources = tuple(sources[: self.NUM_STREAMS])

        logger.info("--- Configuration Initialized ---")
        logger.info(f"FPS: {self.TARGET_FPS} (Interval: {self.frame_interval_s * 1000:.1f}ms)")
        logger.info(f"Streams: {self.NUM_STREAMS}")
        logger.info(f"GPUs: {len(self.AVAILABLE_GPUS)}")
        logger.info(f"Workers: {len(self.AVAILABLE_GPUS) * self.NUM_WORKERS_PER_GPU}")
        logger.info(f"Stream Source(s):\n{pformat(self.stream_sources)}")


@dataclass
class TaskItem:
    """Task payload sent to GPU Worker."""

    sids: list[int]
    buffer_indices: list[int]
    timestamps: list[float]
    dispatch_time: float


@dataclass
class ResultItem:
    """Result payload sent to ResultProcessor."""

    sids: list[int]
    buffer_indices: list[int]  # Needed for ResultProcessor to find original frame in SHM
    outputs: np.ndarray  # Lightweight density maps (Not blended images)
    timestamp: float
