import logging
from dataclasses import dataclass, field
from pprint import pformat
from typing import Any, Literal, Optional

import numpy as np
import torch


logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    path: str = "./ckpts/shufflenet_best_model_214800.pth"
    input_h: int = 512
    input_w: int = 512


@dataclass
class HardwareConfig:
    available_gpus: tuple[int, ...] = field(default_factory=lambda: tuple(range(torch.cuda.device_count())))
    num_workers_per_gpu: int = 1

    @property
    def total_workers(self) -> int:
        return len(self.available_gpus) * self.num_workers_per_gpu


@dataclass
class SourceConfig:
    mode: Literal["video", "rtsp", "camera"] = "camera"

    video_paths: tuple[str, ...] = ("./data/bird_count_demo.mp4",)
    rtsp_urls: tuple[str, ...] = ("rtsp://127.0.0.1:8554/live/test",)
    camera_addresses: tuple[str, ...] = (
        "138.25.209.105",
        "138.25.209.106",
        "138.25.209.108",
        "138.25.209.109",
        "138.25.209.111",
        "138.25.209.112",
        "138.25.209.113",
        "138.25.209.120",
        "138.25.209.121",
        "138.25.209.122",
        "138.25.209.123",
        "138.25.209.124",
        "138.25.209.125",
        "138.25.209.126",
        "138.25.209.127",
        "138.25.209.128",
        "138.25.209.129",
        "138.25.209.130",
        "138.25.209.131",
        "138.25.209.132",
        "138.25.209.203",
        "138.25.209.206",
    )

    def __post_init__(self):
        self.camera_addresses = tuple(f"https://root:root@{ip}/mjpg/1/video.mjpg" for ip in self.camera_addresses)

    def get_raw_sources(self) -> tuple[str, ...]:
        source_maps = {
            "video": self.video_paths,
            "rtsp": self.rtsp_urls,
            "camera": self.camera_addresses,
        }
        return source_maps[self.mode]


@dataclass
class StreamConfig:
    target_fps: float = 10.0
    num_streams: int = 22
    runtime_seconds: Optional[int] = None

    # Buffer settings
    num_buffer: int = 4
    enable_monitor: bool = True

    # Auto-calculated
    frame_interval_s: float = field(init=False)

    def __post_init__(self):
        # Clamp FPS Logic
        if self.target_fps < 1 or self.target_fps > 60:
            logger.warning(f"Clamping Target FPS {self.target_fps} to [1, 60].")
        self.target_fps = max(1.0, min(60.0, self.target_fps))
        self.frame_interval_s = 1.0 / self.target_fps


@dataclass
class SmartPlugConfig:
    enable: bool = True
    email: str = "allenliu0416@163.com"
    password: str = "LTX4947978"
    timeout_seconds: int = 5
    alert_threshold: int = 50
    device_maps: dict[str, str] = field(
        default_factory=lambda: {
            "192.168.0.185": [0, 1, 2, 3],
            "192.168.0.198": [4, 5, 6, 7],
            "192.168.0.102": [8, 9, 10, 11],
            "192.168.0.195": [12, 13, 14, 15],
            "192.168.0.130": [16, 17, 18, 19],
            "192.168.0.164": [20, 21, 22, 23],
            "192.168.0.110": [24, 25, 26, 27],
        }
    )


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    smart_plug: SmartPlugConfig = field(default_factory=SmartPlugConfig)

    active_stream_sources: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        raw_sources = self.source.get_raw_sources()
        required_streams = self.stream.num_streams

        if len(raw_sources) < required_streams:
            raw_sources = raw_sources * (required_streams // len(raw_sources) + 1)

        self.active_stream_sources = tuple(raw_sources[:required_streams])

        self._log_initialization()

    def _log_initialization(self):
        logger.info("--- Configuration Initialized ---")
        logger.info(f"FPS: {self.stream.target_fps} (Interval: {self.stream.frame_interval_s * 1000:.1f}ms)")
        logger.info(f"Streams: {self.stream.num_streams}")
        logger.info(f"GPUs: {len(self.hardware.available_gpus)}")
        logger.info(f"Workers: {self.hardware.total_workers}")
        logger.info(f"Active Source(s):\n{pformat(self.active_stream_sources)}")

    @classmethod
    def from_settings(cls, envs: Any) -> "Config":
        """
        Build from pydantic environment variables。
        """
        gpu_str = getattr(envs, "cuda_visible_devices", "")
        if gpu_str:
            gpus = tuple(int(x) for x in gpu_str.split(",") if x.strip())
        else:
            gpus = (0,)

        hw_cfg = HardwareConfig(available_gpus=gpus, num_workers_per_gpu=envs.num_workers_per_gpu)

        stream_cfg = StreamConfig(
            target_fps=envs.fps,
            num_streams=envs.num_streams,
            num_buffer=envs.num_buffer,
        )

        source_cfg = SourceConfig(mode=envs.source)

        return cls(hardware=hw_cfg, stream=stream_cfg, source=source_cfg)


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
