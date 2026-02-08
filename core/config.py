import logging
import os
from pathlib import Path
from typing import Literal, Optional

import torch
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Setup Module Logger
logger = logging.getLogger(__name__)

# --- Constants & Defaults ---
DEFAULT_MODEL_PATH = Path("./ckpts/shufflenet_best_model_214800.pth")
DEFAULT_DEMO_VIDEO = Path("./data/bird_count_demo.mp4")


class SharedMemoryConfig(BaseModel):
    """
    Configuration for Shared Memory (SHM) management.
    Calculates buffer sizes strictly to avoid memory overflows.
    """

    name_prefix: str = "vstream_shm"

    # Industrial Best Practice: Double buffering is minimum.
    # For 20 streams + variable network latency, we recommend 4-6 buffers.
    num_buffers: int = Field(default=4, ge=2)

    # Resolution for Storage (What is saved in RAM)
    # Usually matches the Camera Source (e.g., 1080x720)
    height: int = 720
    width: int = 1080
    channels: int = 3

    @property
    def frame_bytes(self) -> int:
        return self.height * self.width * self.channels

    @property
    def block_size(self) -> int:
        """Total size in bytes for ONE stream's entire buffer ring."""
        return self.frame_bytes * self.num_buffers


class ModelConfig(BaseModel):
    """
    Configuration for the AI Inference Engine.
    """

    path: Path = DEFAULT_MODEL_PATH

    @field_validator("path")
    @classmethod
    def check_model_exists(cls, v):
        # We allow the file to be missing during dev/docker build,
        # but warn loudly. In prod, this should strictly fail.
        path = Path(v)
        if not path.exists():
            logger.warning(f"Model file not found at: {path}. Ensure it is mounted correctly.")
        return path


class ZoneConfig(BaseModel):
    """
    Represents a physical zone containing multiple IoT devices.
    Uses strict IP validation.
    """

    name: str
    cameras: list[str] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)
    smart_plugs: list[str] = Field(default_factory=list)

    threshold: int = 60


class SmartPlugAuthConfig(BaseModel):
    """Credentials for external device control."""

    email: str = Field(default_factory=lambda: os.getenv("TAPO_EMAIL", ""))
    password: str = Field(default_factory=lambda: os.getenv("TAPO_PASSWORD", ""))


class EnvSettings(BaseSettings):
    """
    Environment-level settings loaded from .env or system env vars.
    Controls the macro behavior of the application.
    """

    debug: bool = False

    # Target FPS for the GrabberProcess.
    # 10 FPS is sufficient for crowd counting, saving PCIe bandwidth.
    fps: int = 10

    # Global override for stream count (useful for debugging)
    # If None, it is calculated from the Zone Topology.
    num_streams: Optional[int] = None
    num_buffers: int = 4
    num_workers_per_gpu: int = 1

    cuda_device: str = "0"

    # Data Source: Real RTSP cameras or a looped Demo Video
    source_type: Literal["camera", "video"] = "camera"
    demo_video_path: Path = DEFAULT_DEMO_VIDEO

    enable_monitor: bool = True
    enable_smart_plug: bool = True
    enable_speaker: bool = True

    show_density_map: bool = True

    # Pydantic V2 config
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Config:
    """
    The Master Configuration Object.
    Aggregates all sub-configs into a single source of truth.
    """

    def __init__(self, envs: EnvSettings):
        self.envs = envs

        # 1. Initialize Sub-Configs
        self.model = ModelConfig()
        self.shm = SharedMemoryConfig(num_buffers=envs.num_buffers)
        self.plug_auth = SmartPlugAuthConfig()

        # 2. Hardware Detection
        self.device = self._detect_device(envs.cuda_device)

        # 3. Topology Loading (The "Business Logic" of where cameras are)
        # In a real industrial app, load this from `topology.yaml`
        self.zones = self._load_default_topology()

        # 4. Stream Reconciliation
        self.stream_sources = self._resolve_stream_sources()
        self.num_streams = len(self.stream_sources)
        self.num_buffers = self.shm.num_buffers

        self.fps = envs.fps
        self.num_workers_per_gpu = envs.num_workers_per_gpu

        self._log_configuration()

    def _detect_device(self, device_id: str) -> torch.device:
        if torch.cuda.is_available():
            d = f"cuda:{device_id}"
            gpu_name = torch.cuda.get_device_name(int(device_id))
            logger.info(f"Hardware Accelerator: {gpu_name} ({d})")
            return torch.device(d)
        logger.warning("Hardware Accelerator: CPU (Performance will be degraded)")
        return torch.device("cpu")

    def _load_default_topology(self) -> tuple[ZoneConfig, ...]:
        """
        Hardcoded default topology.
        Refactored to be cleaner, but preserves original data.
        """
        return (
            ZoneConfig(
                name="Zone_01_Right_Front",
                cameras=["138.25.209.125", "138.25.209.122", "138.25.209.127", "138.25.209.123", "138.25.209.129"],
                speakers=["138.25.209.236", "138.25.209.235", "138.25.209.231"],
            ),
            ZoneConfig(
                name="Zone_02_Left_Front",
                cameras=["138.25.209.112", "138.25.209.105", "138.25.209.130", "138.25.209.108", "138.25.209.126"],
                speakers=["138.25.209.239", "138.25.209.240", "138.25.209.237"],
            ),
            ZoneConfig(
                name="Zone_03_Right_Back",
                cameras=["138.25.209.131", "138.25.209.109", "138.25.209.124", "138.25.209.120", "138.25.209.132"],
                speakers=["138.25.209.234", "138.25.209.233", "138.25.209.232"],
            ),
            ZoneConfig(
                name="Zone_04_Left_Back",
                cameras=["138.25.209.104", "138.25.209.121", "138.25.209.113", "138.25.209.128", "138.25.209.111"],
                speakers=["138.25.209.238"],
            ),
            ZoneConfig(
                name="Zone_05_Right_Outer",
                cameras=["138.25.209.206"],
                speakers=["138.25.209.241"],
            ),
        )

    def _resolve_stream_sources(self) -> tuple[str, ...]:
        """
        Flattens Zone configs into a linear list of RTSP sources.
        Handles the 'Demo Mode' override.
        """
        sources = []
        for zone in self.zones:
            if not zone.cameras:
                continue

            if self.envs.source_type == "video":
                # Replicate the demo video path for every camera slot to simulate full load
                sources.extend([str(self.envs.demo_video_path)] * len(zone.cameras))
            else:
                # Add RTSP prefixes
                sources.extend([f"http://root:root@{ip}/mjpg/1/video.mjpg" for ip in zone.cameras])

        # Apply Global Limit (if set in .env)
        if self.envs.num_streams is not None:
            if self.envs.num_streams < len(sources):
                logger.warning(f"Limiting streams from {len(sources)} to {self.envs.num_streams}")
                return tuple(sources[: self.envs.num_streams])
            elif self.envs.num_streams > len(sources):
                logger.warning(
                    f"Requested {self.envs.num_streams} streams but only found {len(sources)}. "
                    "Running with available sources."
                )

        return tuple(sources)

    @property
    def ip2id(self) -> dict[str, int]:
        """
        Creates a lookup table: Camera IP -> Stream ID.
        Useful for reverse lookups when an external device (like a smart plug)
        needs to know which stream it belongs to based on IP.
        """
        mapping = {}
        current_stream_id = 0

        for zone in self.zones:
            for cam_ip in zone.cameras:
                mapping[cam_ip] = current_stream_id
                current_stream_id += 1

        return mapping

    @property
    def id2ip(self) -> dict[int, str]:
        """
        Creates a lookup table: Stream ID -> Camera IP.
        Useful for logging or displaying the source IP of a specific stream index.
        """
        mapping = {}
        current_stream_id = 0

        for zone in self.zones:
            for cam_ip in zone.cameras:
                mapping[current_stream_id] = cam_ip
                current_stream_id += 1
        return mapping

    @property
    def frame_interval(self) -> float:
        return 1.0 / self.envs.fps

    def _log_configuration(self):
        logger.info("=" * 40)
        logger.info("System Configuration Loaded")
        logger.info(f"Input Mode  : {self.envs.source_type}")
        logger.info(f"Stream Count: {self.num_streams}")
        logger.info(f"Target FPS  : {self.fps}")
        logger.info(f"SHM Buffers : {self.num_buffers}")
        logger.info(f"Num Workers  : {self.num_workers_per_gpu}")
        logger.info("-" * 40)

        logger.info(">- Zone Topology")

        if not hasattr(self, "zones") or not self.zones:
            logger.info("  [!] No Zones Configured (Global Monitor Mode)")
        else:
            for i, zone in enumerate(self.zones):
                z_name = getattr(zone, "name", f"Zone_{i}")
                z_cams = getattr(zone, "cameras", [])
                z_speakers = getattr(zone, "speakers", [])
                z_plugs = getattr(zone, "smart_plugs", "N/A")
                z_thresh = getattr(zone, "threshold", "Global")

                logger.info(f"  + Zone {i}: {z_name}")
                logger.info(f"    |-- Cameras   : {z_cams}")
                logger.info(f"    |-- Speakers   : {z_speakers}")
                logger.info(f"    |-- SmartPlugs : {z_plugs}")
                logger.info(f"    |-- Threshold : {z_thresh}")
                logger.info("")

        logger.info("=" * 40)

    @classmethod
    def load(cls, envs) -> "Config":
        """Factory method to load everything safely."""
        try:
            return cls(envs)
        except Exception as e:
            logger.critical(f"Failed to load configuration: {e}")
            raise
