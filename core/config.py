import logging
import os
import sys
from pathlib import Path
from typing import Literal, Optional

import torch
import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Setup Module Logger
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised on invalid or incomplete system configuration."""


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

    path: Path = None
    type: str = "shufflenet"

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

    model_path: str = None
    image_height: int = 720
    image_width: int = 1080

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
    demo_video_path: Path = Path("../data/bird_count_demo.mp4")

    enable_monitor: bool = True
    enable_smart_plug: bool = True
    enable_speaker: bool = True

    monitor_only: bool = False

    tapo_email: str = None
    tapo_password: str = None

    enable_trigger_gui: bool = True

    # Hold-down delay (seconds): a stream's count must remain above its threshold
    # continuously for this long before the alert fires. Acts as a debounce/cooldown.
    alert_trigger_delay: float = 5.0

    show_density_map: bool = True

    # Audit log path (JSONL). Empty string disables auditing.
    audit_log_path: str = "logs/audit.jsonl"

    # Pydantic V2 config
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Config:
    """
    The Master Configuration Object.
    Aggregates all sub-configs into a single source of truth.
    """

    def __init__(self, envs: EnvSettings):
        self.envs = envs

        # 0. Up-front sanity checks (fail fast with actionable messages).
        self._validate_envs(envs)

        # 1. Initialize Sub-Configs
        # In monitor_only mode we never load a model, so model_path is allowed to be empty.
        if envs.model_path:
            self.model = ModelConfig(path=Path(envs.model_path))
        else:
            self.model = None
        self.shm = SharedMemoryConfig(num_buffers=envs.num_buffers, height=envs.image_height, width=envs.image_width)
        self.plug_auth = SmartPlugAuthConfig(email=envs.tapo_email, password=envs.tapo_password)

        # 2. Hardware Detection
        self.device = self._detect_device(envs.cuda_device)

        # 3. Topology Loading (The "Business Logic" of where cameras are)
        self.zones = self._load_default_topology()

        # 4. Stream Reconciliation
        self.stream_sources = self._resolve_stream_sources()
        self.num_streams = len(self.stream_sources)
        self.num_buffers = self.shm.num_buffers
        self.num_workers_per_gpu = envs.num_workers_per_gpu

        self.fps = envs.fps
        self.frame_interval = 1.0 / self.envs.fps

        self._log_configuration()

    @staticmethod
    def _validate_envs(envs: EnvSettings):
        """Pre-flight validation. Raises ConfigError with actionable messages."""
        # 1. model_path must be present unless we are in monitor-only mode.
        if not envs.monitor_only and not envs.model_path:
            raise ConfigError(
                "model_path is required when monitor_only=False. "
                "Set MODEL_PATH in your .env file (or environment) "
                "or run with MONITOR_ONLY=True for the no-inference dashboard."
            )

        # 2. If a model_path is provided, the file should exist (warn-not-fail to allow
        #    container builds where the weights are mounted later).
        if envs.model_path:
            mp_path = Path(envs.model_path)
            if not mp_path.exists():
                logger.warning(
                    f"model_path={mp_path} does not exist yet. Make sure it is mounted before launching the worker."
                )

        # 3. Demo video mode requires the demo file.
        if envs.source_type == "video":
            demo_path = Path(envs.demo_video_path)
            if not demo_path.exists():
                raise ConfigError(
                    f"source_type=video but demo_video_path does not exist: {demo_path}. "
                    "Provide a valid mp4 path or switch source_type to 'camera'."
                )

        # 4. Topology file must exist (loaded later by _load_default_topology).
        yaml_path = Path(__file__).parents[1] / "topology.yaml"
        if not yaml_path.exists():
            raise ConfigError(
                f"Topology file not found: {yaml_path}. "
                "Create topology.yaml at the project root with at least one zone."
            )

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
        Loads zone configuration from a YAML file. Existence already verified
        in _validate_envs(); here we focus on parse + schema errors.
        """
        yaml_path = Path(__file__).parents[1] / "topology.yaml"

        try:
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigError(f"Failed to parse {yaml_path}: {e}") from e

        zones_raw = data.get("zones", [])
        if not zones_raw:
            raise ConfigError(
                f"{yaml_path} has no zones defined under the 'zones:' key. "
                "Define at least one zone with cameras/speakers/smart_plugs."
            )

        try:
            zones = [ZoneConfig(**z) for z in zones_raw]
        except Exception as e:
            raise ConfigError(f"Invalid zone definition in {yaml_path}: {e}") from e

        logger.info(f"Successfully loaded {len(zones)} zones from {yaml_path}")
        return tuple(zones)

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
                sources.extend([f"https://root:root@{ip}/mjpg/1/video.mjpg" for ip in zone.cameras])  # noqa

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

    def _log_configuration(self):
        logger.info("=" * 40)
        logger.info("System Configuration Loaded")
        logger.info(f"Input Mode  : {self.envs.source_type}")
        logger.info(f"Stream Count: {self.num_streams}")
        logger.info(f"Target FPS  : {self.fps}")
        logger.info(f"SHM Buffers : {self.num_buffers}")
        logger.info(f"SHM Shapes  : {self.shm.height}x{self.shm.width}")
        logger.info(f"Num Workers : {self.num_workers_per_gpu}")
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

    @property
    def sid_to_zone(self):
        if hasattr(self, "_sid_to_zone"):
            return self._sid_to_zone

        sid = 0
        sid_to_zone = {}
        for zone in self.zones:
            for _ in zone.cameras:
                sid_to_zone[sid] = zone
                sid += 1
        self._sid_to_zone = sid_to_zone
        return sid_to_zone

    @property
    def sid_to_ip(self):
        if hasattr(self, "_sid_to_ip"):
            return self._sid_to_ip

        sid = 0
        sid_to_ip = {}
        for zone in self.zones:
            for ip in zone.cameras:
                sid_to_ip[sid] = ip
                sid += 1
        self._sid_to_ip = sid_to_ip
        return sid_to_ip

    @classmethod
    def load(cls, envs) -> "Config":
        """Factory method to load everything safely."""
        try:
            config = cls(envs)
            logger.info("Configuration Loaded Successfully.")
            return config
        except ConfigError as e:
            logger.critical(f"Configuration error: {e}")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"Failed to load configuration: {e}", exc_info=True)
            sys.exit(1)
