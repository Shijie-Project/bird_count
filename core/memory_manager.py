import logging
import multiprocessing.shared_memory as shm
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np


logger = logging.getLogger(__name__)


class BufferState(IntEnum):
    """
    Strict State Machine for Ring Buffer Slots.
    """

    FREE = 0  # Slot is empty, Grabber can write.
    WRITING = 1  # Grabber is currently writing (Lock).
    READY = 2  # Data is written, ready for Inference.
    READING = 3  # Inference is reading this slot (Lock).


# OPTIMIZATION: Manually aligned to 32 bytes to fit CPU cache lines perfectly.
# 1 (uint8) + 3 (padding) + 4 (int32) + 4 (int32) + 4 (padding) + 8 (int64) + 8 (float64) = 32 bytes.
METADATA_DTYPE = np.dtype(
    [
        ("state", np.uint8),
        ("stream_id", np.int32),
        ("buffer_idx", np.int32),
        ("frame_idx", np.int64),
        ("timestamp", np.float64),
    ],
    align=True,  # Ensures fields are aligned for faster CPU access
)


@dataclass(frozen=True, slots=True)
class SharedMemoryConfig:
    """
    Immutable configuration with slots for minimal memory footprint.
    """

    name: str
    meta_name: str
    shape: tuple
    dtype: str
    meta_shape: tuple

    @property
    def frame_size_mb(self) -> float:
        return int(np.prod(self.shape) * np.dtype(self.dtype).itemsize) / (1024 * 1024)

    @property
    def meta_size_kb(self) -> float:
        return int(np.prod(self.meta_shape) * METADATA_DTYPE.itemsize) / 1024.0


class SharedMemoryManager:
    """
    Master process orchestrator for Shared Memory lifecycle.
    """

    def __init__(
        self,
        name_prefix: str,
        num_streams: int,
        num_buffers: int,
        resolution: tuple,
        channels: int = 3,
        dtype: str = "uint8",
        name: str = "MemoryManager",
    ):
        self.name = name

        self.num_streams = num_streams
        self.num_buffers = num_buffers
        self.height, self.width = resolution
        self.channels = channels

        self.shm_name = f"{name_prefix}_frames"
        self.meta_name = f"{name_prefix}_meta"
        self.frame_dtype = dtype

        # Shape: [Stream, Buffer, H, W, C] -> N-B-H-W-C (Optimal for BGR-to-RGB flip)
        self.shape = (num_streams, num_buffers, self.height, self.width, channels)
        self.meta_shape = (num_streams, num_buffers)

        self._shm: Optional[shm.SharedMemory] = None
        self._meta_shm: Optional[shm.SharedMemory] = None

        try:
            self._allocate_memory()
            self._initialize_metadata()
            config = self.get_config()
            logger.info(f"[{self.name}] Frames: {config.frame_size_mb:.2f} MB | Meta: {config.meta_size_kb:.2f} KB")
        except Exception as e:
            logger.error(f"[{self.name}] Allocation failed: {e}")
            self.cleanup()
            raise

    def _allocate_memory(self):
        """Creates blocks, unlinking stale ones if needed."""
        self._unlink_if_exists(self.shm_name)
        self._unlink_if_exists(self.meta_name)

        frame_size = int(np.prod(self.shape) * np.dtype(self.frame_dtype).itemsize)
        self._shm = shm.SharedMemory(name=self.shm_name, create=True, size=frame_size)

        meta_size = int(np.prod(self.meta_shape) * METADATA_DTYPE.itemsize)
        self._meta_shm = shm.SharedMemory(name=self.meta_name, create=True, size=meta_size)

    def _initialize_metadata(self):
        """Vectorized initialization of metadata block."""
        meta_array = np.ndarray(self.meta_shape, dtype=METADATA_DTYPE, buffer=self._meta_shm.buf)
        meta_array["state"] = BufferState.FREE
        meta_array["stream_id"] = -1
        meta_array["buffer_idx"] = -1
        meta_array["frame_idx"] = 0
        meta_array["timestamp"] = 0.0

    def get_config(self) -> SharedMemoryConfig:
        return SharedMemoryConfig(
            name=self.shm_name,
            meta_name=self.meta_name,
            shape=self.shape,
            dtype=self.frame_dtype,
            meta_shape=self.meta_shape,
        )

    def _unlink_if_exists(self, name: str):
        try:
            temp = shm.SharedMemory(name=name)
            temp.unlink()
            temp.close()
            logger.warning(f"[{self.name}] Cleaned orphaned buffer: {name}")
        except FileNotFoundError:
            pass

    def cleanup(self):
        """Mandatory cleanup for Master process."""
        for resource, label in [(self._shm, "Frame"), (self._meta_shm, "Meta")]:
            if resource:
                try:
                    resource.close()
                    resource.unlink()
                    logger.info(f"[{self.name}] {label} buffer unlinked.")
                except Exception as e:
                    logger.error(f"[{self.name}] Error cleaning {label}: {e}")


class SharedMemoryClient:
    """
    Optimized SHM Client for Workers.
    Caches per-stream views to eliminate indexing overhead in hot loops.
    """

    def __init__(self, config: SharedMemoryConfig, name: str = "MemoryClient"):
        self.name = name

        self.config = config
        self.shm: Optional[shm.SharedMemory] = None
        self.meta_shm: Optional[shm.SharedMemory] = None
        self.frames: Optional[np.ndarray] = None
        self.metadata: Optional[np.ndarray] = None

        # OPTIMIZATION: Pre-cached views for each stream
        self.stream_frames: list[np.ndarray] = []
        self.stream_metadata: list[np.ndarray] = []

    def connect(self):
        """Attaches to SHM and caches optimized views."""
        try:
            # 1. Attach Frame Buffer
            self.shm = shm.SharedMemory(name=self.config.name)
            self.frames = np.ndarray(
                self.config.shape,
                dtype=self.config.dtype,
                buffer=self.shm.buf,
                order="C",  # Explicitly enforce C-contiguous order
            )

            # 2. Attach Metadata Buffer
            self.meta_shm = shm.SharedMemory(name=self.config.meta_name)
            self.metadata = np.ndarray(self.config.meta_shape, dtype=METADATA_DTYPE, buffer=self.meta_shm.buf)

            # 3. Cache per-stream views
            # This avoids creating a new Python view object inside the frame loop
            num_streams = self.config.meta_shape[0]
            self.stream_frames = [self.frames[i] for i in range(num_streams)]
            self.stream_metadata = [self.metadata[i] for i in range(num_streams)]

            logger.debug(f"[{self.name}] Connected to {self.config.name}")
        except Exception as e:
            logger.critical(f"[{self.name}] Connection failed: {e}")
            raise

    def disconnect(self):
        """Detaches from memory blocks."""
        if self.shm:
            self.shm.close()
        if self.meta_shm:
            self.meta_shm.close()
        logger.debug(f"[{self.name}] Disconnected.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
