import logging
import multiprocessing.shared_memory as shm
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np


# Setup logger
logger = logging.getLogger(__name__)


class BufferState(IntEnum):
    """
    Strict State Machine for Ring Buffer Slots.
    Flow: FREE -> WRITING -> READY -> READING -> FREE
    """

    FREE = 0  # Slot is empty, Grabber can write here.
    WRITING = 1  # Grabber is currently writing (Lock).
    READY = 2  # Data is written, ready for Inference.
    READING = 3  # Inference is reading this slot (Lock).
    # Note: After Inference, the slot goes back to FREE immediately
    # if no strict sequencing is required for the Consumer,
    # OR flows to CONSUMING if we want to hold the raw frame for the consumer.
    # For this design, we assume Inference extracts data and releases buffer to FREE
    # or keeps it READING until Consumer signals.
    # Let's align with: Grabber -> Inference -> (Result Queue) -> Consumer.
    # So SHM is mostly for Grabber -> Inference.


# Define the exact memory layout for metadata to ensure alignment across processes.
# Structure: [state (u8), stream_id (i32), buffer_idx (i32), timestamp (f64)]
METADATA_DTYPE = np.dtype(
    [
        ("state", np.uint8),
        ("stream_id", np.int32),
        ("buffer_idx", np.int32),
        ("frame_idx", np.int64),
        ("timestamp", np.float64),
    ]
)


@dataclass(frozen=True)
class SharedMemoryConfig:
    """
    Immutable configuration object passed to child processes
    to reconstruct shared memory hooks.
    """

    name: str  # Name of the frame buffer block
    meta_name: str  # Name of the metadata block
    shape: tuple[int, ...]  # (num_streams, num_buffers, height, width, channels)
    dtype: str  # e.g., 'uint8'
    meta_shape: tuple[int, ...]  # (num_streams, num_buffers)

    @property
    def frame_size_mb(self) -> float:
        """Calculate total size of the frame buffer in Megabytes (MB)."""
        return int(np.prod(self.shape) * np.dtype(self.dtype).itemsize) / (1024 * 1024)

    @property
    def meta_size_kb(self) -> float:
        """Calculate total size of the metadata buffer in Kilobytes (KB)."""
        return int(np.prod(self.meta_shape) * METADATA_DTYPE.itemsize) / 1024.0


class SharedMemoryManager:
    """
    Manages the lifecycle of Shared Memory blocks.
    Implements RAII pattern (create on init, cleanup on close).
    """

    def __init__(
        self,
        name_prefix: str,
        num_streams: int,
        num_buffers: int,
        resolution: tuple[int, int],
        channels: int = 3,
        dtype: str = "uint8",
    ):
        """
        Initialize and allocate Shared Memory.
        """
        self.num_streams = num_streams
        self.num_buffers = num_buffers
        self.height, self.width = resolution
        self.channels = channels

        # 1. Define Names
        self.shm_name = f"{name_prefix}_frames"
        self.meta_name = f"{name_prefix}_meta"
        self.frame_dtype = dtype

        # 2. Define Shapes
        # Shape: [StreamID, BufferIndex, H, W, C]
        self.shape = (num_streams, num_buffers, self.height, self.width, channels)
        self.meta_shape = (num_streams, num_buffers)

        # 3. Allocation (Master Process Only)
        self._shm: Optional[shm.SharedMemory] = None
        self._meta_shm: Optional[shm.SharedMemory] = None
        self._config: Optional[SharedMemoryConfig] = None

        try:
            self._allocate_memory()
            self._initialize_metadata()
            logger.info(f"Allocated {self._config.frame_size_mb:.2f} MB for frames. (Shape: {self.shape})")
            logger.info(f"Allocated {self._config.meta_size_kb:.2f} KB for metadata. (Shape: {self.meta_shape})")
        except Exception as e:
            logger.error(f"Allocation failed: {e}")
            self.cleanup()  # Attempt cleanup if partial fail
            raise e

    def _allocate_memory(self):
        """Creates the shared memory blocks. Unlinks old ones if they exist."""
        # Clean up potential leftovers from previous crashes
        self._unlink_if_exists(self.shm_name)
        self._unlink_if_exists(self.meta_name)

        # Create Frame Buffer
        size = int(np.prod(self.shape) * np.dtype(self.frame_dtype).itemsize)
        self._shm = shm.SharedMemory(name=self.shm_name, create=True, size=size)

        # Create Metadata Buffer
        meta_size = int(np.prod(self.meta_shape) * METADATA_DTYPE.itemsize)
        self._meta_shm = shm.SharedMemory(name=self.meta_name, create=True, size=meta_size)

        # Create Config Object
        self._config = self.get_config()

    def _initialize_metadata(self):
        """Zero out the metadata and set states to FREE."""
        meta_array = np.ndarray(self.meta_shape, dtype=METADATA_DTYPE, buffer=self._meta_shm.buf)
        meta_array["state"] = BufferState.FREE
        meta_array["stream_id"] = -1
        meta_array["buffer_idx"] = -1
        meta_array["frame_idx"] = 0
        meta_array["timestamp"] = 0.0

    def get_config(self) -> SharedMemoryConfig:
        """Returns the config object needed for child processes to attach."""
        return SharedMemoryConfig(
            name=self.shm_name,
            meta_name=self.meta_name,
            shape=self.shape,
            dtype=self.frame_dtype,
            meta_shape=self.meta_shape,
        )

    @staticmethod
    def _unlink_if_exists(name: str):
        """Helper to unlink shared memory if it exists (Linux/Unix)."""
        try:
            temp = shm.SharedMemory(name=name)
            temp.unlink()
            temp.close()
            logger.warning(f"Found and unlinked orphaned buffer: {name}")
        except FileNotFoundError:
            pass  # Good, it doesn't exist
        except Exception as e:
            logger.warning(f"Warning during unlink of {name}: {e}")

    def cleanup(self):
        """Master process calls this to destroy memory blocks."""
        if self._shm:
            try:
                self._shm.close()
                self._shm.unlink()
                logger.info("Frame buffer unlinked.")
            except Exception as e:
                logger.error(f"Error cleaning frames: {e}")

        if self._meta_shm:
            try:
                self._meta_shm.close()
                self._meta_shm.unlink()
                logger.info("Metadata buffer unlinked.")
            except Exception as e:
                logger.error(f"Error cleaning meta: {e}")


class SharedMemoryClient:
    """
    Helper for Child Processes (Producer/Consumer) to attach to existing SHM.
    """

    def __init__(self, config: SharedMemoryConfig):
        self.config = config
        self.shm: Optional[shm.SharedMemory] = None
        self.meta_shm: Optional[shm.SharedMemory] = None
        self.frames: Optional[np.ndarray] = None
        self.metadata: Optional[np.ndarray] = None

    def connect(self):
        """Attach to the shared memory blocks."""
        try:
            self.shm = shm.SharedMemory(name=self.config.name)
            self.frames = np.ndarray(self.config.shape, dtype=self.config.dtype, buffer=self.shm.buf)

            self.meta_shm = shm.SharedMemory(name=self.config.meta_name)
            self.metadata = np.ndarray(self.config.meta_shape, dtype=METADATA_DTYPE, buffer=self.meta_shm.buf)

            logger.debug(f"Connected to {self.config.name}")
        except Exception as e:
            logger.critical(f"Failed to connect: {e}")
            raise e

    def close(self):
        """Detach without unlinking."""
        if self.shm:
            self.shm.close()
        if self.meta_shm:
            self.meta_shm.close()
