import logging
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import Config


logger = logging.getLogger(__name__)


@dataclass
class SharedMemoryInfo:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype


class SharedMemoryManager:
    """
    Manager for the raw video frame data (Big Image Data).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Structure: [NUM_STREAMS, NUM_BUFFER, H, W, 3] (uint8)
        self.shape = (cfg.NUM_STREAMS, cfg.NUM_BUFFER, cfg.INPUT_H, cfg.INPUT_W, 3)
        self.dtype = np.uint8
        self.nbytes = int(np.prod(self.shape) * np.dtype(self.dtype).itemsize)

        self.shm_name = f"shm_frames_{time.time()}"

        self.shm: Optional[shm.SharedMemory] = None
        self.buffer: Optional[np.ndarray] = None
        self._linked = False

    def create(self) -> SharedMemoryInfo:
        """Allocates shared memory and returns the shared memory name."""
        try:
            self.shm = shm.SharedMemory(create=True, size=self.nbytes, name=self.shm_name)
            self._linked = True

            # Initialize with zeros to avoid garbage data
            self.buffer = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
            self.buffer.fill(0)

            logger.info(f"[Memory] Frame Shared Memory allocated: {self.nbytes / 1024 / 1024:.2f} MB")
            return SharedMemoryInfo(self.shm.name, self.shape, self.dtype)

        except Exception as e:
            logger.error(f"[Memory] Failed to create frame shared memory: {e}")
            raise

    def close(self):
        """Clean up resources."""
        if self.shm:
            try:
                self.shm.close()
                if self._linked:
                    self.shm.unlink()
                    self._linked = False
                logger.info("[Memory] Frame Shared Memory released.")
            except Exception as e:
                logger.warning(f"[Memory] Error closing frame SHM: {e}")


class SharedMetadataManager:
    """
    Manager for metadata synchronization (Buffer State & Latest Cursor).

    Buffer Meta Structure: [NUM_STREAMS, NUM_BUFFER, 2] (float64)
       - [..., 0]: State (0=Free, 1=Locked, 2=Ready)
       - [..., 1]: Timestamp
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Structure: [Streams, NUM_BUFFER, Features(State, Timestamp)]
        self.shape = (cfg.NUM_STREAMS, cfg.NUM_BUFFER, 2)
        self.dtype = np.float64
        self.nbytes = int(np.prod(self.shape) * np.dtype(self.dtype).itemsize)

        self.shm_name = f"shm_meta_{time.time()}"

        self.shm: Optional[shm.SharedMemory] = None
        self.buffer: Optional[np.ndarray] = None
        self.latest_cursor = None
        self._linked = False

    def create(self) -> SharedMemoryInfo:
        """
        Allocates shared metadata memory and initializes the cursor.
        """
        try:
            self.shm = shm.SharedMemory(create=True, size=self.nbytes, name=self.shm_name)
            self._linked = True

            self.buffer = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
            self.buffer.fill(0)  # Init all as Free (0)

            # Create latest cursor (tracks the last written buffer index for each stream)
            # Init -1 means no data written yet
            self.latest_cursor = mp.Array("i", self.cfg.NUM_STREAMS)
            for i in range(self.cfg.NUM_STREAMS):
                self.latest_cursor[i] = 0

            logger.info(f"[Memory] Metadata Shared Memory allocated: {self.nbytes / 1024:.2f} KB")
            return SharedMemoryInfo(self.shm.name, self.shape, self.dtype), self.latest_cursor

        except Exception as e:
            logger.error(f"[Memory] Failed to create metadata shared memory: {e}")
            self.close()  # Cleanup if partial failure
            raise

    def close(self):
        """Clean up resources."""
        if self.shm:
            try:
                self.shm.close()
                if self._linked:
                    self.shm.unlink()
                    self._linked = False
                logger.info("[Memory] Metadata Shared Memory released.")
            except Exception as e:
                logger.warning(f"[Memory] Error closing metadata SHM: {e}")
