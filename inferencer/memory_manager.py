import logging
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import time
from typing import Any, Optional

import numpy as np

from .config import Config


logger = logging.getLogger(__name__)


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
        self._linked = False

    def create(self):
        """Allocates shared memory and returns the shared memory name."""
        try:
            self.shm = shm.SharedMemory(create=True, size=self.nbytes, name=self.shm_name)
            self._linked = True

            # Initialize with zeros to avoid garbage data
            arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
            arr[:] = 0

            logger.info(f"[Memory] Frame Shared Memory allocated: {self.nbytes / 1024 / 1024:.2f} MB")
            return self.shm.name
        except Exception as e:
            logger.error(f"[Memory] Failed to create frame shared memory: {e}")
            raise

    def close(self):
        """Clean up resources."""
        if self.shm:
            self.shm.close()
            if self._linked:
                self.shm.unlink()
                self._linked = False
            logger.info("[Memory] Frame Shared Memory released.")


class SharedMetadataManager:
    """
    Manager for metadata synchronization (Buffer State & Latest Cursor).

    Buffer Meta Structure: [NUM_STREAMS, NUM_BUFFER, 2] (float64)
       - [..., 0]: State (0=Free, 1=Locked, 2=Ready)
       - [..., 1]: Timestamp
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # --- 1. Define Metadata Shape ---
        # [Streams, Buffers, Features(State, Timestamp)]
        self.shape = (cfg.NUM_STREAMS, cfg.NUM_BUFFER, 2)
        self.dtype = np.float64
        self.nbytes = int(np.prod(self.shape) * np.dtype(self.dtype).itemsize)

        self.shm_name = f"shm_meta_{time.time()}"

        # Resources
        self.shm: Optional[shm.SharedMemory] = None
        self.buffer_meta: Optional[np.ndarray] = None
        self.latest_cursor: Optional[Any] = None  # multiprocessing.Array

        self._linked = False

    def create(self) -> tuple[str, Any]:
        """
        Allocates shared metadata memory and initializes the cursor.
        Returns: (shm_name, latest_cursor_proxy)
        """
        try:
            # --- 2. Allocate SHM ---
            self.shm = shm.SharedMemory(create=True, size=self.nbytes, name=self.shm_name)
            self._linked = True

            # --- 3. Create View and Initialize ---
            self.buffer_meta = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
            self.buffer_meta.fill(0)  # Init all as Free (0)

            # --- 4. Create Latest Cursor ---
            # Tracks the last written buffer index for each stream.
            # -1 indicates no data has been written yet.
            self.latest_cursor = mp.Array("i", self.cfg.NUM_STREAMS)
            for i in range(self.cfg.NUM_STREAMS):
                self.latest_cursor[i] = -1

            logger.info(f"[Memory] Metadata Shared Memory allocated: {self.nbytes / 1024:.2f} KB")

            return self.shm.name, self.latest_cursor

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
