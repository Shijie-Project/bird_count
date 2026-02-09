from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..inference_process import BatchInferenceResult, InferenceResult
from ..memory_manager import SharedMemoryClient


class BaseHandler(ABC):
    """
    Abstract Base Class for all result handlers.
    Optimized for selective data fetching and bulk processing.
    """

    def __init__(self, name: Optional[str] = None, needs_frames: bool = True):
        # If False, we skip the expensive SHM frame indexing.
        self.name = name or self.__class__.__name__
        self.needs_frames = needs_frames

    def start(self):
        """Optional hook called when Consumer process starts."""
        pass

    def handle_batch(self, batch_result: BatchInferenceResult, shm_client: SharedMemoryClient):
        """
        Processes a BATCH of results.
        Optimized to avoid unnecessary SHM access if needs_frames is False.
        """
        if not batch_result.results:
            return

        # 1. Bulk check: If the handler doesn't need frames, pass None for all frames
        if not self.needs_frames:
            for result in batch_result.results:
                self.handle(result, None)
            return

        # 2. Sequential processing for handlers that need frames (e.g., Visualization, VideoWriter)
        # Using vectorized properties of batch_result for cleaner access
        for result in batch_result.results:
            try:
                # Direct Zero-Copy Access via numpy view
                # This is efficient, but still incurs a tiny overhead for array indexing
                raw_frame = shm_client.frames[result.stream_id, result.buffer_idx]
                self.handle(result, raw_frame)
            except Exception as e:
                # We log at the handler level to avoid crashing the whole ResultConsumer
                import logging

                logging.getLogger(__name__).error(f"[{self.name}] Error handling frame: {e}")

    @abstractmethod
    def handle(self, result: InferenceResult, frame: Optional[np.ndarray]):
        """
        Process a single frame result.

        Args:
            result: The DTO containing inference metadata (count, alert_flag, threshold, etc).
            frame: The raw video frame [H, W, 3] (uint8) or None if needs_frames=False.
                   NOTE: This is a direct view into Shared Memory.
        """
        pass

    def stop(self):
        """Optional hook called when Consumer process stops."""
        pass
