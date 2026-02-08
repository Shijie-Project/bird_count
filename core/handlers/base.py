from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..inference_process import BatchInferenceResult, InferenceResult
from ..memory_manager import SharedMemoryClient


class BaseHandler(ABC):
    """
    Abstract Base Class for all result handlers.
    Ensures a consistent interface for the ResultConsumer.
    """

    def start(self):
        """
        Optional hook called when Consumer process starts.
        Use this to initialize resources (db connections, window creation, etc).
        """
        pass

    def handle_batch(self, batch_result: Optional[BatchInferenceResult], shm_client: SharedMemoryClient):
        """
        [New] Process a BATCH of results.

        Default Implementation:
        Iterates through the batch and calls the single-frame `handle` method for each item.
        Subclasses (like DBHandler) can override this to perform bulk operations (e.g., bulk insert).
        """
        for result in batch_result.results:
            # Lazy Loading: Only fetch the frame if the handler logic actually needs it.
            # Here we assume most handlers (Vis, VideoWriter) need the frame.
            # Direct Zero-Copy Access via numpy view
            raw_frame = shm_client.frames[result.stream_id, result.buffer_idx]

            self.handle(result, raw_frame)

    @abstractmethod
    def handle(self, result: InferenceResult, frame: np.ndarray):
        """
        Process a single frame result.

        Args:
            result: The DTO containing inference metadata (count, latency, stream_id).
            frame: The raw video frame [H, W, 3] (uint8) read from Shared Memory.
                   NOTE: This is a direct view into Shared Memory.
                   Do NOT modify it in-place unless you are sure.
                   Copy it (frame.copy()) if you need to draw on it.
        """
        pass

    def stop(self):
        """
        Optional hook called when Consumer process stops.
        Clean up resources here.
        """
        pass
