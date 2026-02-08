import logging
import multiprocessing as mp
import queue
import time

from .config import Config
from .handlers import BaseHandler
from .inference_engine import InferenceResult
from .shared_memory import BufferState, SharedMemoryClient, SharedMemoryConfig
from .utils import setup_logging


# Setup logger
logger = logging.getLogger(__name__)


class ResultConsumer(mp.Process):
    """
    The Final Destination.
    Consumes results, visualizes/saves data, and RELEASES the Shared Memory buffer.
    """

    def __init__(self, config: Config, shm_config: SharedMemoryConfig, result_queue: mp.Queue):
        super().__init__(name="ResultConsumer")
        self.config = config
        self.shm_config = shm_config
        self.result_queue = result_queue
        self._stop_event = mp.Event()
        self.handlers: list[BaseHandler] = []

    def register_handler(self, handler: BaseHandler):
        """Add a plugin to process results."""
        self.handlers.append(handler)

    def run(self):
        setup_logging(self.config.envs.debug)
        logger.info("Process Started.")

        # 1. Attach to Shared Memory (Client Mode)
        # We need this to (A) Read the frame for visualization, (B) Release the lock.
        shm_client = SharedMemoryClient(self.shm_config)
        try:
            shm_client.connect()
        except Exception as e:
            logger.critical(f"Failed to connect to SHM: {e}")
            return

        # Start Handlers
        for h in self.handlers:
            h.start()

        # 2. Consumption Loop
        while not self._stop_event.is_set():
            try:
                # Blocking get with timeout allows checking _stop_event regularly
                result: InferenceResult = self.result_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                # 3. Access Raw Data
                # Using the buffer_idx provided by InferenceEngine
                stream_id = result.stream_id
                buffer_idx = result.buffer_idx

                # Check consistency (Optional but good for debugging)
                current_state = shm_client.metadata[stream_id, buffer_idx]["state"]
                if current_state != BufferState.PROCESSING:
                    logger.warning(f"Frame {result.frame_idx} state mismatch! Got {current_state}")

                # Zero-copy read for visualization
                raw_frame = shm_client.frames[stream_id, buffer_idx]

                # 4. Execute Handlers (e.g., Draw, Save)
                for handler in self.handlers:
                    try:
                        handler.handle(result, raw_frame)
                    except Exception as e:
                        logger.error(f"Handler {type(handler).__name__} failed: {e}")

                # 5. CRITICAL: Release Buffer
                # This completes the cycle. The Producer can now overwrite this slot.
                shm_client.metadata[stream_id, buffer_idx]["state"] = BufferState.FREE

                # Log occasional latency stats
                if result.frame_idx % 100 == 0:
                    total_latency = (time.time() - result.timestamp) * 1000
                    logger.info(
                        f"Stream {stream_id} processed. Count: {result.count:.1f}. E2E Latency: {total_latency:.1f}ms"
                    )

            except Exception as e:
                logger.error(f"Error processing result: {e}")
                # Even if visualization fails, we MUST try to free the buffer
                # otherwise we leak memory slots.
                try:
                    shm_client.metadata[result.stream_id, result.buffer_idx]["state"] = BufferState.FREE
                except Exception:
                    pass

        # 6. Cleanup
        for h in self.handlers:
            h.stop()

        shm_client.close()
        logger.info("Process Exited.")

    def stop(self):
        self._stop_event.set()
