import logging
import multiprocessing as mp
import queue

from .config import Config
from .handlers import BaseHandler
from .inference_process import BatchInferenceResult
from .memory_manager import BufferState, SharedMemoryClient, SharedMemoryConfig
from .utils import setup_logging


# Setup logger
logger = logging.getLogger(__name__)


class ResultProcess(mp.Process):
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

        # 1. Attach to Shared Memory
        shm_client = SharedMemoryClient(self.shm_config)
        shm_client.connect()

        # Start Handlers
        for h in self.handlers:
            h.start()

        # 2. Consumption Loop
        while not self._stop_event.is_set():
            try:
                # [Batch Mode] Get the whole packet
                batch_packet: BatchInferenceResult = self.result_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            try:
                for handler in self.handlers:
                    try:
                        handler.handle_batch(batch_packet, shm_client)
                    except Exception as e:
                        logger.error(f"Handler {type(handler).__name__} batch error: {e}")

            finally:
                # --- 4. Critical: Batch Release ---
                for result in batch_packet.results:
                    try:
                        # Direct access is fastest
                        shm_client.metadata[result.stream_id, result.buffer_idx]["state"] = BufferState.FREE
                    except Exception:
                        pass  # Ignore errors during cleanup to ensure loop continues

        # 5. Cleanup
        for h in self.handlers:
            h.stop()

        shm_client.close()
        logger.info("Process Exited.")

    def stop(self):
        self._stop_event.set()
