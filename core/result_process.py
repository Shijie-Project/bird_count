import logging
import multiprocessing as mp
import queue

from .config import Config
from .handlers import BaseHandler
from .inference_process import BatchInferenceResult
from .memory_manager import BufferState, SharedMemoryClient, SharedMemoryConfig
from .utils import setup_logging


logger = logging.getLogger(__name__)


class ResultProcess(mp.Process):
    """
    High-throughput Result Consumer.
    Evaluates thresholds, triggers alerts, and manages the SHM lifecycle.
    """

    def __init__(self, config: Config, shm_config: SharedMemoryConfig, result_queue: mp.Queue):
        super().__init__(name="ResultConsumer")
        self.config = config
        self.shm_config = shm_config
        self.result_queue = result_queue
        self._stop_event = mp.Event()
        self.handlers: list[BaseHandler] = []

        # O(1) Threshold Mapping
        self.stream_thresholds: dict[int, float] = {
            sid: zone.threshold for sid, zone in self.config.sid_to_zone.items()
        }

        self.manual_override_streams: set[int] = set()

    def register_handler(self, handler: BaseHandler):
        """Registers a plugin to process finalized results."""
        self.handlers.append(handler)

    def _handle_manual_trigger(self, stream_id: int):
        """Callback for GUI to inject manual alerts."""
        if stream_id in self.manual_override_streams:
            self.manual_override_streams.remove(stream_id)
            logger.info(f"[{self.name}] Manual Hijack DISABLED for Stream {stream_id}.")
        else:
            self.manual_override_streams.add(stream_id)
            logger.info(f"[{self.name}] Manual Hijack ENABLED for Stream {stream_id}. Forcing all alerts to True.")

    def _evaluate_alerts_inplace(self, batch_packet: BatchInferenceResult):
        """
        Fast in-place alert evaluation.
        Avoids creating new objects to reduce GC pressure.
        """
        for result in batch_packet.results:
            # 1. Check if this stream is currently being hijacked
            if result.stream_id in self.manual_override_streams:
                result.alert_flag = True
                # logger.debug(f"Stream {result.stream_id} is hijacked. Forcing alert_flag=True")
                continue  # Skip normal threshold logic for hijacked streams

            # 2. Normal Threshold Logic
            threshold = self.stream_thresholds.get(result.stream_id, 0.0)
            result.alert_flag = result.count >= threshold

    def run(self):
        """Main consumption loop."""
        setup_logging(self.config.envs.debug)
        logger.info(f"[{self.name}] Process Started.")

        # 1. Resource Setup
        shm_client = SharedMemoryClient(self.shm_config)
        shm_client.connect()

        for h in self.handlers:
            h.start()

        # 2. Optional GUI Setup
        gui = None
        if self.config.envs.enable_trigger_gui:
            try:
                from .gui import ManualTriggerGUI

                # The callback now toggles the 'hijack' state in this process
                gui = ManualTriggerGUI(self.config, self._handle_manual_trigger)
                gui.setup()
            except ImportError:
                logger.warning(f"[{self.name}] GUI requested but .gui module not found.")

        # 3. Consumption Loop
        while not self._stop_event.is_set():
            if gui:
                gui.update()

            try:
                # Use a small timeout to allow checking the stop_event
                batch_packet: BatchInferenceResult = self.result_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            try:
                # A. Logical Evaluation (In-place)
                self._evaluate_alerts_inplace(batch_packet)

                # B. Plugin Dispatch
                for handler in self.handlers:
                    try:
                        handler.handle_batch(batch_packet, shm_client)
                    except Exception as e:
                        logger.error(f"[{self.name}] Handler {type(handler).__name__} error: {e}")

            except Exception as e:
                logger.error(f"[{self.name}] Unexpected loop error: {e}")

            finally:
                # C. CRITICAL: Atomic SHM Release
                # We extract indices only once for performance
                sids = batch_packet.stream_ids
                b_idxs = batch_packet.buffer_indices

                # Check for -1 (manual triggers) to avoid indexing errors
                if b_idxs and b_idxs[0] != -1:
                    shm_client.metadata[sids, b_idxs]["state"] = BufferState.FREE

        # 4. Cleanup Sequence
        if gui:
            gui.destroy()

        for h in self.handlers:
            h.stop()

        shm_client.disconnect()
        logger.info(f"[{self.name}] Stopped.")

    def stop(self):
        self._stop_event.set()
