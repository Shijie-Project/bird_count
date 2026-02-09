import logging
import multiprocessing as mp
import queue
import time
from dataclasses import replace
from typing import Optional

from .config import Config
from .handlers import BaseHandler, init_handlers
from .inference_process import BatchInferenceResult, InferenceResult
from .memory_manager import BufferState, SharedMemoryClient, SharedMemoryConfig
from .utils import setup_logging


logger = logging.getLogger(__name__)


class ResultProcess(mp.Process):
    """
    The Final Destination.
    Consumes results, evaluates business logic (thresholds),
    and dispatches processed data to handlers.
    """

    def __init__(self, config: Config, shm_config: SharedMemoryConfig, result_queue: mp.Queue):
        super().__init__(name="ResultConsumer")
        self.config = config
        self.shm_config = shm_config
        self.result_queue = result_queue
        self._stop_event = mp.Event()
        self.handlers: list[BaseHandler] = []

        # Pre-calculate threshold mapping: stream_id -> threshold
        # This ensures O(1) lookup performance during high-speed inference
        self.stream_thresholds = {sid: zone.threshold for sid, zone in self.config.sid_to_zone.items()}

    def register_handlers(self):
        """Add a plugin to process results."""
        handlers = init_handlers(self.config)
        for handler in handlers:
            self.handlers.append(handler)

    def _handle_manual_trigger(self, stream_id: int):
        """Callback for the GUI class to inject a mock alert."""
        logger.info(f"[ResultProcess] Manual trigger activated for stream {stream_id}")

        # Create a mock result that bypasses normal threshold checks
        mock_res = InferenceResult(
            stream_id=stream_id,
            buffer_idx=-1,  # -1 indicates no physical SHM buffer
            frame_idx=-1,
            timestamp=time.time(),
            count=99.0,  # Arbitrary high count
            latency=0.0,
            alert_flag=True,  # Force the alert flag to True
        )

        batch = BatchInferenceResult(results=[mock_res])

        # Dispatch to all handlers
        for handler in self.handlers:
            try:
                # Note: shm_client is passed as None because there is no frame for manual trigger
                handler.handle_batch(batch, None)  # type: ignore
            except Exception as e:
                logger.error(f"[ResultProcess] Manual trigger handler error: {e}")

    def _evaluate_alert(self, batch_packet: BatchInferenceResult):
        """Evaluates bird count thresholds for each result in the batch and updates alert flags."""
        new_results = []
        for result in batch_packet.results:
            threshold = self.stream_thresholds.get(result.stream_id, 0)
            new_result = replace(result, alert_flag=result.count >= threshold)
            new_results.append(new_result)
        batch_packet.results = new_results

    def run(self):
        setup_logging(self.config.envs.debug)
        logger.info("[ResultProcess] Started.")

        # Attach to Shared Memory
        shm_client = SharedMemoryClient(self.shm_config)
        shm_client.connect()

        # Start Handlers
        for h in self.handlers:
            h.start()

        # Initialize GUI Class if enabled
        gui: Optional["ManualTriggerGUI"] = None
        if self.config.envs.enable_trigger_gui:
            from .gui import ManualTriggerGUI

            gui = ManualTriggerGUI(self.config, self._handle_manual_trigger)
            gui.setup()

        # 2. Consumption Loop
        while not self._stop_event.is_set():
            if gui:
                gui.update()

            try:
                # [Batch Mode] Get the whole packet
                batch_packet: BatchInferenceResult = self.result_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            try:
                self._evaluate_alert(batch_packet)

                for handler in self.handlers:
                    try:
                        handler.handle_batch(batch_packet, shm_client)
                    except Exception as e:
                        logger.error(f"[ResultProcess] Handler {type(handler).__name__} batch error: {e}")

            finally:
                # --- 4. Critical: Batch Release ---
                shm_client.metadata[batch_packet.stream_ids, batch_packet.buffer_indices]["state"] = BufferState.FREE

        # 5. Cleanup
        if gui:
            gui.destroy()

        for h in self.handlers:
            h.stop()

        shm_client.close()
        logger.info("[ResultProcess] Exited.")

    def stop(self):
        self._stop_event.set()
