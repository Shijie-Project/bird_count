import logging
import multiprocessing as mp
import queue
import time

from .config import Config
from .handlers import BaseHandler
from .inference_process import BatchInferenceResult
from .memory_manager import BufferState, SharedMemoryClient, SharedMemoryConfig
from .utils import setup_logging


logger = logging.getLogger(__name__)


class ResultProcess(mp.Process):
    """
    High-throughput Result Consumer.
    Evaluates thresholds, triggers alerts (with hold-down delay), and manages the SHM lifecycle.
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

        # Hold-down: tracks the wall-clock time at which each stream's count first crossed
        # the threshold continuously. The alert only fires once now - first_seen >= trigger_delay.
        # Cleared on cancel_all() and whenever the count drops below threshold.
        self.alert_first_seen: dict[int, float] = {}
        self.trigger_delay: float = float(getattr(self.config.envs, "alert_trigger_delay", 0.0))

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

    def _handle_cancel_all(self):
        """
        Callback for GUI 'Cancel All' button.
        - Clears every manual hijack.
        - Resets the hold-down timer for every stream.
        - Asks every handler to abort in-flight per-device alert lifecycles.
        """
        cleared_hijacks = sorted(self.manual_override_streams)
        self.manual_override_streams.clear()
        self.alert_first_seen.clear()

        for h in self.handlers:
            try:
                h.cancel_all()
            except Exception as e:
                logger.error(f"[{self.name}] cancel_all failed on {type(h).__name__}: {e}")

        logger.info(
            f"[{self.name}] CANCEL ALL invoked. Cleared hijacks={cleared_hijacks}, "
            f"reset {len(cleared_hijacks)} hijack(s) and all hold-down timers."
        )

    def _get_active_devices_snapshot(self) -> dict[str, set[str]]:
        """Aggregate active-device sets across handlers for GUI status display."""
        snapshot: dict[str, set[str]] = {}
        for h in self.handlers:
            try:
                snapshot[type(h).__name__] = h.get_active_devices()
            except Exception:
                snapshot[type(h).__name__] = set()
        return snapshot

    def _evaluate_alerts_inplace(self, batch_packet: BatchInferenceResult):
        """
        Fast in-place alert evaluation with hold-down logic.
        Avoids creating new objects to reduce GC pressure.
        """
        now = time.time()
        for result in batch_packet.results:
            sid = result.stream_id

            # 1. Manual hijack short-circuits everything (no hold-down).
            if sid in self.manual_override_streams:
                result.alert_flag = True
                continue

            threshold = self.stream_thresholds.get(sid, float("inf"))
            above = result.count >= threshold

            if not above:
                # Drop below threshold -> reset hold-down timer.
                self.alert_first_seen.pop(sid, None)
                result.alert_flag = False
                continue

            # 2. Hold-down: alert only fires after sustained breach.
            first_seen = self.alert_first_seen.get(sid)
            if first_seen is None:
                self.alert_first_seen[sid] = now
                result.alert_flag = self.trigger_delay <= 0.0
            else:
                result.alert_flag = (now - first_seen) >= self.trigger_delay

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

                gui = ManualTriggerGUI(
                    self.config,
                    on_trigger_callback=self._handle_manual_trigger,
                    on_cancel_all_callback=self._handle_cancel_all,
                    status_provider=self._get_active_devices_snapshot,
                )
                gui.setup()
            except ImportError:
                logger.warning(f"[{self.name}] GUI requested but .gui module not found.")
            except Exception as e:
                logger.warning(f"[{self.name}] GUI initialization failed: {e}")
                gui = None

        # 3. Consumption Loop
        while not self._stop_event.is_set():
            # Always service the GUI event loop, even when no results are flowing.
            if gui:
                gui.update()

            try:
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
                sids = batch_packet.stream_ids
                b_idxs = batch_packet.buffer_indices

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
