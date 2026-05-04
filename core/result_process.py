import logging
import multiprocessing as mp
import queue
import time
from typing import Optional

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

    SHM lifecycle:
    - Synchronous handlers (e.g. SmartPlug, Speaker) finish before this process loops, so
      their buffers are freed immediately.
    - Asynchronous handlers (e.g. MonitorHandler -> DisplayProcess) "claim" the buffer by
      returning the (sid, b_idx) pairs from handle_batch(). Those slots stay in READING
      state until the handler acks via ack_queue. A stale-ack sweep force-releases any
      buffer held longer than ack_timeout seconds (safety net for crashed sub-processes).
    """

    ACK_TIMEOUT_SEC = 5.0
    MAX_ACKS_PER_DRAIN = 128

    def __init__(
        self,
        config: Config,
        shm_config: SharedMemoryConfig,
        result_queue: mp.Queue,
        ack_queue: Optional[mp.Queue] = None,
    ):
        super().__init__(name="ResultConsumer")
        self.config = config
        self.shm_config = shm_config
        self.result_queue = result_queue
        self.ack_queue = ack_queue
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

        # Pending-ack registry: (sid, b_idx) -> claim_timestamp.
        self.pending_acks: dict[tuple[int, int], float] = {}

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

    def _release_buffers(self, shm_client: SharedMemoryClient, pairs):
        """Bulk-mark a list/set of (sid, b_idx) pairs FREE."""
        if not pairs:
            return
        sids = [p[0] for p in pairs]
        b_idxs = [p[1] for p in pairs]
        if b_idxs and b_idxs[0] != -1:
            shm_client.metadata[sids, b_idxs]["state"] = BufferState.FREE

    def _drain_ack_queue(self, shm_client: SharedMemoryClient):
        """Non-blocking drain of incoming acks. Releases acked buffers."""
        if self.ack_queue is None:
            return
        released: list[tuple[int, int]] = []
        for _ in range(self.MAX_ACKS_PER_DRAIN):
            try:
                pairs = self.ack_queue.get_nowait()
            except queue.Empty:
                break
            except Exception as e:
                logger.error(f"[{self.name}] ack_queue read error: {e}")
                break
            for pair in pairs:
                key = (int(pair[0]), int(pair[1]))
                if self.pending_acks.pop(key, None) is not None:
                    released.append(key)
        if released:
            self._release_buffers(shm_client, released)

    def _sweep_stale_acks(self, shm_client: SharedMemoryClient, now: float):
        """Force-release buffers whose ack never arrived (handler crash safety net)."""
        if not self.pending_acks:
            return
        cutoff = now - self.ACK_TIMEOUT_SEC
        stale = [k for k, t in self.pending_acks.items() if t < cutoff]
        if not stale:
            return
        for k in stale:
            self.pending_acks.pop(k, None)
        self._release_buffers(shm_client, stale)
        logger.warning(
            f"[{self.name}] Force-released {len(stale)} stale ack(s) after {self.ACK_TIMEOUT_SEC}s timeout."
        )

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
        last_stale_sweep = time.time()
        while not self._stop_event.is_set():
            # Always service GUI + ack queue, even when no results are flowing.
            if gui:
                gui.update()
            self._drain_ack_queue(shm_client)

            # Stale-ack sweep at most every second.
            now = time.time()
            if now - last_stale_sweep >= 1.0:
                self._sweep_stale_acks(shm_client, now)
                last_stale_sweep = now

            try:
                batch_packet: BatchInferenceResult = self.result_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            sids = batch_packet.stream_ids
            b_idxs = batch_packet.buffer_indices
            all_claimed: set[tuple[int, int]] = set()

            try:
                # A. Logical Evaluation (In-place)
                self._evaluate_alerts_inplace(batch_packet)

                # B. Plugin Dispatch — collect claimed buffer pairs from each handler.
                for handler in self.handlers:
                    try:
                        claimed = handler.handle_batch(batch_packet, shm_client)
                        if claimed:
                            all_claimed |= claimed
                    except Exception as e:
                        logger.error(f"[{self.name}] Handler {type(handler).__name__} error: {e}")

            except Exception as e:
                logger.error(f"[{self.name}] Unexpected loop error: {e}")

            finally:
                # C. SHM Release: free unclaimed slots immediately, defer claimed ones until ack.
                if b_idxs and b_idxs[0] != -1:
                    claim_ts = time.time()
                    to_release: list[tuple[int, int]] = []
                    for s, b in zip(sids, b_idxs):
                        key = (int(s), int(b))
                        if key in all_claimed:
                            self.pending_acks[key] = claim_ts
                        else:
                            to_release.append(key)
                    self._release_buffers(shm_client, to_release)

        # 4. Cleanup Sequence
        if gui:
            gui.destroy()

        for h in self.handlers:
            h.stop()

        # Drain any final acks / release leftover buffers so SHM is clean.
        self._drain_ack_queue(shm_client)
        if self.pending_acks:
            self._release_buffers(shm_client, list(self.pending_acks.keys()))
            self.pending_acks.clear()

        shm_client.disconnect()
        logger.info(f"[{self.name}] Stopped.")

    def stop(self):
        self._stop_event.set()
