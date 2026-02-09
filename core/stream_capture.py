import logging
import multiprocessing as mp
import threading
import time

import cv2
import numpy as np

from .config import Config
from .memory_manager import BufferState, SharedMemoryClient, SharedMemoryConfig
from .utils import setup_logging


logger = logging.getLogger(__name__)


class CameraThread(threading.Thread):
    """
    Optimized Thread for high-speed frame capture and SHM ingestion.
    """

    def __init__(self, stream_id: int, source: str, shm_client: SharedMemoryClient, config: Config):
        super().__init__(name=f"CamThread-{stream_id}", daemon=True)
        self.stream_id = stream_id
        self.source = source
        self.config = config

        # Local views of SHM slices for zero-overhead access
        self.metadata = shm_client.stream_metadata[stream_id]
        self.frames = shm_client.stream_frames[stream_id]

        self.num_buffers = config.num_buffers
        self.target_size = (config.shm.width, config.shm.height)  # (W, H) for OpenCV

        self._running = True
        self.frame_idx = 0
        self.consecutive_failures = 0
        self._interval = config.frame_interval

    def run(self):
        """Optimized connection and capture loop."""
        while self._running:
            # OPTIMIZATION: Use API preference for hardware acceleration if available
            # e.g., cv2.CAP_FFMPEG or specific HW backends
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)

            # Optional: Set buffer size to minimize latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self._handle_connection_error(f"[{self.name}] Source open failed")
                continue

            logger.info(f"[{self.name}] Connected.")
            self.consecutive_failures = 0

            while self._running:
                t_start = time.perf_counter()

                # Step 1: Grab (Fast)
                # .grab() + .retrieve() is slightly faster than .read() for multi-threaded apps
                if not cap.grab():
                    break

                # Step 2: Retrieve (Decoding happens here)
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    break

                # Step 3: Process and Ingest
                self._process_frame(frame)

                # Step 4: Precise FPS Throttle
                elapsed = time.perf_counter() - t_start
                if elapsed < self._interval:
                    time.sleep(self._interval - elapsed)

            cap.release()
            if self._running:
                time.sleep(2.0)  # Backoff before retry

    def _process_frame(self, frame: np.ndarray):
        """
        Ingests frame into SHM with minimal CPU overhead.
        """
        self.frame_idx += 1

        # --- 1. Vectorized Slot Selection (Acceleration Point) ---
        # Find FREE buffers first, then fallback to READY
        states = self.metadata["state"]
        free_indices = np.where(states == BufferState.FREE)[0]

        if free_indices.size > 0:
            target_idx = free_indices[0]
        else:
            ready_indices = np.where(states == BufferState.READY)[0]
            if ready_indices.size > 0:
                # Select the oldest READY buffer to overwrite
                # We use frame_idx to find the oldest
                f_indices = self.metadata["frame_idx"][ready_indices]
                target_idx = ready_indices[np.argmin(f_indices)]
            else:
                # All buffers are WRITING or READING
                logger.debug(f"[{self.name}] Congestion: No buffer available. Dropping #{self.frame_idx}")
                return

        target_meta = self.metadata[target_idx]
        try:
            # Mark as WRITING to prevent InferenceProcess from reading partial data
            target_meta["state"] = BufferState.WRITING

            # Optimization: Only resize if necessary
            if frame.shape[1] != self.target_size[0] or frame.shape[0] != self.target_size[1]:
                frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

            # Optimization: Move BGR->RGB conversion to GPU (Lazy conversion)
            # If your InferenceProcess handles BGR, skip this.
            # Otherwise, if we must do it on CPU, this is the bottleneck.
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # np.copyto is optimized for memory throughput
            np.copyto(self.frames[target_idx], frame)  # BGR

            # Update Metadata in a single atomic-like sequence
            target_meta["frame_idx"] = self.frame_idx
            target_meta["timestamp"] = time.time()
            target_meta["stream_id"] = self.stream_id
            target_meta["buffer_idx"] = target_idx

            # Commit to READY state
            target_meta["state"] = BufferState.READY

        except Exception as e:
            logger.error(f"[{self.name}] Write failed: {e}")
            target_meta["state"] = BufferState.FREE

    def _handle_connection_error(self, msg: str):
        self.consecutive_failures += 1
        wait = min(2**self.consecutive_failures, 30)
        logger.error(f"[{self.name}] {msg}. Retrying in {wait}s.")
        time.sleep(wait)

    def stop(self):
        self._running = False


class GrabberProcess(mp.Process):
    """
    Orchestrates multiple CameraThreads in a dedicated process.
    """

    def __init__(self, config: Config, shm_config: SharedMemoryConfig):
        super().__init__(name="GrabberProcess")
        self.config = config
        self.shm_config = shm_config
        self.threads: list[CameraThread] = []
        self._stop_event = mp.Event()

    def run(self):
        setup_logging(self.config.envs.debug)
        logger.info(f"[{self.name}] Process Started.")

        shm_client = SharedMemoryClient(self.shm_config)
        shm_client.connect()

        # Start a thread for each source
        for i, source in enumerate(self.config.stream_sources):
            t = CameraThread(i, source, shm_client, self.config)
            self.threads.append(t)
            t.start()

        # Watchdog: Monitoring thread health
        while not self._stop_event.is_set():
            for i, t in enumerate(self.threads):
                if not t.is_alive() and not self._stop_event.is_set():
                    logger.warning(f"Thread {t.name} died. Reviving...")
                    new_t = CameraThread(i, self.config.stream_sources[i], shm_client, self.config)
                    self.threads[i] = new_t
                    new_t.start()
            time.sleep(1.0)

        # Cleanup
        for t in self.threads:
            t.stop()
            t.join(timeout=1.0)

        shm_client.disconnect()
        logger.info(f"[{self.name}] Process Stopped.")

    def stop(self):
        self._stop_event.set()
