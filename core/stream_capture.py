import logging
import multiprocessing as mp
import threading
import time

import cv2
import numpy as np

from .config import Config
from .memory_manager import BufferState, SharedMemoryClient, SharedMemoryConfig
from .utils import setup_logging


# Setup logger
logger = logging.getLogger(__name__)

LOGGING_DEBUG = False


class CameraThread(threading.Thread):
    """
    Independent thread for a single RTSP stream.

    Responsibility:
    1. Connect to RTSP source.
    2. Decode frames (blocking I/O).
    3. Push to Shared Memory IF a slot is FREE (Non-blocking).
    4. Drop frames if the system is congested.
    """

    def __init__(self, stream_id: int, source: str, shm_client: SharedMemoryClient, config: Config):
        super().__init__(name=f"CamThread-{stream_id}", daemon=True)
        self.stream_id = stream_id
        self.source = source
        self.config = config

        self.metadata = shm_client.metadata[stream_id]
        self.frames = shm_client.frames[stream_id]

        # Buffer Management
        self.num_buffers = config.num_buffers
        self.lastest_ready_buffer = -1

        # State
        self.running = True
        self.frame_idx = 0

        # Metrics
        self.consecutive_failures = 0

        # Stream Control
        self._interval = config.frame_interval

        self.dummy_frame = self._create_dummy_frame()

    def _create_dummy_frame(self):
        h, w = self.config.shm.height, self.config.shm.width
        img = np.zeros((h, w, 3), dtype=np.uint8)
        text = f"CAM {self.stream_id}: NO SIGNAL"
        font_scale = 1.0 if w < 1000 else 2.0
        cv2.putText(img, text, (50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
        return img

    def run(self):
        while self.running:
            # 1. Connection Loop
            cap = cv2.VideoCapture(self.source)

            if not cap.isOpened():
                self._process_frame(self.dummy_frame)
                self._handle_connection_error("Failed to open source")
                continue  # Retry loop

            logger.info(f"[Stream {self.stream_id}] Connected. Source: {self.source}")
            self.consecutive_failures = 0  # Reset backoff

            # 2. Frame Loop
            while self.running:
                start_time = time.perf_counter()

                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"[Stream {self.stream_id}] Stream EOF or lost signal.")
                    break  # Break inner loop to trigger reconnection

                self._process_frame(frame)

                elapsed = time.perf_counter() - start_time
                # Calculate how long we need to sleep to hit target FPS
                wait_time = self._interval - elapsed

                if wait_time > 0:
                    # Precise sleep to throttle the loop
                    time.sleep(wait_time)

            # Cleanup before reconnection
            cap.release()
            time.sleep(1.0)  # Prevent rapid-fire reconnection spam

        logger.info(f"[Stream {self.stream_id}] Thread stopped.")

    def stop(self):
        """Signal thread to stop."""
        self.running = False

    def _process_frame(self, frame: np.ndarray):
        """
        Writes the frame to Shared Memory using a "Best Effort" strategy.
        Policy:
        1. Prefer FREE buffers.
        2. Overwrite READY buffers (Oldest first) to maintain low latency.
        3. NEVER touch PROCESSING/WRITING buffers (Safety).
        4. If all buffers are busy, DROP the frame.
        """

        self.frame_idx += 1

        target_idx = -1
        overwrite_candidate = -1

        # --- 1. Search for a Target Slot ---
        # Scan all buffers starting from the next one in the ring
        for i in range(1, self.num_buffers + 1):
            # Calculate index (Round Robin)
            # Assuming self.lastest_ready_buffer is effectively "current_write_head"
            idx = (self.lastest_ready_buffer + i) % self.num_buffers

            # Access metadata directly (Zero copy read)
            # Note: Ensure self.metadata is sliced for this stream_id: self.shm.metadata[self.stream_id]
            current_meta = self.metadata[idx]
            state = current_meta["state"]

            if state == BufferState.FREE:
                target_idx = idx
                break

            if state == BufferState.READY and overwrite_candidate == -1:
                overwrite_candidate = idx

        # --- 2. Backpressure Handling (Drop Frame) ---
        if target_idx == -1:
            if overwrite_candidate != -1:
                target_idx = overwrite_candidate

                if LOGGING_DEBUG:
                    old_frame_idx = self.metadata[target_idx]["frame_idx"]
                    logger.debug(
                        f"[Stream {self.stream_id}] Buffer Full. "
                        f"Overwriting frame #{old_frame_idx} with #{self.frame_idx} at buffer {target_idx}."
                    )
            else:
                logger.warning(f"[Stream {self.stream_id}] All buffers busy. Drop frame #{self.frame_idx}.")
                return

        # --- 3. Write Data (Critical Section) ---
        target_h, target_w = self.config.shm.height, self.config.shm.width
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        target_meta = self.metadata[target_idx]
        try:
            # 3.1 Lock the slot immediately
            target_meta["state"] = BufferState.WRITING

            # 3.2 Zero-copy Write
            np.copyto(self.frames[target_idx], frame)

            # 3.4 Update Metadata
            target_meta["stream_id"] = self.stream_id
            target_meta["buffer_idx"] = target_idx
            target_meta["frame_idx"] = self.frame_idx
            target_meta["timestamp"] = time.time()

            # 3.5 Commit: Release to Consumer
            target_meta["state"] = BufferState.READY

            # Update local pointer only on success
            self.lastest_ready_buffer = target_idx

        except Exception as e:
            logger.error(f"[Stream {self.stream_id}] Write frame #{self.frame_idx} failed: {e}")
            # Emergency Rollback
            if target_meta is not None:
                target_meta["state"] = BufferState.FREE

    def _cycle_buffer_index(self, current_idx):
        return (current_idx + 1) % self.num_buffers

    def _handle_connection_error(self, msg: str):
        """Exponential Backoff for reconnection."""
        self.consecutive_failures += 1
        # Cap wait time at 30 seconds
        wait_time = min(2**self.consecutive_failures, 30)

        logger.error(f"[Stream {self.stream_id}] {msg}. Retrying in {wait_time}s...")
        time.sleep(wait_time)


class GrabberProcess(mp.Process):
    """
    The Main Process that manages all Camera Threads.
    Separated from Main Logic to allow restarting independent of the Inference Engine.
    """

    def __init__(self, config: Config, shm_config: SharedMemoryConfig):
        super().__init__(name="GrabberProcess")
        self.config = config
        self.shm_config = shm_config
        self.threads: list[CameraThread] = []
        self._stop_event = mp.Event()

    def run(self):
        setup_logging(self.config.envs.debug)
        logger.info("Process Started.")

        # 1. Attach to Shared Memory (as a Client)
        shm_client = SharedMemoryClient(self.shm_config)
        try:
            shm_client.connect()
        except Exception as e:
            logger.critical(f"Could not connect to SHM: {e}")
            return

        # 2. Spawn Threads
        for i, source in enumerate(self.config.stream_sources):
            t = CameraThread(stream_id=i, source=source, shm_client=shm_client, config=self.config)
            self.threads.append(t)
            t.start()

        # 3. Watchdog Loop
        while not self._stop_event.is_set():
            # Check for dead threads and restart them
            for i, t in enumerate(self.threads):
                if not t.is_alive():
                    logger.warning(f"Stream {i} died. Restarting...")
                    # Re-instantiate
                    new_t = CameraThread(
                        stream_id=i,
                        source=self.config.stream_sources[i],
                        shm_client=shm_client,
                        config=self.config,
                    )
                    self.threads[i] = new_t
                    new_t.start()

            time.sleep(1.0)

        # 4. Cleanup
        logger.info("Stopping threads...")
        for t in self.threads:
            t.stop()
            t.join()

        shm_client.close()
        logger.info("Process Exited.")

    def stop(self):
        self._stop_event.set()
