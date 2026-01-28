import logging
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import threading
import time

import cv2
import numpy as np

from .config import Config
from .memory_manager import SharedMemoryInfo


logger = logging.getLogger(__name__)


class FrameGrabber(threading.Thread):
    def __init__(
        self,
        sid: int,
        src: str,
        shm_frames_info: SharedMemoryInfo,
        shm_meta_info: SharedMemoryInfo,
        latest_cursor: mp.Array,
        stop_event: threading.Event,
        cfg: Config,
    ) -> None:
        super().__init__(daemon=True, name=f"grabber-{sid}")
        self.sid = sid
        self.src = src
        self.shm_frames_info = shm_frames_info
        self.shm_meta_info = shm_meta_info
        self.latest_cursor = latest_cursor
        self.stop_event = stop_event
        self.cfg = cfg

        self._interval = cfg.stream.frame_interval_s

    def run(self) -> None:
        # 1. Attach Video Frames SHM
        try:
            existing_shm_frames = shm.SharedMemory(name=self.shm_frames_info.name)
            full_frames = np.ndarray(
                self.shm_frames_info.shape, dtype=self.shm_frames_info.dtype, buffer=existing_shm_frames.buf
            )
            shm_frames = full_frames[self.sid]  # [NUM_BUFFER, H, W, 3]
        except Exception as e:
            logger.error(f"[Stream {self.sid}] Frames SHM attach failed: {e}")
            return

        # 2. Attach Metadata SHM
        try:
            existing_shm_meta = shm.SharedMemory(name=self.shm_meta_info.name)
            full_meta = np.ndarray(
                self.shm_meta_info.shape, dtype=self.shm_meta_info.dtype, buffer=existing_shm_meta.buf
            )
            shm_meta = full_meta[self.sid]
        except Exception as e:
            logger.error(f"[Stream {self.sid}] Meta SHM attach failed: {e}")
            return

        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(self.src)
            if not cap.isOpened():
                logger.warning(f"[Stream {self.sid}] Connection failed. Retrying in 5s...")
                time.sleep(5)
                continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.info(f"[Stream {self.sid}] Connected.")
            last_ok_time = time.time()

            while not self.stop_event.is_set():
                if not cap.grab():
                    if time.time() - last_ok_time > 5.0:
                        logger.warning(f"[Stream {self.sid}] Timeout. Reconnecting...")
                        break
                    time.sleep(0.01)
                    continue

                last_ok_time = time.time()
                next_slot = int(last_ok_time / self._interval) + 1
                target_ts = next_slot * self._interval

                ret, frame = cap.retrieve()
                if not ret:
                    break

                try:
                    self.process_grabber_frame(frame, shm_frames, shm_meta)

                    wait = target_ts - time.time()
                    if wait > 0:
                        time.sleep(wait)

                except Exception as e:
                    logger.error(f"[Stream {self.sid}] Process error: {e}")
                    break

            cap.release()

    def process_grabber_frame(
        self,
        frame: np.ndarray,
        shm_frames: np.ndarray,
        shm_meta: np.ndarray,
    ):
        """
        Core Grabber Logic:
        1. Determine next buffer index.
        2. Check lock state (Drop if locked).
        3. Write data & update metadata.
        """
        # CPU Preprocessing
        frame = cv2.resize(frame, (self.cfg.model.input_w, self.cfg.model.input_h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        current_idx = self.latest_cursor[self.sid]
        next_idx = (current_idx + 1) % self.cfg.stream.num_buffer

        # 2. Check Buffer State
        # 0.0 = Free, 2.0 = Ready (Overwritable), 1.0 = Locked (Protected)
        state = shm_meta[next_idx, 0]

        if state == 1.0:
            # [Flow Control] Target buffer is in use. Drop this frame.
            logger.debug(f"[Stream {self.sid}] Buffer {next_idx} Locked. Dropping frame.")
            return

        # 3. Write Data (Zero Copy)
        np.copyto(shm_frames[next_idx], frame)

        # 4. Update Metadata
        # Mark as Ready (2.0)
        shm_meta[next_idx, 0] = 2.0
        shm_meta[next_idx, 1] = time.time()

        # 5. Update Global Cursor
        # Using lock ensures Scheduler reads consistent state if it checks concurrently
        with self.latest_cursor.get_lock():
            self.latest_cursor[self.sid] = next_idx


def spawn_grabbers(cfg, shm_frames_info, shm_meta_info, latest_cursor, stop_event) -> list[FrameGrabber]:
    grabbers = []
    for i, src in enumerate(cfg.active_stream_sources):
        t = FrameGrabber(
            sid=i,
            src=src,
            shm_frames_info=shm_frames_info,
            shm_meta_info=shm_meta_info,
            latest_cursor=latest_cursor,
            stop_event=stop_event,
            cfg=cfg,
        )
        t.start()
        grabbers.append(t)
    return grabbers
