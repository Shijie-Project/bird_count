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

        self.timeout_seconds = cfg.source.timeout_seconds
        self.input_h = cfg.model.input_h
        self.input_w = cfg.model.input_w
        self._interval = cfg.stream.frame_interval_s

    def run(self) -> None:
        try:
            # Attach Frames SHM
            existing_shm_frames = shm.SharedMemory(name=self.shm_frames_info.name)
            full_frames = np.ndarray(
                self.shm_frames_info.shape, dtype=self.shm_frames_info.dtype, buffer=existing_shm_frames.buf
            )
            shm_frames = full_frames[self.sid]  # [NUM_BUFFER, H, W, 3]

            # Attach Meta SHM
            existing_shm_meta = shm.SharedMemory(name=self.shm_meta_info.name)
            full_meta = np.ndarray(
                self.shm_meta_info.shape, dtype=self.shm_meta_info.dtype, buffer=existing_shm_meta.buf
            )
            shm_meta = full_meta[self.sid]

        except Exception as e:
            logger.error(f"[Stream {self.sid}] SHM attach failed: {e}")
            return

        video_src = f"https://root:root@{self.src}/mjpg/1/video.mjpg" if self.cfg.source.mode == "camera" else self.src

        error_frame = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)

        while not self.stop_event.is_set():
            cap = cv2.VideoCapture(video_src)

            if not cap.isOpened():
                logger.warning(f"[Stream {self.sid}] Connection failed.")
                cap.release()

                self._pacing_write(error_frame, shm_frames, shm_meta)

                time.sleep(self.timeout_seconds)
                continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            logger.info(f"[Stream {self.sid}] Connected.")

            last_ok_time = time.time()

            while not self.stop_event.is_set():
                if not cap.grab():
                    if time.time() - last_ok_time > self.timeout_seconds:
                        logger.warning(f"[Stream {self.sid}] Timeout. Reconnecting...")
                        break
                    time.sleep(self._interval)
                    continue

                last_ok_time = time.time()

                ret, frame = cap.retrieve()
                if not ret:
                    break

                try:
                    if frame.shape[:2] != (self.input_h, self.input_w):
                        frame = cv2.resize(frame, (self.input_w, self.input_h))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    self._pacing_write(frame, shm_frames, shm_meta)

                except Exception as e:
                    logger.error(f"[Stream {self.sid}] Process error: {e}")
                    break

            cap.release()

    def _pacing_write(self, frame, shm_frames, shm_meta):
        now = time.time()
        next_slot = int(now / self._interval) + 1
        target_ts = next_slot * self._interval

        # 写入数据
        self._write_frame_to_shm(frame, shm_frames, shm_meta)

        wait = target_ts - time.time()
        if wait > 0:
            time.sleep(wait)

    def _write_frame_to_shm(self, frame, shm_frames, shm_meta):
        # Determine index
        current_idx = self.latest_cursor[self.sid]
        next_idx = (current_idx + 1) % self.cfg.stream.num_buffer

        # Check Lock
        if shm_meta[next_idx, 0] == 1.0:
            logger.warning(f"[Stream {self.sid}] Buffer {next_idx} Locked. Dropping frame.")
            return  # Drop if locked

        # Write Data
        np.copyto(shm_frames[next_idx], frame)

        # Update Metadata
        shm_meta[next_idx, 0] = 2.0
        shm_meta[next_idx, 1] = time.time()

        # 6. Update Cursor
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
