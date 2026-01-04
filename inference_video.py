import logging
import multiprocessing as mp
import queue
import threading
import time
import warnings
from dataclasses import dataclass, field
from pprint import pformat
from typing import Optional

import cv2
import numpy as np
import torch
from pydantic_settings import BaseSettings

from models import ShuffleNetV2_x1_0
from utils import GrabberItem, ResultItem, TaskItem


class Envs(BaseSettings):
    debug: bool = False
    fps: int = 10
    num_streams: int = 22
    cuda_visible_devices: str = "0"
    num_workers_per_gpu: int = 1


envs = Envs()


logging.basicConfig(
    level=logging.DEBUG if envs.debug else logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("InferenceEngine")
warnings.filterwarnings("ignore")


# --- Setup Logging ---
@dataclass
class Config:
    # Model Settings
    MODEL_PATH: str = "./ckpts/shufflenet_model_best.pth"
    INPUT_SIZE: tuple[int, int] = (512, 512)

    # Hardware Settings
    AVAILABLE_GPUS: tuple[int, ...] = (0,)
    NUM_WORKERS_PER_GPU: int = 1

    # Sources
    RTSP_URL: tuple[str, ...] = ("rtsp://127.0.0.1:8554/live/test",)

    # Stream
    TARGET_FPS: float = 10.0  # Target FPS per stream.
    NUM_STREAMS: int = 22  # Number of concurrent video streams.
    RUNTIME_SECONDS: Optional[int] = None

    # --- 2. Auto Configs (do not modify) ---
    frame_interval_s: float = field(init=False)
    stream_sources: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        # set frame_interval_s
        if self.TARGET_FPS < 1 or self.TARGET_FPS > 30:
            logger.warning(f"Current target FPS {self.TARGET_FPS} will be clamped to [1, 30].")
        self.TARGET_FPS = max(1.0, min(30.0, self.TARGET_FPS))

        self.frame_interval_s = 1.0 / self.TARGET_FPS

        # set stream_sources
        sources = self.RTSP_URL
        if len(sources) < self.NUM_STREAMS:
            sources = sources * (self.NUM_STREAMS // len(sources) + 1)
        self.stream_sources = tuple(sources[: self.NUM_STREAMS])

        logger.info("--- Configuration ---")
        logger.info(f"FPS: {self.TARGET_FPS} (Interval: {self.frame_interval_s * 1000:.1f}ms)")
        logger.info(f"Streams: {self.NUM_STREAMS}")
        logger.info(f"Available GPUs: {len(self.AVAILABLE_GPUS)}")
        logger.info(f"Total Workers: {len(self.AVAILABLE_GPUS) * self.NUM_WORKERS_PER_GPU}")
        logger.info(f"Source(s):\n{pformat(self.stream_sources)}")


# --- 1. 抓帧线程 (Frame Grabber) ---
class FrameGrabber(threading.Thread):
    def __init__(
        self,
        sid: int,
        src: str,
        queue_out: "mp.Queue[GrabberItem]",
        stop_event: threading.Event,
        cfg: Config,
    ) -> None:
        super().__init__(daemon=True, name=f"grabber-{sid}")
        self.sid = sid
        self.src = src
        self.queue_out = queue_out
        self.stop_event = stop_event
        self.cfg = cfg

        # Pre-calculate interval
        self._interval = cfg.frame_interval_s

    def connection(self):
        """Establishes RTSP connection with retry mechanism."""
        retries = 5
        while retries > 0 and not self.stop_event.is_set():
            cap = cv2.VideoCapture(self.src)
            try:
                # Backend settings for timeout and buffer
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            except Exception:
                pass

            if cap.isOpened():
                logger.info(f"[Stream {self.sid}] Connected.")
                return cap

            retries -= 1
            logger.warning(f"[Stream {self.sid}] Connect failed. Retrying ({retries} left)...")
            cap.release()
            time.sleep(2)

        return None

    def run(self) -> None:
        # --- Outer Loop: Connection Management ---
        while not self.stop_event.is_set():
            cap = self.connection()
            if cap is None:
                logger.error(f"[Stream {self.sid}] Critical failure. Retrying in 10s...")
                time.sleep(10)
                continue

            last_ok_time = time.perf_counter()

            # --- Inner Loop: Processing Cycle ---
            while not self.stop_event.is_set():

                # =========================================================
                # 1. Buffer Draining / Flush
                # =========================================================
                # We do this FIRST to get the freshest frame available NOW.
                grabbed = False
                flush_count = 0
                max_flush = int(self.cfg.TARGET_FPS) + 5

                for _ in range(max_flush):
                    if not cap.grab():
                        break
                    grabbed = True
                    flush_count += 1

                if not grabbed:
                    # Timeout check
                    if time.perf_counter() - last_ok_time > 5.0:
                        logger.warning(f"[Stream {self.sid}] No frames for 5s (Timeout). Reconnecting...")
                        break
                    time.sleep(0.001)
                    continue

                # =========================================================
                # 2. Retrieve & Process (The Variable Workload)
                # =========================================================
                ret, frame = cap.retrieve()
                if not ret:
                    continue

                # This represents the physical time the frame was decoded.
                capture_ts = time.time()

                # Update watchdog
                last_ok_time = time.perf_counter()

                try:
                    # Heavy CPU operations (Resize, Color Convert)
                    # We do this BEFORE sleeping to ensure we are ready to send.
                    frame = cv2.resize(frame, self.cfg.INPUT_SIZE)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    logger.error(f"[Stream {self.sid}] Processing error: {e}")
                    continue

                # =========================================================
                # 3. Calculate Alignment Timestamp
                # =========================================================
                # Now that we have the processed data ready, we calculate
                # WHEN it should be sent. We aim for the NEXT immediate slot.
                now = time.time()
                current_slot = int(now / self._interval)
                next_slot = current_slot + 1
                target_ts = next_slot * self._interval

                # =========================================================
                # 4. Sleep Until Target Time (Strict Sync)
                # =========================================================
                # We wait here. The thread holds the processed data and pauses.
                time_to_wait = target_ts - now

                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                else:
                    # If time_to_wait < 0, it means processing took too long
                    # and we missed the bus. We send it immediately (best effort)
                    # or you could choose to drop it here if strictness is vital.
                    pass

                    # =========================================================

                # 5. Send (Aligned Action)
                # =========================================================
                # At this exact moment, all 22 threads should be waking up
                # together (approx. within 1-2ms) to push data.
                try:
                    self.queue_out.put_nowait(GrabberItem(self.sid, frame, capture_ts))

                    # Optional Debug:
                    logger.debug(f"[Stream {self.sid}] Frame pushed at TS: {target_ts:.3f}")

                except queue.Full:
                    pass

            # --- End of Inner Loop ---
            if cap:
                cap.release()
            logger.info(f"[Stream {self.sid}] Disconnected/Reconnecting...")

        logger.info(f"[Stream {self.sid}] Thread Stopped.")


# --- 3. Main Scheduler Loop ---
def run(cfg: Config) -> None:
    """
    Main Loop: Collects frames -> Forms batches -> Dispatches to GPU Workers.
    Implements 'Bucket Accumulation with Timeout' strategy.
    """
    ctx = mp.get_context("spawn")
    shutdown = ctx.Event()

    # A. Initialize Per-Stream Input Queues (Maxsize=1 ensures freshness)
    stream_queues: list["mp.Queue[GrabberItem]"] = [ctx.Queue(maxsize=1) for _ in range(cfg.NUM_STREAMS)]

    # B. Start Frame Grabber Threads
    stop_event = threading.Event()
    grabbers = []
    for i, src in enumerate(cfg.stream_sources):
        t = FrameGrabber(i, src, stream_queues[i], stop_event, cfg)
        t.start()
        grabbers.append(t)

    while True:
        pass


def main():
    # Environment variable overrides for easy testing
    gpus = tuple(int(x) for x in envs.cuda_visible_devices.split(",") if x.strip() != "")

    cfg = Config(
        TARGET_FPS=envs.fps,
        NUM_STREAMS=envs.num_streams,
        AVAILABLE_GPUS=gpus if gpus else (0,),
        NUM_WORKERS_PER_GPU=envs.num_workers_per_gpu,
        RUNTIME_SECONDS=None,
    )
    run(cfg)


if __name__ == "__main__":
    # Required for CUDA
    mp.set_start_method("spawn", force=True)
    main()
