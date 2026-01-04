import logging
import multiprocessing as mp
import os
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

from models import ShuffleNetV2_x1_0
from utils import SimulatedCamera


logging.basicConfig(
    level=logging.INFO if os.getenv("DEBUG", "0") == "0" else logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("InferenceEngine")
warnings.filterwarnings("ignore")


# --- Define Types ---
GrabberItem = tuple[int, np.ndarray, float]
TaskItem = tuple[list[int], torch.Tensor]
ResultItem = tuple[list[int], torch.Tensor, torch.Tensor]


# --- Setup Logging ---
@dataclass
class Config:
    # --- Core Parameters (Adjust these to control performance) ---
    TARGET_FPS: float = 10.0  # Target FPS per stream. All timeouts are derived from this.
    NUM_STREAMS: int = 4  # Number of concurrent video streams.

    # Model Settings
    MODEL_PATH: str = "./ckpts/shufflenet_model_best.pth"
    INPUT_SIZE: tuple[int, int] = (512, 512)

    # Hardware Settings
    AVAILABLE_GPUS: tuple[int, ...] = (0,)
    WORKERS_PER_GPU: int = 1

    # 数据源
    USE_VIDEO: bool = False
    RTSP_URL: tuple[str, ...] = ("rtsp://127.0.0.1:8554/live/test",)
    VIDEO_FILE: tuple[str, ...] = ("./data/bird_count/bird_count_demo.mp4",)

    # 运行时长 (None 为无限)
    RUNTIME_SECONDS: Optional[int] = None

    # --- 2. 自动计算的参数 (不要手动改) ---
    frame_interval_s: float = field(init=False)
    max_batch_wait_s: float = field(init=False)
    stream_sources: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        # A. 计算帧间隔
        self.frame_interval_s = 1.0 / max(0.1, self.TARGET_FPS)

        # B. 动态超时策略：
        # 允许主进程为了凑 Batch 最多等待帧间隔的 15%
        # 留 85% 的时间给数据传输和 GPU 计算，确保不积压
        self.max_batch_wait_s = self.frame_interval_s * 0.15

        # C. 源设置
        sources = self.VIDEO_FILE if self.USE_VIDEO else self.RTSP_URL
        if len(sources) < self.NUM_STREAMS:
            sources = sources * (self.NUM_STREAMS // len(sources) + 1)
        self.stream_sources = tuple(sources[: self.NUM_STREAMS])

        logger.info("--- Configuration ---")
        logger.info(f"Target FPS: {self.TARGET_FPS} (Interval: {self.frame_interval_s * 1000:.1f}ms)")
        logger.info(f"Batch Wait Budget: {self.max_batch_wait_s * 1000:.1f}ms (15% of interval)")
        logger.info(f"Streams: {self.NUM_STREAMS}")
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

        # Limit the capture rate
        self._target_period_s = 1.0 / max(1.0, self.cfg.TARGET_FPS)
        self._last_emit = 0.0
        self._start_time = time.perf_counter()

    def run(self) -> None:
        # --- Outer Loop: Handles Connection & Reconnection ---
        while not self.stop_event.is_set():
            # 1. Attempt Connection
            if os.path.exists(self.src):
                cap = SimulatedCamera(self.src, target_fps=self.cfg.TARGET_FPS)
            else:
                cap = cv2.VideoCapture(self.src)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # Set timeouts to avoid hanging indefinitely
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                except Exception:
                    pass

            if not cap.isOpened():
                logger.error(f"Stream {self.sid}: Failed to open {self.src}. Retrying in 5s...")
                cap.release()
                time.sleep(5)
                continue

            logger.info(f"Stream {self.sid}: Connected.")

            # --- Inner Loop: Frame Reading ---
            last_success_time = time.perf_counter()

            while not self.stop_event.is_set():
                # Use grab() to check for availability (faster than retrieve)
                ret = cap.grab()

                if not ret:
                    # Connection Watchdog: If no frames for 5s, break to reconnect
                    if time.perf_counter() - last_success_time > 5.0:
                        logger.warning(f"Stream {self.sid}: Connection lost (timeout). Reconnecting...")
                        break  # Break inner loop -> Trigger outer loop reconnection

                    time.sleep(0.01)
                    continue

                now = time.perf_counter()
                last_success_time = now

                # FPS Throttling: Skip frame if we are reading too fast
                # (Crucial for files, optional for RTSP depending on backend)
                if now - self._last_emit < self._target_period_s:
                    time.sleep(0.001)
                    continue

                # Actual Decode
                ret, frame = cap.retrieve()
                if not ret:
                    continue

                # --- Optimization Start ---
                # 1. Resize on CPU
                frame = cv2.resize(frame, self.cfg.INPUT_SIZE)
                # 2. Convert Color (Keep uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # --- Optimization End ---

                self._last_emit = now
                try:
                    # Load Shedding: Drop frame if queue is full
                    self.queue_out.put_nowait((self.sid, frame, time.time()))
                except queue.Full:
                    if now - self._start_time > 10.0:
                        logger.debug(f"Stream {self.sid}: Queue full, dropped old frame.")
                    pass
                except Exception as e:
                    logger.error(f"Stream {self.sid} processing error: {e}")

            # Clean up before reconnecting
            cap.release()
            logger.info(f"Stream {self.sid}: Disconnected/Reconnecting...")

        logger.info(f"Stream {self.sid}: Thread Stopped.")


# --- 2. Inference Worker Process ---
def inference_worker(
    worker_id: int,
    gpu_id: int,
    cfg: Config,
    task_queue: "mp.Queue[TaskItem]",
    result_queue: "mp.Queue[ResultItem]",
    shutdown: "mp.Event",
) -> None:
    """
    GPU Worker. Handles data transfer to GPU, preprocessing, and inference.
    """
    # Initialize CUDA
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    # Enable cuDNN benchmark for fixed input sizes
    torch.backends.cudnn.benchmark = True

    model = ShuffleNetV2_x1_0()
    try:
        st = torch.load(cfg.MODEL_PATH, map_location=device)
        model.load_state_dict(st)
    except Exception as e:
        logger.error(f"Worker {worker_id}: Load model failed: {e}")
        return

    model.to(device).eval()

    # Optimization: Pre-allocate constants on GPU to avoid repetitive transfers
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    logger.info(f"[Worker {worker_id}] Initialized on GPU {gpu_id}.")

    while not shutdown.is_set():
        try:
            # Wait for a batch of data.
            # Timeout is short to allow checking the shutdown event.
            batch_data = task_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        stream_ids, frames_uint8 = batch_data

        # Optimization: Full GPU Preprocessing Pipeline
        # Input: uint8 CPU Tensor -> GPU -> Float -> Normalize
        try:
            # 1. Async transfer to GPU (Non-blocking)
            input_tensor = frames_uint8.to(device, non_blocking=True)

            # 2. Permute: (B, H, W, 3) -> (B, 3, H, W)
            input_tensor = input_tensor.permute(0, 3, 1, 2)

            # 3. Cast to float and Normalize (Leveraging GPU parallel power)
            input_tensor = input_tensor.float().div(255.0)
            input_tensor = (input_tensor - mean) / std

            # 4. Inference with Automatic Mixed Precision (AMP) for FP16 speedup
            with torch.inference_mode():
                with torch.amp.autocast("cuda"):
                    outputs, _ = model(input_tensor)

            # 5. Result Transfer
            # Move only the necessary output (density map) back to CPU
            out_cpu = outputs.detach().half().cpu()

            result_queue.put((stream_ids, out_cpu, frames_uint8))

        except Exception as e:
            logger.error(f"[Worker {worker_id}] Inference Error: {e}")

    logger.info(f"[Worker {worker_id}] Shutdown.")


# --- 3. Main Scheduler Loop ---
def run_scheduler(cfg: Config) -> None:
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

    # C. Setup Workers and Queues
    total_workers = len(cfg.AVAILABLE_GPUS) * cfg.WORKERS_PER_GPU

    # Task Queue: CPU -> GPU (Buffer size 2 to pipeline batches)
    task_queues: list["mp.Queue[TaskItem]"] = [ctx.Queue(maxsize=2) for _ in range(total_workers)]
    # Result Queue: GPU -> CPU
    result_queue: "mp.Queue[ResultItem]" = ctx.Queue(maxsize=total_workers * 2)

    workers = []
    worker_idx = 0
    for gpu in cfg.AVAILABLE_GPUS:
        for _ in range(cfg.WORKERS_PER_GPU):
            p = ctx.Process(
                target=inference_worker,
                args=(worker_idx, gpu, cfg, task_queues[worker_idx], result_queue, shutdown),
                name=f"worker-{worker_idx}",
                daemon=True,
            )
            p.start()
            workers.append(p)
            worker_idx += 1

    # D. Scheduling Constants
    # Distribute streams evenly across workers (Round-Robin)
    streams_per_worker = (cfg.NUM_STREAMS + total_workers - 1) // total_workers
    target_batch_size = streams_per_worker

    # Buffer for accumulating frames: pending_batches[worker_id] = [item1, item2...]
    pending_batches: list[list[GrabberItem]] = [[] for _ in range(total_workers)]
    batch_start_times: list[Optional[float]] = [None] * total_workers

    logger.info(f"Scheduler: Managing {cfg.NUM_STREAMS} streams with {total_workers} workers.")
    logger.info(f"Scheduler: Target Batch Size = {target_batch_size}")

    # E. Main Execution Loop
    time.sleep(3)  # Wait for workers to initialize
    start_time = time.time()

    total_frames_processed = 0
    fps_log_time = time.time()

    try:
        while not shutdown.is_set():
            # Check runtime limit
            if cfg.RUNTIME_SECONDS and (time.time() - start_time > cfg.RUNTIME_SECONDS):
                logger.info("Runtime limit reached.")
                break

            did_work = False
            now = time.perf_counter()

            # --- Phase 1: Data Collection ---
            # Poll all stream queues for new data
            for sid, q in enumerate(stream_queues):
                try:
                    # Non-blocking get
                    item = q.get_nowait()  # (sid, frame, ts)

                    # Determine target worker (Round-Robin)
                    w_idx = sid % total_workers

                    # If this is the first item in the batch, start the timer
                    if not pending_batches[w_idx]:
                        batch_start_times[w_idx] = now

                    pending_batches[w_idx].append(item)
                    did_work = True
                except queue.Empty:
                    pass

            # --- Phase 2: Batch Dispatch ---
            # Check accumulated buffers for each worker
            for w_idx in range(total_workers):
                batch = pending_batches[w_idx]
                if not batch:
                    continue

                # Dispatch Conditions:
                # 1. Batch is Full: We have enough frames to saturate the worker.
                # 2. Timeout: The first frame has waited too long. We must dispatch
                #    to maintain FPS, even if the batch is not full.
                is_full = len(batch) >= target_batch_size
                time_waited = now - batch_start_times[w_idx]
                is_timeout = time_waited > cfg.max_batch_wait_s

                if is_full or is_timeout:
                    # Extract IDs and Stack Frames
                    sids = [x[0] for x in batch]
                    # Stack numpy arrays into a single block (B, H, W, 3)
                    frames = np.stack([x[1] for x in batch])

                    # Convert to Tensor (Share memory efficiently with uint8)
                    tensor_uint8 = torch.from_numpy(frames)
                    logger.debug(f"Worker {w_idx}: Dispatched batch of {len(sids)} frames.")
                    try:
                        task_queues[w_idx].put_nowait((sids, tensor_uint8))
                        # Batch sent successfully, clear buffer
                        pending_batches[w_idx] = []
                        batch_start_times[w_idx] = None
                        did_work = True
                    except queue.Full:
                        # Drop Strategy:
                        # If worker queue is full, it means GPU is lagging.
                        # We drop the accumulated batch to prioritize realtime latency.
                        pending_batches[w_idx] = []
                        batch_start_times[w_idx] = None
                        logger.warning(f"Worker {w_idx} busy, dropping batch.")

            # --- Phase 3: Result Harvesting ---
            # Retrieve results to prevent queue blocking.
            # (In a real app, you would send these to a Monitor/Visualization process)
            while True:
                try:
                    res = result_queue.get_nowait()
                    # res: (sids, density_maps)
                    total_frames_processed += len(res[0])
                    did_work = True
                except queue.Empty:
                    break

            # --- Phase 4: Logging ---
            if time.time() - fps_log_time > 5.0:
                dur = time.time() - fps_log_time
                fps = total_frames_processed / dur
                logger.info(f"System Throughput: {fps:.1f} FPS (Target: {cfg.NUM_STREAMS * cfg.TARGET_FPS})")
                total_frames_processed = 0
                fps_log_time = time.time()

            # --- Phase 5: CPU Idle Optimization ---
            # If no work was done, sleep briefly to avoid 100% CPU usage
            if not did_work:
                time.sleep(0.001)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        stop_event.set()
        shutdown.set()

        # Cleanup Workers
        for p in workers:
            if p.is_alive():
                p.terminate()
        logger.info("Shutdown complete.")


def main():
    # Environment variable overrides for easy testing
    target_fps = float(os.getenv("FPS", "10.0"))
    num_streams = int(os.getenv("NUM_STREAMS", "22"))
    use_video = os.getenv("USE_VIDEO", "0") == "1"

    visible = os.getenv("CUDA_VISIBLE_DEVICES", "0").strip()
    gpus = tuple(int(x) for x in visible.split(",") if x.strip() != "")

    cfg = Config(
        TARGET_FPS=target_fps,
        NUM_STREAMS=num_streams,
        AVAILABLE_GPUS=gpus if gpus else (0,),
        WORKERS_PER_GPU=1,
        USE_VIDEO=use_video,
        RUNTIME_SECONDS=None,
    )

    logger.info("Starting Optimized Inference Engine...")
    run_scheduler(cfg)


if __name__ == "__main__":
    # Required for CUDA
    mp.set_start_method("spawn", force=True)
    main()
