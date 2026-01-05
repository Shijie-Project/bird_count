import logging
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import queue
import signal
import threading
import time
import warnings
from dataclasses import dataclass, field
from pprint import pformat
from typing import Literal, Optional

import cv2
import numpy as np
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

from handlers import VisualizationHandler
from models import ShuffleNetV2_x1_0


# --- Configuration & Env ---
class Envs(BaseSettings):
    debug: bool = False
    fps: int = 10
    num_streams: int = 22
    cuda_visible_devices: str = "0"
    num_workers_per_gpu: int = 1
    source: Literal["camera", "video", "rtsp"] = "video"
    enable_monitor: bool = True
    num_buffer: int = 3

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


envs = Envs()


# Setup Logger
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
    INPUT_H: int = 512
    INPUT_W: int = 512

    # Hardware Settings
    AVAILABLE_GPUS: tuple[int, ...] = tuple(range(torch.cuda.device_count()))
    NUM_WORKERS_PER_GPU: int = 1

    # Sources
    SOURCE: Literal["video", "rtsp", "camera"] = field(default="video")
    VIDEO_PATH: tuple[str, ...] = ("./data/bird_count/bird_count_demo.mp4",)
    RTSP_URL: tuple[str, ...] = ("rtsp://127.0.0.1:8554/live/test",)
    CAMERA_ADD: tuple[str, ...] = (
        "http://root:root@138.25.209.105/mjpg/1/video.mjpg",
        "http://root:root@138.25.209.109/mjpg/1/video.mjpg",
        "http://root:root@138.25.209.111/mjpg/1/video.mjpg",
        "http://root:root@138.25.209.112/mjpg/1/video.mjpg",
        "http://root:root@138.25.209.113/mjpg/1/video.mjpg",
        "http://root:root@138.25.209.203/mjpg/1/video.mjpg",
    )

    # Stream
    TARGET_FPS: float = 10.0
    NUM_STREAMS: int = 22
    RUNTIME_SECONDS: Optional[int] = None

    # Buffer
    NUM_BUFFER: int = 3

    # Monitor
    ENABLE_MONITOR: bool = True

    # Auto-calculated fields
    frame_interval_s: float = field(init=False)
    stream_sources: tuple[str, ...] = field(init=False)

    def __post_init__(self):
        # Clamp FPS
        if self.TARGET_FPS < 1 or self.TARGET_FPS > 60:
            logger.warning(f"Clamping Target FPS {self.TARGET_FPS} to [1, 60].")
        self.TARGET_FPS = max(1.0, min(60.0, self.TARGET_FPS))
        self.frame_interval_s = 1.0 / self.TARGET_FPS

        # Setup Sources
        source_maps = {
            "video": self.VIDEO_PATH,
            "rtsp": self.RTSP_URL,
            "camera": self.CAMERA_ADD,
        }
        sources = source_maps[self.SOURCE]
        # Round-robin fill
        if len(sources) < self.NUM_STREAMS:
            sources = sources * (self.NUM_STREAMS // len(sources) + 1)
        self.stream_sources = tuple(sources[: self.NUM_STREAMS])

        logger.info("--- Configuration Initialized ---")
        logger.info(f"FPS: {self.TARGET_FPS} (Interval: {self.frame_interval_s * 1000:.1f}ms)")
        logger.info(f"Streams: {self.NUM_STREAMS}")
        logger.info(f"GPUs: {len(self.AVAILABLE_GPUS)}")
        logger.info(f"Workers: {len(self.AVAILABLE_GPUS) * self.NUM_WORKERS_PER_GPU}")
        logger.info(f"Stream Source(s):\n{pformat(self.stream_sources)}")


@dataclass
class TaskItem:
    """Data sent from Scheduler to GPU Worker (Lightweight indices)."""

    sids: list[int]
    buffer_indices: list[int]  # Which buffer (0 or 1) contains the fresh frame
    timestamps: list[float]
    dispatch_time: float


@dataclass
class ResultItem:
    """Data sent from GPU Worker back to Scheduler/Monitor"""

    sids: list[int]
    outputs: np.ndarray  # Shape: (B, H, W, 3), dtype=uint8
    timestamp: float  # time.time()


# --- Shared Memory Management ---
class SharedMemoryManager:
    """
    Manages a block of shared memory for zero-copy image transfer between processes.
    Implements a Double-Buffer strategy to avoid read/write tearing.
    Structure: [NUM_STREAMS, num_buffers, H, W, 3] (uint8)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Dimensions: Streams x Buffers x H x W x Channels
        self.shape = (cfg.NUM_STREAMS, cfg.NUM_BUFFER, cfg.INPUT_H, cfg.INPUT_W, 3)
        self.dtype = np.uint8
        self.nbytes = int(np.prod(self.shape) * np.dtype(self.dtype).itemsize)
        self.shm_name = f"shm_frames_{time.time()}"
        self.shm: Optional[shm.SharedMemory] = None
        self._linked = False

    def create(self):
        """Allocates shared memory."""
        try:
            self.shm = shm.SharedMemory(create=True, size=self.nbytes, name=self.shm_name)
            self._linked = True
            # Zero out memory
            arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
            arr[:] = 0
            logger.info(f"Shared Memory allocated: {self.nbytes / 1024 / 1024:.2f} MB")
            return self.shm.name
        except Exception as e:
            logger.error(f"Failed to create shared memory: {e}")
            raise

    def close(self):
        """Cleanup."""
        if self.shm:
            self.shm.close()
            if self._linked:
                self.shm.unlink()
                self._linked = False
            logger.info("Shared Memory released.")

    def get_ndarray(self) -> np.ndarray:
        """Returns the numpy view of the shared memory."""
        if not self.shm:
            raise RuntimeError("Shared memory not initialized.")
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)


# --- 1. Frame Grabber (Thread) ---
class FrameGrabber(threading.Thread):
    def __init__(
        self,
        sid: int,
        src: str,
        shm_name: str,
        shm_shape: tuple,
        meta_array: mp.Array,  # Shared array for [timestamp, buffer_idx, new_data_flag]
        stop_event: threading.Event,
        cfg: Config,
    ) -> None:
        super().__init__(daemon=True, name=f"grabber-{sid}")
        self.sid = sid
        self.src = src
        self.shm_name = shm_name
        self.shm_shape = shm_shape
        self.meta_array = meta_array  # Access to shared metadata
        self.stop_event = stop_event
        self.cfg = cfg

        self._interval = cfg.frame_interval_s

    def run(self) -> None:
        # Attach to existing Shared Memory
        try:
            existing_shm = shm.SharedMemory(name=self.shm_name)
            # Create numpy view: [NUM_STREAMS, 2, H, W, 3]
            full_array = np.ndarray(self.shm_shape, dtype=np.uint8, buffer=existing_shm.buf)
            # Slice only my stream: [2, H, W, 3]
            my_buffers = full_array[self.sid]
        except Exception as e:
            logger.error(f"[Stream {self.sid}] Shared Memory attach failed: {e}")
            return

        while not self.stop_event.is_set():
            # Connection Logic
            cap = cv2.VideoCapture(self.src)
            if not cap.isOpened():
                logger.warning(f"[Stream {self.sid}] Connection failed. Retrying in 5s...")
                time.sleep(5)
                continue

            # Optimization: Set backend buffer size to minimal
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            logger.info(f"[Stream {self.sid}] Connected.")

            last_ok_time = time.time()

            # Frame Loop
            while not self.stop_event.is_set():
                if not cap.grab():
                    # Timeout check
                    if time.time() - last_ok_time > 5.0:
                        logger.warning(f"[Stream {self.sid}] No frames for 5s (Timeout). Reconnecting...")
                        break
                    # Timeout check could be added here
                    time.sleep(0.01)
                    continue

                # Throttle check (if reading from file/fast source)
                # For RTSP, the network dictates speed, but this helps alignment
                last_ok_time = time.time()
                next_slot = int(last_ok_time / self._interval) + 1
                target_ts = next_slot * self._interval

                ret, frame = cap.retrieve()
                if not ret:
                    break

                try:
                    # CPU Preprocessing
                    frame = cv2.resize(frame, (self.cfg.INPUT_W, self.cfg.INPUT_H))  # w, h
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Determine write buffer (toggle between 0 and 1)
                    # meta_array structure per stream: [timestamp, current_buffer_idx, is_dirty]
                    # We read current, write to next (1 - current)
                    # Actually, since we are single writer, we can just toggle locally

                    # Read current state
                    with self.meta_array.get_lock():
                        # index 0: timestamp, index 1: buffer_idx, index 2: is_dirty
                        current_buf_idx = int(self.meta_array[1])
                        next_buf_idx = (current_buf_idx + 1) % self.cfg.NUM_BUFFER

                    # Write to Shared Memory (Zero Copy to other processes)
                    np.copyto(my_buffers[next_buf_idx], frame)

                    # Wait for target time (Alignment)
                    wait = target_ts - time.time()
                    if wait > 0:
                        time.sleep(wait)

                    now = time.time()
                    # logger.debug(f"[Stream {self.sid}] Frame grabbed @ {now:.3f}s.")

                    # Update Metadata atomically
                    with self.meta_array.get_lock():
                        self.meta_array[0] = now  # Timestamp
                        self.meta_array[1] = next_buf_idx  # New valid buffer
                        self.meta_array[2] = 1.0  # Set Dirty Flag (New Data Available)

                except Exception as e:
                    logger.error(f"[Stream {self.sid}] Process error: {e}")
                    break

            cap.release()
            existing_shm.close()  # Close handle for this thread (not unlink)


class ResultProcessor(mp.Process):
    def __init__(self, result_queue, stop_event, handlers: list, cfg: Config):
        super().__init__(name="Result-Processor", daemon=True)
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.handlers = handlers
        self.cfg = cfg

    def run(self):
        for h in self.handlers:
            h.setup()

        logger.info("[Post-Processor] Started.")

        while not self.stop_event.is_set():
            try:
                item = self.result_queue.get(timeout=0.01)
                for h in self.handlers:
                    h.handle(item.sids, item.outputs, item.timestamp)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Result Processor Error: {e}")

        for h in self.handlers:
            h.cleanup()


# --- 2. Inference Worker Process ---
def inference_worker(
    worker_id: int,
    gpu_id: int,
    cfg: Config,
    task_queue: "mp.Queue[TaskItem]",
    result_queue: "mp.Queue[ResultItem]",
    shm_name: str,
    shm_shape: tuple,
    shutdown: "mp.Event",
) -> None:
    """
    GPU Worker. Reads directly from Shared Memory based on indices received.
    """
    # 1. Setup GPU
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    # 2. Attach Shared Memory
    try:
        existing_shm = shm.SharedMemory(name=shm_name)
        # Full view: [NUM_STREAMS, 2, H, W, 3]
        shm_array = np.ndarray(shm_shape, dtype=np.uint8, buffer=existing_shm.buf)
    except Exception as e:
        logger.error(f"[Worker {worker_id}] SHM Error: {e}")
        return

    # 3. Load Model
    model = ShuffleNetV2_x1_0()
    try:
        st = torch.load(cfg.MODEL_PATH, map_location=device)
        model.load_state_dict(st)
        model.to(device).eval()
    except Exception as e:
        logger.error(f"[Worker {worker_id}] Model Load Failed: {e}")
        return

    # 4. Pre-allocate GPU constants
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Visualization params
    alpha, beta, threshold = 0.5, 0.5, 0.01

    logger.info(f"[Worker {worker_id}] Ready on GPU {gpu_id}.")

    while not shutdown.is_set():
        try:
            task = task_queue.get(timeout=0.01)
        except queue.Empty:
            continue

        try:
            # --- Efficient Data Loading ---
            # Extract specific buffers for this batch.
            # Note: numpy indexing with lists copies data, but this copy is unavoidable
            # to form a contiguous batch. It is fast in memory.
            # shm_array[sid, buf_idx]
            t0 = time.perf_counter()
            wall_now = time.time()

            queue_wait_ms = (wall_now - task.dispatch_time) * 1000
            avg_frame_age_ms = sum([(wall_now - ts) for ts in task.timestamps]) / len(task.timestamps) * 1000

            # Advanced Slicing: shm_array[task.sids, task.buffer_indices]
            # selects the correct buffer for each stream in the batch
            batch_frames_np = shm_array[task.sids, task.buffer_indices]

            # Convert to Tensor (Pin Memory behavior is automatic if passing from CPU tensor usually,
            # but here we go direct. For max speed, we could use a pinned CPU buffer,
            # but let's keep it simple and robust).
            input_tensor = torch.from_numpy(batch_frames_np).to(device, non_blocking=True)

            # Preprocessing on GPU
            input_tensor = input_tensor.permute(0, 3, 1, 2).float().div(255.0)
            input_tensor = (input_tensor - mean) / std

            t1 = time.perf_counter()  # Data Prep Done

            # Inference
            with torch.inference_mode():
                with torch.amp.autocast("cuda"):
                    outputs, _ = model(input_tensor)

            torch.cuda.synchronize()
            t2 = time.perf_counter()  # Inference Done

            # --- Post Processing (Visualization) ---
            # Moving back to CPU for OpenCV ops (since cv2 functions run on CPU)
            # If strictly optimizing, visualization should happen in a separate process
            # or purely via GPU shaders, but here we follow original logic.

            outputs_cpu = outputs.detach().float().cpu().numpy()

            # We need the original images for blending. They are in batch_frames_np
            # Visualization logic mirrored from original
            batch_vis = []

            for i in range(len(task.sids)):
                vis_map = outputs_cpu[i, 0]
                orig_img = batch_frames_np[i]  # RGB

                # Normalize density map
                norm_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min() + 1e-5)
                norm_map = cv2.resize(norm_map, (orig_img.shape[1], orig_img.shape[0]))

                mask = norm_map > threshold

                # We need BGR for OpenCV saving/display usually, but Handler might expect RGB.
                # Assuming Handler expects RGB based on original code flow.
                overlay = orig_img.copy()

                if mask.any():
                    dm_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    # applyColorMap returns BGR, we need RGB if orig is RGB
                    dm_color = cv2.cvtColor(dm_color, cv2.COLOR_BGR2RGB)

                    roi_orig = orig_img[mask]
                    roi_heat = dm_color[mask]
                    blended = cv2.addWeighted(roi_orig, alpha, roi_heat, beta, 0)
                    overlay[mask] = blended

                batch_vis.append(overlay)

            t3 = time.perf_counter()  # Post-proc Done

            # Send result
            final_output = np.stack(batch_vis)
            # We send numpy array instead of Tensor to avoid pickling CUDA tensors to CPU queues
            result_queue.put(ResultItem(task.sids, final_output, time.time()))

            # --- Log Analysis ---
            # Data Prep Time
            t_data_ms = (t1 - t0) * 1000
            # GPU Inference Time
            t_infer_ms = (t2 - t1) * 1000
            # CPU Post-Process Time
            t_post_ms = (t3 - t2) * 1000
            # Total Step Time
            t_total_ms = (t3 - t0) * 1000

            # Format nicely
            log_msg = (
                f"[Worker {worker_id}] Batch={len(task.sids)} | "
                f"BufIdx={task.buffer_indices} | "
                f"FrameAge={avg_frame_age_ms:.1f}ms | "
                f"Q-Wait={queue_wait_ms:.1f}ms | "
                f"DataPrep={t_data_ms:.1f}ms | "
                f"GPU-Infer={t_infer_ms:.1f}ms | "
                f"PostProc={t_post_ms:.1f}ms | "
                f"TOTAL={t_total_ms:.1f}ms"
            )
            logger.debug(log_msg)

        except Exception as e:
            logger.error(f"[Worker {worker_id}] Inference Error: {e}")
            import traceback

            traceback.print_exc()

    existing_shm.close()
    logger.info(f"[Worker {worker_id}] Shutdown.")


# --- 4. Main Engine / Scheduler ---
def run_scheduler(cfg: Config):
    """
    Central Controller.
    1. Manages Shared Memory.
    2. Spawns Workers & Grabbers.
    3. Dispatches Tasks based on stream status.
    """
    # Signal handling for clean cleanup
    shutdown_event = mp.Event()

    def signal_handler(sig, frame):
        logger.info("Signal received, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 1. Initialize Shared Memory Manager
    shm_mgr = SharedMemoryManager(cfg)
    shm_name = shm_mgr.create()

    # 2. Shared Metadata Arrays
    # Format per stream: [timestamp (double), buffer_idx (double), is_dirty (double)]
    # Using 'd' (double) for all to keep simple array structure
    # is_dirty: 1.0 means new frame ready, 0.0 means processed
    meta_arrays = [mp.Array("d", 3) for _ in range(cfg.NUM_STREAMS)]

    # 3. Setup Workers Variables
    ctx = mp.get_context("spawn")
    total_workers = len(cfg.AVAILABLE_GPUS) * cfg.NUM_WORKERS_PER_GPU
    task_queues = [ctx.Queue(maxsize=2) for _ in range(total_workers)]  # Small buffer to prevent latency
    result_queue = ctx.Queue(maxsize=total_workers * 2)

    worker_ready = mp.Array("i", total_workers)
    for i in range(total_workers):
        worker_ready[i] = 0

    # 4. Start Workers FIRST
    logger.info(f"Spawning {total_workers} GPU workers...")
    workers = []
    w_idx = 0
    for gpu in cfg.AVAILABLE_GPUS:
        for _ in range(cfg.NUM_WORKERS_PER_GPU):
            p = ctx.Process(
                target=inference_worker,
                args=(
                    w_idx,
                    gpu,
                    cfg,
                    task_queues[w_idx],
                    result_queue,
                    shm_name,
                    shm_mgr.shape,
                    shutdown_event,
                    worker_ready,
                ),
                name=f"worker-{w_idx}",
                daemon=True,
            )
            p.start()
            workers.append(p)
            w_idx += 1

    # 5. [KEY FIX] Startup Barrier
    # Wait until ALL workers signal they are initialized.
    # This prevents the Scheduler from filling queues before consumers are ready.
    logger.info("Waiting for workers to initialize (load models)...")
    try:
        while not shutdown_event.is_set():
            ready_count = 0
            with worker_ready.get_lock():
                ready_count = sum(worker_ready)

            if ready_count == total_workers:
                logger.info("All Workers READY. Starting Pipeline.")
                break

            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown_event.set()
        shm_mgr.close()
        return

    # 6. Start Result Processor
    handlers = [VisualizationHandler(cfg)]
    res_proc = ResultProcessor(result_queue, shutdown_event, handlers, cfg)
    res_proc.start()

    # 7. Start Grabbers (Now it's safe to start producing data)
    grabber_stop = threading.Event()
    grabbers = []
    for i, src in enumerate(cfg.stream_sources):
        t = FrameGrabber(
            sid=i,
            src=src,
            shm_name=shm_name,
            shm_shape=shm_mgr.shape,
            meta_array=meta_arrays[i],
            stop_event=grabber_stop,
            cfg=cfg,
        )
        t.start()
        grabbers.append(t)

    logger.info("Scheduler loop starting...")

    # --- Main Loop ---
    start_time = time.time()

    try:
        while not shutdown_event.is_set():
            loop_start = time.time()
            next_slot = int(loop_start / cfg.frame_interval_s) + 1
            target_ts = next_slot * cfg.frame_interval_s

            if cfg.RUNTIME_SECONDS and (loop_start - start_time > cfg.RUNTIME_SECONDS):
                logger.info("Runtime limit reached.")
                break

            # A. Identify Streams with New Data
            # We iterate metadata. Accessing mp.Array is faster than queue.
            # Strategy: Round Robin dispatch

            # Group streams by which worker they belong to (Static Load Balancing)
            # or Dynamic. Let's do Dynamic accumulation.

            pending_tasks = [[] for _ in range(total_workers)]

            for sid in range(cfg.NUM_STREAMS):
                ma = meta_arrays[sid]
                # Non-blocking check first
                if ma[2] > 0.5:  # is_dirty == 1.0
                    with ma.get_lock():
                        if ma[2] > 0.5:
                            # Capture state
                            ts = ma[0]
                            buf_idx = int(ma[1])
                            # Clear dirty flag immediately so Grabber knows it's consumed
                            # (Though Grabber doesn't really check this, it just overwrites next buffer)
                            ma[2] = 0.0

                            # Assign to worker (Round Robin based on SID)
                            w_target = sid % total_workers
                            pending_tasks[w_target].append((sid, buf_idx, ts))

            # B. Dispatch Batches
            for w_id in range(total_workers):
                batch = pending_tasks[w_id]
                if not batch:
                    continue

                # Unpack batch
                sids = [b[0] for b in batch]
                buf_idxs = [b[1] for b in batch]
                timestamps = [b[2] for b in batch]

                try:
                    task_queues[w_id].put_nowait(
                        TaskItem(
                            sids=sids,
                            buffer_indices=buf_idxs,
                            timestamps=timestamps,
                            dispatch_time=time.time(),
                        )
                    )
                except queue.Full:
                    # Drop frame if worker is overwhelmed (Latency > Throughput)
                    logger.debug(f"[Worker {w_id}] Task Queue Full.")
                    pass

            # C. Rate Limit / Idle Wait
            wait = target_ts - time.time()
            if wait > 0:
                time.sleep(wait)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception as e:
        logger.error(f"Scheduler Fatal Error: {e}")
    finally:
        logger.info("Stopping components...")
        shutdown_event.set()
        grabber_stop.set()

        for t in grabbers:
            t.join(timeout=1.0)

        for p in workers:
            p.terminate()

        res_proc.terminate()

        # Vital: Cleanup Shared Memory
        shm_mgr.close()
        logger.info("System Shutdown Complete.")


def main():
    # Use config from envs
    gpus = tuple(int(x) for x in envs.cuda_visible_devices.split(",") if x.strip() != "")

    cfg = Config(
        TARGET_FPS=envs.fps,
        NUM_STREAMS=envs.num_streams,
        AVAILABLE_GPUS=gpus if gpus else (0,),
        NUM_WORKERS_PER_GPU=envs.num_workers_per_gpu,
        NUM_BUFFER=envs.num_buffer,
        SOURCE=envs.source,  # type: ignore
        ENABLE_MONITOR=envs.enable_monitor,
    )

    # Required for Shared Memory & CUDA in subprocesses
    mp.set_start_method("spawn", force=True)

    run_scheduler(cfg)


if __name__ == "__main__":
    main()
