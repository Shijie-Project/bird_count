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
from models.shufflenet import get_shufflenet_density_model


# --- Configuration & Env ---
class Envs(BaseSettings):
    debug: bool = False
    fps: int = 10
    num_streams: int = 22
    cuda_visible_devices: str = "0"
    num_workers_per_gpu: int = 1
    source: Literal["camera", "video", "rtsp"] = "video"
    enable_monitor: bool = True
    num_buffer: int = 10  # Increased buffer size to ensure safety with the locking mechanism

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
    NUM_BUFFER: int = 10  # Suggest using >= 5 for stability with locking

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
    """Task payload sent to GPU Worker."""

    sids: list[int]
    buffer_indices: list[int]
    timestamps: list[float]
    dispatch_time: float


@dataclass
class ResultItem:
    """Result payload sent to ResultProcessor."""

    sids: list[int]
    buffer_indices: list[int]  # Needed for ResultProcessor to find original frame in SHM
    outputs: np.ndarray  # Lightweight density maps (Not blended images)
    timestamp: float


# --- Shared Memory Management ---
class SharedMemoryManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Structure: [NUM_STREAMS, NUM_BUFFER, H, W, 3] (uint8)
        self.shape = (cfg.NUM_STREAMS, cfg.NUM_BUFFER, cfg.INPUT_H, cfg.INPUT_W, 3)
        self.dtype = np.uint8
        self.nbytes = int(np.prod(self.shape) * np.dtype(self.dtype).itemsize)
        self.shm_name = f"shm_frames_{time.time()}"
        self.shm: Optional[shm.SharedMemory] = None
        self._linked = False

    def create(self):
        try:
            self.shm = shm.SharedMemory(create=True, size=self.nbytes, name=self.shm_name)
            self._linked = True
            arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
            arr[:] = 0
            logger.info(f"Shared Memory allocated: {self.nbytes / 1024 / 1024:.2f} MB")
            return self.shm.name
        except Exception as e:
            logger.error(f"Failed to create shared memory: {e}")
            raise

    def close(self):
        if self.shm:
            self.shm.close()
            if self._linked:
                self.shm.unlink()
                self._linked = False
            logger.info("Shared Memory released.")


def init_shared_metadata(cfg: Config):
    """
    Initialize shared metadata table and cursor.
    buffer_meta structure: [NUM_STREAMS, NUM_BUFFER, 2] (float64)
       - [..., 0]: State (0=Free, 1=Locked, 2=Ready)
       - [..., 1]: Timestamp
    """
    # 1. Calculate required size
    shape = (cfg.NUM_STREAMS, cfg.NUM_BUFFER, 2)
    dtype = np.float64
    nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)

    # 2. Allocate SHM
    shm_name = f"shm_meta_{time.time()}"
    shm_obj = shm.SharedMemory(create=True, size=nbytes, name=shm_name)

    # 3. Create view and initialize
    buffer_meta = np.ndarray(shape, dtype=dtype, buffer=shm_obj.buf)
    buffer_meta.fill(0)  # Init all as Free

    # 4. Create latest cursor (tracks the last written buffer index for each stream)
    # Init -1 means no data written yet
    latest_cursor = mp.Array("i", cfg.NUM_STREAMS)
    for i in range(cfg.NUM_STREAMS):
        latest_cursor[i] = -1

    return shm_obj, buffer_meta, latest_cursor


def process_grabber_frame(
    sid: int,
    frame: np.ndarray,
    buffer_meta: np.ndarray,  # [NUM_STREAMS, NUM_BUFFER, 2]
    latest_cursor: mp.Array,
    my_buffers: np.ndarray,
    cfg: Config,
    now: float,
):
    """
    Core Grabber Logic:
    1. Determine next buffer index.
    2. Check lock state (Drop if locked).
    3. Write data & update metadata.
    """
    # CPU Preprocessing
    frame = cv2.resize(frame, (cfg.INPUT_W, cfg.INPUT_H))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. Get current cursor position
    current_idx = latest_cursor[sid]

    if current_idx == -1:
        next_idx = 0
    else:
        next_idx = (current_idx + 1) % cfg.NUM_BUFFER

    # 2. Check Buffer State
    # 0.0 = Free, 2.0 = Ready (Overwritable), 1.0 = Locked (Protected)
    state = buffer_meta[sid, next_idx, 0]

    if state == 1.0:
        # [Flow Control] Target buffer is in use. Drop this frame.
        logger.debug(f"[Stream {sid}] Buffer {next_idx} Locked. Dropping frame.")
        return

        # 3. Write Data (Zero Copy)
    np.copyto(my_buffers[next_idx], frame)

    # 4. Update Metadata
    # Mark as Ready (2.0)
    buffer_meta[sid, next_idx, 0] = 2.0
    buffer_meta[sid, next_idx, 1] = now

    # 5. Update Global Cursor
    # Using lock ensures Scheduler reads consistent state if it checks concurrently
    with latest_cursor.get_lock():
        latest_cursor[sid] = next_idx


class FrameGrabber(threading.Thread):
    def __init__(
        self,
        sid: int,
        src: str,
        shm_frames_name: str,
        shm_frames_shape: tuple,
        shm_meta_name: str,
        shm_meta_shape: tuple,
        latest_cursor: mp.Array,
        stop_event: threading.Event,
        cfg: Config,
    ) -> None:
        super().__init__(daemon=True, name=f"grabber-{sid}")
        self.sid = sid
        self.src = src
        self.shm_frames_name = shm_frames_name
        self.shm_frames_shape = shm_frames_shape
        self.shm_meta_name = shm_meta_name
        self.shm_meta_shape = shm_meta_shape
        self.latest_cursor = latest_cursor
        self.stop_event = stop_event
        self.cfg = cfg
        self._interval = cfg.frame_interval_s

    def run(self) -> None:
        # 1. Attach Video Frames SHM
        try:
            existing_shm_frames = shm.SharedMemory(name=self.shm_frames_name)
            full_frames = np.ndarray(self.shm_frames_shape, dtype=np.uint8, buffer=existing_shm_frames.buf)
            my_buffers = full_frames[self.sid]
        except Exception as e:
            logger.error(f"[Stream {self.sid}] Frames SHM attach failed: {e}")
            return

        # 2. Attach Metadata SHM
        try:
            existing_shm_meta = shm.SharedMemory(name=self.shm_meta_name)
            buffer_meta = np.ndarray(self.shm_meta_shape, dtype=np.float64, buffer=existing_shm_meta.buf)
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
                    process_grabber_frame(
                        self.sid, frame, buffer_meta, self.latest_cursor, my_buffers, self.cfg, time.time()
                    )

                    wait = target_ts - time.time()
                    if wait > 0:
                        time.sleep(wait)

                except Exception as e:
                    logger.error(f"[Stream {self.sid}] Process error: {e}")
                    break

            cap.release()
            existing_shm_frames.close()
            existing_shm_meta.close()


class ResultProcessor(mp.Process):
    """
    Processes results from Workers.
    Optimized: Reads original frames from Shared Memory (Zero-Copy) instead of receiving them via Queue.
    Responsible for Unlocking Buffers after processing.
    """

    def __init__(
        self,
        result_queue,
        stop_event,
        handlers: list,
        cfg: Config,
        shm_frames_name: str,
        shm_frames_shape: tuple,
        shm_meta_name: str,
        shm_meta_shape: tuple,
    ):
        super().__init__(name="Result-Processor", daemon=True)
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.handlers = handlers
        self.cfg = cfg
        self.shm_frames_name = shm_frames_name
        self.shm_frames_shape = shm_frames_shape
        self.shm_meta_name = shm_meta_name
        self.shm_meta_shape = shm_meta_shape

    def run(self):
        # 1. Attach Shared Memories
        try:
            shm_frames = shm.SharedMemory(name=self.shm_frames_name)
            full_frames = np.ndarray(self.shm_frames_shape, dtype=np.uint8, buffer=shm_frames.buf)

            shm_meta = shm.SharedMemory(name=self.shm_meta_name)
            buffer_meta = np.ndarray(self.shm_meta_shape, dtype=np.float64, buffer=shm_meta.buf)
        except Exception as e:
            logger.error(f"[ResultProcessor] SHM Attach Failed: {e}")
            return

        for h in self.handlers:
            h.setup()

        logger.info("[Post-Processor] Started.")

        # Visualization Constants
        alpha, beta, threshold = 0.5, 0.5, 0.01

        while not self.stop_event.is_set():
            try:
                item = self.result_queue.get(timeout=0.005)
            except queue.Empty:
                continue

            try:
                # 2. Extract Data
                # item.outputs is the density map (B, 1, H, W) or similar
                # We need to perform visualization here to offload the GPU Worker

                final_vis_batch = []

                # Zero-Copy Read from Shared Memory
                # We use advanced indexing to get the batch of original frames
                batch_orig_frames = full_frames[item.sids, item.buffer_indices]

                for i in range(len(item.sids)):
                    sid = item.sids[i]
                    buf_idx = item.buffer_indices[i]

                    if self.cfg.ENABLE_MONITOR:
                        # Visualization Logic
                        vis_map = item.outputs[i, 0]  # Assuming (B, 1, H, W) layout
                        orig_img = batch_orig_frames[i]

                        norm_map = (vis_map - vis_map.min()) / (vis_map.max() - vis_map.min() + 1e-5)
                        norm_map = cv2.resize(norm_map, (orig_img.shape[1], orig_img.shape[0]))
                        mask = norm_map > threshold

                        overlay = orig_img.copy()
                        if mask.any():
                            dm_color = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
                            dm_color = cv2.cvtColor(dm_color, cv2.COLOR_BGR2RGB)

                            blended = cv2.addWeighted(orig_img[mask], alpha, dm_color[mask], beta, 0)
                            overlay[mask] = blended

                        final_vis_batch.append(overlay)

                    # --- CRITICAL: UNLOCK BUFFER ---
                    # Mark buffer as Free (0.0) so Grabber can reuse it
                    buffer_meta[sid, buf_idx, 0] = 0.0

                if self.cfg.ENABLE_MONITOR and final_vis_batch:
                    # Stack list to numpy array for the handler
                    vis_np = np.stack(final_vis_batch)
                    for h in self.handlers:
                        h.handle(item.sids, vis_np, item.timestamp)

            except Exception as e:
                logger.error(f"Result Processor Error: {e}")
                import traceback

                traceback.print_exc()

        for h in self.handlers:
            h.cleanup()

        shm_frames.close()
        shm_meta.close()


def worker_init_model(gpu_id: int, cfg: Config):
    """Worker Step 1: Initialize device and model."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    model = get_shufflenet_density_model()
    st = torch.load(cfg.MODEL_PATH, map_location=device)
    model.load_state_dict(st)
    model.to(device).eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    return device, model, mean, std


def worker_warmup(device: torch.device, model: torch.nn.Module, mean, std, cfg: Config):
    """Worker Step 2: Warmup execution with correct memory layout."""
    with torch.inference_mode():
        with torch.amp.autocast("cuda"):
            for bsz in range(1, cfg.NUM_STREAMS + 1):
                # Mimic actual input: (B, H, W, 3) -> Permute -> Float
                dummy = torch.zeros((bsz, cfg.INPUT_H, cfg.INPUT_W, 3), dtype=torch.uint8, device=device)
                input_tensor = dummy.permute(0, 3, 1, 2).float().div(255.0)
                input_tensor = (input_tensor - mean) / std
                _ = model(input_tensor)
            torch.cuda.synchronize()


def worker_execute_batch(
    task: TaskItem, shm_array: np.ndarray, device: torch.device, model: torch.nn.Module, mean, std
) -> tuple[torch.Tensor, float, float, float]:
    """Worker Step 3: Core Inference Logic (Data Prep -> Infer)."""
    t0 = time.perf_counter()

    # 1. Load Data (Zero Copy from SHM)
    batch_frames_np = shm_array[task.sids, task.buffer_indices]
    input_tensor = torch.from_numpy(batch_frames_np).to(device, non_blocking=True)

    # 2. Preprocess
    input_tensor = input_tensor.permute(0, 3, 1, 2).float().div(255.0)
    input_tensor = (input_tensor - mean) / std
    t1 = time.perf_counter()

    # 3. Infer
    with torch.inference_mode():
        with torch.amp.autocast("cuda"):
            outputs, _ = model(input_tensor)

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    return outputs, t0, t1, t2


def inference_worker(
    worker_id: int,
    gpu_id: int,
    cfg: Config,
    task_queue: "mp.Queue[TaskItem]",
    result_queue: "mp.Queue[ResultItem]",
    shm_name: str,
    shm_shape: tuple,
    shutdown: "mp.Event",
    worker_ready: "mp.Array",
) -> None:
    try:
        # 1. Init
        device, model, mean, std = worker_init_model(gpu_id, cfg)

        # 2. Attach SHM
        existing_shm = shm.SharedMemory(name=shm_name)
        shm_array = np.ndarray(shm_shape, dtype=np.uint8, buffer=existing_shm.buf)

        # 3. Warmup
        worker_warmup(device, model, mean, std, cfg)

        logger.info(f"[Worker {worker_id}] Initialized on GPU {gpu_id}. Signalling READY.")
        with worker_ready.get_lock():
            worker_ready[worker_id] = 1

    except Exception as e:
        logger.error(f"[Worker {worker_id}] Init Fatal Error: {e}")
        return

    while not shutdown.is_set():
        try:
            task = task_queue.get(timeout=0.005)
        except queue.Empty:
            continue

        try:
            # 4. Infer
            outputs, t0, t1, t2 = worker_execute_batch(task, shm_array, device, model, mean, std)

            # 5. Send Result (Lightweight)
            # Send CPU numpy array of density map. No visualization here!
            outputs_cpu = outputs.detach().float().cpu().numpy()

            result_queue.put(
                ResultItem(
                    sids=task.sids,
                    buffer_indices=task.buffer_indices,  # Pass indices so ResultProcessor can find frames
                    outputs=outputs_cpu,
                    timestamp=time.time(),
                )
            )

            # 6. Log
            t3 = time.perf_counter()
            wall_now = time.time()
            queue_wait_ms = (wall_now - task.dispatch_time) * 1000
            avg_frame_age_ms = sum([(wall_now - ts) for ts in task.timestamps]) / len(task.timestamps) * 1000

            log_msg = (
                f"[Worker {worker_id}] Batch={len(task.sids)} | "
                f"BufIdx={task.buffer_indices} | "
                f"FrameAge={avg_frame_age_ms:.1f}ms | "
                f"Q-Wait={queue_wait_ms:.1f}ms | "
                f"DataPrep={(t1 - t0) * 1000:.1f}ms | "
                f"Infer={(t2 - t1) * 1000:.1f}ms | "
                f"Total={(t3 - t0) * 1000:.1f}ms"
            )
            logger.debug(log_msg)

        except Exception as e:
            logger.error(f"[Worker {worker_id}] Inference Error: {e}")
            import traceback

            traceback.print_exc()

    existing_shm.close()
    logger.info(f"[Worker {worker_id}] Shutdown.")


def spawn_workers(
    ctx, cfg: Config, shm_name: str, shm_shape: tuple, shutdown_event
) -> tuple[list, list, mp.Queue, mp.Array]:
    """Create and start worker processes."""
    total_workers = len(cfg.AVAILABLE_GPUS) * cfg.NUM_WORKERS_PER_GPU
    task_queues = [ctx.Queue(maxsize=2) for _ in range(total_workers)]
    result_queue = ctx.Queue(maxsize=total_workers * 2)
    worker_ready = mp.Array("i", total_workers)

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
                    shm_shape,
                    shutdown_event,
                    worker_ready,
                ),
                name=f"worker-{w_idx}",
                daemon=True,
            )
            p.start()
            workers.append(p)
            w_idx += 1

    return workers, task_queues, result_queue, worker_ready


def spawn_grabbers(
    cfg: Config,
    shm_frames_name: str,
    shm_frames_shape: tuple,
    shm_meta_name: str,
    shm_meta_shape: tuple,
    latest_cursor: mp.Array,
) -> tuple[list, threading.Event]:
    stop_event = threading.Event()
    grabbers = []

    for i, src in enumerate(cfg.stream_sources):
        t = FrameGrabber(
            sid=i,
            src=src,
            shm_frames_name=shm_frames_name,
            shm_frames_shape=shm_frames_shape,
            shm_meta_name=shm_meta_name,
            shm_meta_shape=shm_meta_shape,
            latest_cursor=latest_cursor,
            stop_event=stop_event,
            cfg=cfg,
        )
        t.start()
        grabbers.append(t)
    return grabbers, stop_event


def wait_for_startup_barrier(worker_ready: mp.Array, total_workers: int, shutdown_event: mp.Event, shm_mgr):
    """Wait for all workers to signal readiness."""
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
        raise


def scheduler_dispatch_loop(
    cfg: Config,
    buffer_meta: np.ndarray,
    latest_cursor: mp.Array,
    task_queues: list,
    shutdown_event: mp.Event,
    total_workers: int,
):
    """
    Main Scheduler Loop:
    1. Scan `latest_cursor` for new frames.
    2. Check `buffer_meta` if frame is Ready (2.0).
    3. Lock (1.0) and Dispatch.
    """
    start_time = time.time()

    while not shutdown_event.is_set():
        loop_start = time.time()
        next_slot = int(loop_start / cfg.frame_interval_s) + 1
        target_ts = next_slot * cfg.frame_interval_s

        if cfg.RUNTIME_SECONDS and (loop_start - start_time > cfg.RUNTIME_SECONDS):
            logger.info("Runtime limit reached.")
            break

        # 1. Accumulate Tasks
        pending_tasks = [[] for _ in range(total_workers)]

        for sid in range(cfg.NUM_STREAMS):
            idx = latest_cursor[sid]

            if idx == -1:
                continue

            # Check State: Is it Ready (2.0)?
            if buffer_meta[sid, idx, 0] > 1.5:
                # [LOCK] Mark as Locked (1.0) immediately
                buffer_meta[sid, idx, 0] = 1.0

                ts = buffer_meta[sid, idx, 1]

                # Round Robin Dispatch
                w_target = sid % total_workers
                pending_tasks[w_target].append((sid, idx, ts))

        # 2. Dispatch
        for w_id in range(total_workers):
            batch = pending_tasks[w_id]
            if not batch:
                continue

            try:
                task_queues[w_id].put_nowait(
                    TaskItem(
                        sids=[b[0] for b in batch],
                        buffer_indices=[b[1] for b in batch],
                        timestamps=[b[2] for b in batch],
                        dispatch_time=time.time(),
                    )
                )
            except queue.Full:
                # Rollback: Since we can't process it, release lock?
                # Actually, dropping is safer to avoid backlog.
                # Mark as Free (0.0) effectively dropping the frame logic
                logger.debug(f"[Worker {w_id}] Task queue full. Dropping batch of {len(batch)} frames.")
                for sid, idx, _ in batch:
                    buffer_meta[sid, idx, 0] = 0.0
                pass

        # 3. Wait for FPS Alignment. This is import to ensure we gather enough data for inference.
        wait = target_ts - time.time()
        if wait > 0:
            time.sleep(wait)


def run_scheduler(cfg: Config):
    shutdown_event = mp.Event()

    def signal_handler(sig, frame):
        logger.info("Signal received, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 1. Init Frames Memory
    shm_mgr_frames = SharedMemoryManager(cfg)
    shm_frames_name = shm_mgr_frames.create()

    # 2. Init Metadata Memory
    shm_obj_meta, buffer_meta, latest_cursor = init_shared_metadata(cfg)
    shm_meta_name = shm_obj_meta.name

    # 3. Start Grabbers
    grabbers, grabber_stop = spawn_grabbers(
        cfg, shm_frames_name, shm_mgr_frames.shape, shm_meta_name, buffer_meta.shape, latest_cursor
    )

    # 4. Spawn Workers
    ctx = mp.get_context("spawn")
    workers, task_queues, result_queue, worker_ready = spawn_workers(
        ctx, cfg, shm_frames_name, shm_mgr_frames.shape, shutdown_event
    )

    # 5. Start Result Processor
    # Pass SHM info to RP so it can attach and read original frames
    handlers = [VisualizationHandler(cfg)]
    res_proc = ResultProcessor(
        result_queue,
        shutdown_event,
        handlers,
        cfg,
        shm_frames_name,
        shm_mgr_frames.shape,
        shm_meta_name,
        buffer_meta.shape,
    )
    res_proc.start()

    # 6. Wait for Workers Ready
    try:
        wait_for_startup_barrier(worker_ready, len(workers), shutdown_event, shm_mgr_frames)
    except KeyboardInterrupt:
        return

    logger.info("Scheduler loop starting...")

    try:
        scheduler_dispatch_loop(cfg, buffer_meta, latest_cursor, task_queues, shutdown_event, len(workers))
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception as e:
        logger.error(f"Scheduler Fatal Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        logger.info("Stopping components...")
        shutdown_event.set()
        grabber_stop.set()

        for t in grabbers:
            t.join(timeout=1.0)
        for p in workers:
            p.terminate()
        res_proc.terminate()

        # Cleanup SHM
        shm_mgr_frames.close()

        # Cleanup Meta SHM
        shm_obj_meta.close()
        shm_obj_meta.unlink()

        logger.info("System Shutdown Complete.")


def main():
    gpus = tuple(int(x) for x in envs.cuda_visible_devices.split(",") if x.strip() != "")
    cfg = Config(
        TARGET_FPS=envs.fps,
        NUM_STREAMS=envs.num_streams,
        AVAILABLE_GPUS=gpus if gpus else (0,),
        NUM_WORKERS_PER_GPU=envs.num_workers_per_gpu,
        NUM_BUFFER=envs.num_buffer,
        SOURCE=envs.source,
        ENABLE_MONITOR=envs.enable_monitor,
    )
    mp.set_start_method("spawn", force=True)
    run_scheduler(cfg)


if __name__ == "__main__":
    main()
