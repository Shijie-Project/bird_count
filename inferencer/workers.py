import logging
import multiprocessing as mp
import multiprocessing.shared_memory as shm
import queue
import time
import traceback
from typing import Optional

import numpy as np
import torch

from models.shufflenet import get_shufflenet_density_model

from .config import Config, ResultItem, TaskItem
from .memory_manager import SharedMemoryInfo


logger = logging.getLogger(__name__)


class InferenceWorker:
    """
    Inference worker unit running in a separate process.
    Responsibilities: Initialize model -> Attach SHM -> Inference Loop.
    """

    def __init__(
        self,
        worker_id: int,
        gpu_id: int,
        cfg: "Config",
        task_queue: "mp.Queue[TaskItem]",
        result_queue: "mp.Queue[ResultItem]",
        shm_frames_info: SharedMemoryInfo,
        shutdown_event: "mp.Event",
        worker_ready_array: "mp.Array",
    ):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.cfg = cfg
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.shm_frames_info = shm_frames_info
        self.shutdown_event = shutdown_event
        self.worker_ready_array = worker_ready_array

        # Runtime State - Initialized in run()
        self.device = None
        self.model = None
        self.mean = None
        self.std = None
        self.shm_handle = None
        self.shm_array = None

    def _init_model(self):
        """Step 1: Initialize Device and PyTorch Model."""
        self.device = torch.device(f"cuda:{self.gpu_id}")
        torch.cuda.set_device(self.device)
        torch.backends.cudnn.benchmark = True

        # Assuming get_shufflenet_density_model is defined elsewhere
        # from models import get_shufflenet_density_model
        self.model = get_shufflenet_density_model()

        st = torch.load(self.cfg.model.path, map_location=self.device)
        self.model.load_state_dict(st)
        self.model.to(self.device).eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _attach_shm(self):
        """Step 2: Attach to Shared Memory."""
        self.shm_handle = shm.SharedMemory(name=self.shm_frames_info.name)
        self.shm_array = np.ndarray(
            self.shm_frames_info.shape, dtype=self.shm_frames_info.dtype, buffer=self.shm_handle.buf
        )

    def _warmup(self):
        """Step 3: Model Warmup."""
        with torch.inference_mode():
            with torch.amp.autocast("cuda"):
                for bsz in range(1, self.cfg.stream.num_streams + 1):
                    # Mimic actual input: (B, H, W, 3) -> Permute -> Float
                    dummy = torch.zeros(
                        (bsz, self.cfg.model.input_h, self.cfg.model.input_w, 3), dtype=torch.uint8, device=self.device
                    )
                    input_tensor = dummy.permute(0, 3, 1, 2).float().div(255.0)
                    input_tensor = (input_tensor - self.mean) / self.std
                    _ = self.model(input_tensor)
                torch.cuda.synchronize()

    def _process_batch(self, task: "TaskItem"):
        """Step 4: Process a single batch (Core Inference Logic)."""
        t0 = time.perf_counter()

        # 1. Load Data (Zero Copy from SHM)
        batch_frames_np = self.shm_array[task.sids, task.buffer_indices]
        input_tensor = torch.from_numpy(batch_frames_np).to(self.device, non_blocking=True)

        # 2. Preprocess
        input_tensor = input_tensor.permute(0, 3, 1, 2).float().div(255.0)
        input_tensor = (input_tensor - self.mean) / self.std
        t1 = time.perf_counter()

        # 3. Infer
        with torch.inference_mode():
            with torch.amp.autocast("cuda"):
                outputs = self.model(input_tensor)

        torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Convert to CPU numpy array
        outputs_cpu = outputs.detach().float().cpu().numpy()

        self.result_queue.put(
            ResultItem(
                sids=task.sids,
                buffer_indices=task.buffer_indices,
                outputs=outputs_cpu,
                timestamp=time.time(),
            )
        )

        return outputs, t0, t1, t2

    def _log_performance(self, task, t0, t1, t2):
        t3 = time.perf_counter()
        wall_now = time.time()
        queue_wait_ms = (wall_now - task.dispatch_time) * 1000
        avg_frame_age_ms = sum([(wall_now - ts) for ts in task.timestamps]) / len(task.timestamps) * 1000

        log_msg = (
            f"[Worker {self.worker_id}] Batch={len(task.sids)} | "
            f"BufIdx={task.buffer_indices} | "
            f"FrameAge={avg_frame_age_ms:.1f}ms | "
            f"Q-Wait={queue_wait_ms:.1f}ms | "
            f"DataPrep={(t1 - t0) * 1000:.1f}ms | "
            f"Infer={(t2 - t1) * 1000:.1f}ms | "
            f"Total={(t3 - t0) * 1000:.1f}ms"
        )
        logger.debug(log_msg)

    def run(self):
        """Process Entry Point (Replaces original inference_worker function)."""
        try:
            self._init_model()
            self._attach_shm()
            self._warmup()

            with self.worker_ready_array.get_lock():
                self.worker_ready_array[self.worker_id] = 1

            logger.info(f"[Worker {self.worker_id}] Initialized on GPU {self.gpu_id}. Signalling READY.")

        except Exception as e:
            logger.error(f"[Worker {self.worker_id}] Init Fatal Error: {e}")
            traceback.print_exc()
            return

        # Main Loop
        while not self.shutdown_event.is_set():
            try:
                task = self.task_queue.get(timeout=0.005)
            except queue.Empty:
                continue

            try:
                outputs, t0, t1, t2 = self._process_batch(task)
                self._log_performance(task, t0, t1, t2)
            except Exception as e:
                logger.error(f"[Worker {self.worker_id}] Inference Error: {e}")
                traceback.print_exc()

        # Cleanup
        if self.shm_handle:
            self.shm_handle.close()
        logger.info(f"[Worker {self.worker_id}] Shutdown.")


class WorkerManager:
    """
    Manages the lifecycle of all Worker processes.
    """

    def __init__(self, cfg, shm_frames_info: SharedMemoryInfo):
        self.cfg = cfg
        self.shm_frames_info = shm_frames_info

        self.gpus = self.cfg.hardware.available_gpus
        self.workers_per_gpu = self.cfg.hardware.num_workers_per_gpu
        self.total_workers = self.cfg.hardware.total_workers

        # Resource Holders
        self.workers: list[mp.Process] = []
        self.task_queues: list[mp.Queue] = []
        self.result_queue: Optional[mp.Queue] = None
        self.worker_ready_array: Optional[mp.Array] = None

        self.ctx = mp.get_context("spawn")
        self.shutdown_event = self.ctx.Event()

    def start(self) -> tuple[list[mp.Process], list[mp.Queue], mp.Queue, mp.Array]:
        """Start all Workers."""
        self.task_queues = [self.ctx.Queue(maxsize=2) for _ in range(self.total_workers)]
        self.result_queue = self.ctx.Queue(maxsize=self.total_workers * 2)
        self.worker_ready_array = self.ctx.Array("i", self.total_workers)

        w_idx = 0
        for gpu in self.gpus:
            for _ in range(self.workers_per_gpu):
                # Instantiate Worker Logic
                worker_logic = InferenceWorker(
                    worker_id=w_idx,
                    gpu_id=gpu,
                    cfg=self.cfg,
                    task_queue=self.task_queues[w_idx],
                    result_queue=self.result_queue,
                    shm_frames_info=self.shm_frames_info,
                    shutdown_event=self.shutdown_event,
                    worker_ready_array=self.worker_ready_array,
                )

                # Start Process (target points to the instance's run method)
                p = self.ctx.Process(
                    target=worker_logic.run,
                    name=f"worker-{w_idx}",
                    daemon=True,
                )
                p.start()
                self.workers.append(p)
                w_idx += 1

        return self.workers, self.task_queues, self.result_queue, self.worker_ready_array

    def stop(self):
        """Gracefully stop all Workers."""
        logger.info("Signal shutdown to workers...")
        self.shutdown_event.set()
        for p in self.workers:
            p.join(timeout=2.0)
            if p.is_alive():
                logger.warning(f"Worker {p.name} did not stop gracefully, terminating.")
                p.terminate()
