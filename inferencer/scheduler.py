import logging
import multiprocessing as mp
import queue
import threading
import time
import traceback
from typing import Optional

from .config import Config, TaskItem
from .grabber import spawn_grabbers
from .handlers import Handlers
from .memory_manager import SharedMemoryManager, SharedMetadataManager
from .processor import ResultProcessor
from .workers import WorkerManager


logger = logging.getLogger(__name__)


class Scheduler:
    """
    Central Orchestrator.
    Manages Shared Memory, Grabbers, Workers, and the Dispatch Loop.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Synchronization Events
        self.shutdown_event = mp.Event()
        self.grabber_stop_event = threading.Event()

        # Components (Initialized in start)
        self.shm_mgr_frames: Optional[SharedMemoryManager] = None
        self.shm_mgr_meta: Optional[SharedMetadataManager] = None
        self.worker_manager: Optional[WorkerManager] = None
        self.result_processor: Optional[ResultProcessor] = None

        # Resources
        self.grabbers: list[threading.Thread] = []
        self.workers: list[mp.Process] = []
        self.task_queues: list[mp.Queue] = []
        self.result_queue: Optional[mp.Queue] = None
        self.worker_ready: Optional[mp.Array] = None

        # SHM Data Pointers
        self.shm_frames_info = None
        self.shm_meta_info = None
        self.latest_cursor = None

    def _setup_resources(self):
        """Step 1 & 2: Init Shared Memory."""
        logger.info("Initializing Shared Memory...")
        self.shm_mgr_frames = SharedMemoryManager(self.cfg)
        self.shm_frames_info = self.shm_mgr_frames.create()

        self.shm_mgr_meta = SharedMetadataManager(self.cfg)
        self.shm_meta_info, self.latest_cursor = self.shm_mgr_meta.create()

    def _start_components(self):
        """Step 3, 4 & 5: Spawn Threads and Processes."""
        logger.info("Spawning components...")

        # 1. Start Grabbers (Input)
        self.grabbers = spawn_grabbers(
            self.cfg, self.shm_frames_info, self.shm_meta_info, self.latest_cursor, self.grabber_stop_event
        )

        # 2. Start Workers (Inference)
        self.worker_manager = WorkerManager(self.cfg, self.shm_frames_info)
        self.workers, self.task_queues, self.result_queue, self.worker_ready = self.worker_manager.start()

        # 3. Start Result Processor (Output)
        self.result_processor = ResultProcessor(
            self.result_queue,
            self.shutdown_event,
            [handler(self.cfg) for handler in Handlers],
            self.shm_frames_info,
            self.shm_meta_info,
            self.cfg,
        )
        self.result_processor.start()

    def _wait_for_workers_ready(self):
        """Step 6: Barrier to ensure models are loaded before dispatching."""
        logger.info("Waiting for workers to initialize (load models)...")
        total_workers = len(self.workers)

        while not self.shutdown_event.is_set():
            ready_count = 0
            # 使用 try-except 防止在关闭时访问已关闭的资源
            try:
                with self.worker_ready.get_lock():
                    ready_count = sum(self.worker_ready)
            except (OSError, ValueError):
                break

            if ready_count == total_workers:
                logger.info("All Workers READY. Starting Pipeline.")
                return
            time.sleep(0.5)

    def _dispatch_loop(self):
        """
        Main Loop: Scans SHM -> Locks Frames -> Dispatches to Worker Queues.
        """
        start_time = time.time()
        logger.info("Scheduler loop started.")

        while not self.shutdown_event.is_set():
            loop_start = time.time()

            # FPS Control Logic
            next_slot = int(loop_start / self.cfg.stream.frame_interval_s) + 1
            target_ts = next_slot * self.cfg.stream.frame_interval_s

            # Runtime Limit Check
            if self.cfg.stream.runtime_seconds and (loop_start - start_time > self.cfg.stream.runtime_seconds):
                logger.info("Runtime limit reached.")
                self.shutdown_event.set()
                break

            # --- 1. Accumulate Tasks ---
            pending_tasks = [[] for _ in range(self.cfg.hardware.total_workers)]

            for sid in range(self.cfg.stream.num_streams):
                idx = self.latest_cursor[sid]

                if idx == -1:
                    continue

                # Check State: Is it Ready (2.0)?
                # Direct access to SHM buffer via manager
                if self.shm_mgr_meta.buffer[sid, idx, 0] > 1.5:
                    # [LOCK] Mark as Locked (1.0) immediately
                    self.shm_mgr_meta.buffer[sid, idx, 0] = 1.0
                    ts = self.shm_mgr_meta.buffer[sid, idx, 1]

                    # Round Robin Dispatch
                    w_target = sid % self.cfg.hardware.total_workers
                    pending_tasks[w_target].append((sid, idx, ts))

            # --- 2. Dispatch ---
            for w_id in range(self.cfg.hardware.total_workers):
                batch = pending_tasks[w_id]
                if not batch:
                    continue

                try:
                    self.task_queues[w_id].put_nowait(
                        TaskItem(
                            sids=[b[0] for b in batch],
                            buffer_indices=[b[1] for b in batch],
                            timestamps=[b[2] for b in batch],
                            dispatch_time=time.time(),
                        )
                    )
                except queue.Full:
                    logger.debug(f"[Worker {w_id}] Queue Full. Dropping {len(batch)} frames.")
                    # Rollback: Set status to 0.0 (Free) so they are overwritten
                    for sid, idx, _ in batch:
                        self.shm_mgr_meta.buffer[sid, idx, 0] = 0.0

            # --- 3. FPS Sleep ---
            wait = target_ts - time.time()
            if wait > 0:
                time.sleep(wait)

    def cleanup(self):
        """Graceful Shutdown & Resource Release."""
        logger.info("Stopping components...")

        self.shutdown_event.set()
        self.grabber_stop_event.set()

        # Stop Result Processor
        if self.result_processor and self.result_processor.is_alive():
            self.result_processor.terminate()

        # Stop Workers
        if self.worker_manager:
            self.worker_manager.stop()  # 使用 WorkerManager 新写的 stop 方法

        # Stop Grabbers
        for t in self.grabbers:
            t.join(timeout=1.0)

        # Close SHM
        if self.shm_mgr_frames:
            self.shm_mgr_frames.close()
        if self.shm_mgr_meta:
            self.shm_mgr_meta.close()

        logger.info("System Shutdown Complete.")

    def run(self):
        """Entry point to run the whole pipeline."""
        try:
            self._setup_resources()
            self._start_components()
            self._wait_for_workers_ready()
            self._dispatch_loop()

        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt. Stopping...")
        except Exception as e:
            logger.error(f"Scheduler Fatal Error: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()
