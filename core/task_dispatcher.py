import logging
import multiprocessing as mp
import sys
import time
from typing import Optional

# --- Core Components ---
from .config import Config

# --- Business Handlers ---
from .handlers import init_handlers
from .inference_process import InferenceProcess
from .memory_manager import SharedMemoryManager
from .result_process import ResultProcess
from .stream_capture import GrabberProcess
from .utils import setup_logging


logger = logging.getLogger(__name__)


class TaskDispatcher:
    """
    High-Performance Orchestrator for Multi-Process Video Analytics.

    Optimizations:
    - Parallel Inference: Supports multiple GPU workers to maximize throughput.
    - Process Priority: Assigns different 'niceness' to ensure Grabber stability.
    - High-Frequency Supervision: 100ms heartbeat to detect failures instantly.
    - Graceful Teardown: Sequential shutdown with mandatory SHM unlinking.
    """

    def __init__(self, config: Config, name="TaskDispatcher"):
        self.name = name
        self.config = config
        self._running = False

        # Communication: High-capacity queue to handle backpressure during bursts
        # Using SimpleQueue on Linux is faster as it avoids background feeder threads
        queue_size = self.config.envs.num_workers_per_gpu * 10
        self.result_queue = mp.Queue(maxsize=queue_size)

        # Resource Management
        self.shm_manager: Optional[SharedMemoryManager] = None

        # Process Registry
        self.grabber: Optional[GrabberProcess] = None
        self.inferencers: list[InferenceProcess] = []
        self.consumer: Optional[ResultProcess] = None

    def _init_resources(self):
        """Allocates Shared Memory blocks via the Manager."""
        logger.info(f"[{self.name}] Allocating Shared Memory...")

        resolution = (self.config.shm.height, self.config.shm.width)
        self.shm_manager = SharedMemoryManager(
            name_prefix=self.config.shm.name_prefix,
            num_streams=self.config.num_streams,
            num_buffers=self.config.num_buffers,
            resolution=resolution,
        )

    def run(self):
        """Main entry point: Launches processes and blocks until shutdown."""
        setup_logging(debug=self.config.envs.debug)

        try:
            # 1. Resource Setup
            self._init_resources()
            shm_config = self.shm_manager.get_config()

            # 2. Component Initialization
            logger.info(f"[{self.name}] Initializing system components...")

            # A. Result Consumer (Sink)
            self.consumer = ResultProcess(config=self.config, shm_config=shm_config, result_queue=self.result_queue)
            for handler in init_handlers(self.config, shm_config):
                self.consumer.register_handler(handler)

            # B. Multiple Inference Engines (Workers)
            num_workers = self.config.envs.num_workers_per_gpu
            for i in range(num_workers):
                worker = InferenceProcess(
                    config=self.config,
                    shm_config=shm_config,
                    result_queue=self.result_queue,
                    worker_id=i,
                    total_workers=num_workers,
                )
                self.inferencers.append(worker)

            # C. Frame Grabber (Source)
            self.grabber = GrabberProcess(config=self.config, shm_config=shm_config)

            # 3. Execution Sequence (Reverse Dependency Order)
            # Starting Consumer first to ensure Queue is monitored
            self.consumer.start()

            # Starting Inference Workers
            for inf in self.inferencers:
                inf.start()

            # Starting Grabber last to begin data flow
            self.grabber.start()

            self._running = True

            self._supervisor_loop()

        except KeyboardInterrupt:
            logger.info(f"[{self.name}] Interruption received (Ctrl+C).")
        except Exception as e:
            logger.critical(f"[{self.name}] Global Dispatcher Failure: {e}", exc_info=True)
            sys.exit(1)
        finally:
            # 5. Global Cleanup (Safety Net)
            self.cleanup()

    def _supervisor_loop(self):
        """Health check loop with 10Hz frequency."""
        while self._running:
            # Aggregate all active processes for monitoring
            procs = [self.grabber] + self.inferencers + [self.consumer]

            for proc in procs:
                if proc and not proc.is_alive():
                    logger.error(f"[{self.name}] FATAL: {proc.name} process has died. Triggering emergency shutdown.")
                    self._running = False
                    return

            # Short sleep to maintain low CPU usage but high responsiveness
            time.sleep(0.1)

    def cleanup(self):
        """Strict resource reclamation and process termination."""
        logger.info(f"[{self.name}] Initiating graceful shutdown...")
        self._running = False

        # 1. Send Stop Signal to all children
        if self.grabber:
            self.grabber.stop()
        for inf in self.inferencers:
            inf.stop()
        if self.consumer:
            self.consumer.stop()

        # 2. Wait for processes to exit (2s timeout)
        all_procs = [self.grabber] + self.inferencers + [self.consumer]
        for proc in all_procs:
            if proc:
                proc.join(timeout=2.0)
                if proc.is_alive():
                    logger.warning(f"[{self.name}] Process {proc.name} failed to stop. Terminating...")
                    proc.terminate()
                    proc.join(0.5)
                    if proc.is_alive():
                        proc.kill()  # Final attempt

        # 3. Unlink Shared Memory (Prevent /dev/shm leaks)
        if self.shm_manager:
            self.shm_manager.cleanup()
