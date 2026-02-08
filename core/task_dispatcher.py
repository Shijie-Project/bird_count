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
    Central Orchestrator.

    Responsibilities:
    1. Resource Management: Allocates & Cleans up Shared Memory.
    2. Process Lifecycle: Starts processes in strict dependency order.
    3. Supervisor: Monitors process health and handles shutdown.
    """

    def __init__(self, config: Config):
        self.config = config
        self._running = False

        # --- Communication Channels ---
        # Maxsize limits the number of pending results.
        # If Consumer is slow, Engine blocks, then Grabber drops frames (Backpressure).
        self.result_queue = mp.Queue(maxsize=config.envs.num_workers_per_gpu * 2)

        # --- Resources ---
        self.shm_manager: Optional[SharedMemoryManager] = None

        # --- Processes ---
        self.grabber: Optional[GrabberProcess] = None
        self.inferencer: Optional[InferenceProcess] = None
        self.consumer: Optional[ResultProcess] = None

    def _init_resources(self):
        """
        Initialize System Resources (Shared Memory).
        """
        logger.info("Allocating Shared Memory Resources...")

        # Determine strict resolution from config
        resolution = (self.config.shm.height, self.config.shm.width)

        # Initialize Manager (Allocates /dev/shm)
        self.shm_manager = SharedMemoryManager(
            name_prefix=self.config.shm.name_prefix,
            num_streams=self.config.num_streams,
            num_buffers=self.config.num_buffers,
            resolution=resolution,
        )

        logger.info("Resources Ready.")

    def run(self):
        """
        Main Entry Point. Blocks until system shutdown.
        """
        # Ensure logging is configured for the Main Process
        setup_logging(debug=self.config.envs.debug)

        try:
            # 1. Allocate Resources
            self._init_resources()

            # Get the config object that creates hooks for child processes
            shm_config = self.shm_manager.get_config()

            # 2. Initialize Processes
            logger.info("Initializing Sub-processes...")

            # A. Consumer (The Sink)
            # Must be initialized with handlers to process results
            self.consumer = ResultProcess(config=self.config, shm_config=shm_config, result_queue=self.result_queue)
            handlers = init_handlers(self.config)
            for handler in handlers:
                self.consumer.register_handler(handler)

            # B. Inference Engine (The Processor)
            self.inferencer = InferenceProcess(
                config=self.config, shm_config=shm_config, result_queue=self.result_queue
            )

            # C. Grabber (The Source)
            self.grabber = GrabberProcess(config=self.config, shm_config=shm_config)

            # 3. Start Sequence (Reverse Dependency Order)
            # We start consumers first so they are ready when data arrives.

            self.consumer.start()
            logger.info("-> [1/3] Result Consumer Started.")

            self.inferencer.start()
            logger.info("-> [2/3] Inference Engine Started.")

            self.grabber.start()
            logger.info("-> [3/3] Grabber Process Started.")

            self._running = True

            logger.info(">>> System Operational. Entering Supervisor Loop. <<<")

            # 4. Supervisor Loop
            self._supervisor_loop()

        except KeyboardInterrupt:
            logger.info("Stop Signal Received (Ctrl+C).")
        except Exception as e:
            logger.critical(f"Fatal Error: {e}", exc_info=True)
            sys.exit(1)
        finally:
            # 5. Global Cleanup (Safety Net)
            self.cleanup()

    def _supervisor_loop(self):
        """
        Monitors child processes health.
        If a critical process dies, we trigger a system shutdown (fail-safe).
        """
        while self._running:
            # Check if critical processes are alive
            if not self.grabber.is_alive():
                logger.error("CRITICAL: Grabber Process died unexpectedly!")
                break
            if not self.inferencer.is_alive():
                logger.error("CRITICAL: Inference Engine died unexpectedly!")
                break
            if not self.consumer.is_alive():
                logger.error("CRITICAL: Result Consumer died unexpectedly!")
                break

            # Heartbeat interval
            time.sleep(1.0)

    def cleanup(self):
        """
        Graceful Shutdown Sequence.
        Ensures all processes stop and Shared Memory is unlinked.
        """
        logger.info("Initiating Cleanup...")
        self._running = False

        # 1. Stop Processes
        processes = [("Grabber", self.grabber), ("Inferencer", self.inferencer), ("Consumer", self.consumer)]

        for name, proc in processes:
            if proc and proc.is_alive():
                logger.info(f"Stopping {name}...")
                proc.stop()

        # 2. Wait for Join (Timeout 2s)
        for name, proc in processes:
            if proc:
                proc.join(timeout=2.0)
                if proc.is_alive():
                    logger.warning(f"{name} did not exit gracefully. Terminating.")
                    proc.terminate()

        # 3. Clean Shared Memory
        # This is CRITICAL. If skipped, /dev/shm will fill up.
        if self.shm_manager:
            logger.info("Cleaning Shared Memory...")
            try:
                self.shm_manager.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup SHM: {e}")

        logger.info("Cleanup Complete.")
