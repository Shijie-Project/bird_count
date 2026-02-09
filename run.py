import logging
import multiprocessing as mp
import sys
from typing import Optional

from core.config import Config, EnvSettings
from core.task_dispatcher import TaskDispatcher
from core.utils import setup_logging


logger = logging.getLogger("main")


def main():
    envs = EnvSettings()

    setup_logging(debug=envs.debug)

    logger.info(">>> System Boot Sequence Initiated <<<")

    try:
        mp.set_start_method("spawn", force=True)
        logger.info("Multiprocessing Context: spawn (Enforced)")
    except RuntimeError as e:
        logger.warning(f"Multiprocessing context already set: {e}")

    cfg = Config.load(envs)
    dispatcher: Optional[TaskDispatcher] = None

    try:
        logger.info("Initializing Task Dispatcher...")
        dispatcher = TaskDispatcher(cfg)

        logger.info("Services Starting... (Press Ctrl+C to stop)")
        dispatcher.run()  # This blocks until shutdown

    except KeyboardInterrupt:
        logger.info("Shutdown Signal Received (SIGINT).")
    except Exception as e:
        logger.critical(f"Fatal System Error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if dispatcher:
            logger.info("Executing Graceful Shutdown...")
            try:
                dispatcher.cleanup()
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")

        logger.info(">>> System Shutdown Complete <<<")


if __name__ == "__main__":
    main()
