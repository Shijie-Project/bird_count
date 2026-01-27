import logging
import multiprocessing as mp
import signal


logger = logging.getLogger(__name__)


def setup_shutdown_event():
    shutdown_event = mp.Event()

    def signal_handler(sig, frame):
        logger.info("Signal received, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    return shutdown_event
