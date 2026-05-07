import logging
import multiprocessing as mp
from typing import Optional

from ..config import Config
from ..memory_manager import SharedMemoryConfig
from .base import BaseHandler
from .monitor import MonitorHandler
from .smart_plug import SmartPlugHandler
from .speaker import SpeakerHandler
from .video_recorder import VideoRecorderHandler


logger = logging.getLogger(__name__)


def init_handlers(
    config: Config,
    shm_config: SharedMemoryConfig,
    ack_queue: "Optional[mp.Queue]" = None,
) -> list[BaseHandler]:
    """
    Factory function to instantiate and return enabled handlers.
    """
    handlers = []

    # 1. Visualization - always register;
    # the handler is runtime-toggleable via the InteractionGUI.
    handlers.append(MonitorHandler(config, shm_config, ack_queue=ack_queue))
    logger.info("Handler Registered: Monitor")

    # 2. Smart Plug Control
    if config.envs.enable_smart_plug:
        handlers.append(SmartPlugHandler(config))
        logger.info("Handler Registered: Smart Plug")

    # 3. Audio Alert
    if config.envs.enable_speaker:
        handlers.append(SpeakerHandler(config))
        logger.info("Handler Registered: Speaker")

    # 4. Continuous Video Recorder - always register;
    # the handler is runtime-toggleable via the debug GUI.
    handlers.append(VideoRecorderHandler(config))
    logger.info("Handler Registered: VideoRecorder")

    return handlers
