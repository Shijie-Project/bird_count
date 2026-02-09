import logging

from ..config import Config
from ..memory_manager import SharedMemoryConfig
from .base import BaseHandler
from .monitor import MonitorHandler
from .smart_plug import SmartPlugHandler
from .speaker import SpeakerHandler


logger = logging.getLogger(__name__)


def init_handlers(config: Config, shm_config: SharedMemoryConfig) -> list[BaseHandler]:
    """
    Factory function to instantiate and return enabled handlers.
    """
    handlers = []

    # 1. Visualization
    if config.envs.enable_monitor:
        handlers.append(MonitorHandler(config, shm_config))
        logger.info("Handler Registered: Monitor")

    # 2. Smart Plug Control
    if config.envs.enable_smart_plug:
        handlers.append(SmartPlugHandler(config))
        logger.info("Handler Registered: Smart Plug")

    # 3. Audio Alert
    if config.envs.enable_speaker:
        handlers.append(SpeakerHandler(config))
        logger.info("Handler Registered: Speaker")

    return handlers
