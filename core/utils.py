import logging
import sys

import torch


logger = logging.getLogger(__name__)

# Global cache for memory format
_OPTIMAL_MEMORY_FORMAT = None


def setup_logging(debug: bool = False) -> None:
    """
    Configure specific logging format for industrial monitoring.
    Includes process name/ID to debug multi-process issues.
    """
    level = logging.DEBUG if debug else logging.INFO

    # Format optimized for parsing (e.g., by ELK stack or grep)
    log_fmt = "%(asctime)s.%(msecs)03d | %(levelname)-8s | PID:%(process)-5d | %(processName)-15s |%(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=log_fmt,
        datefmt=date_fmt,
        stream=sys.stdout,
    )

    # Suppress noisy libraries that spam logs
    noisy_loggers = ["urllib3", "PIL", "matplotlib"]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_optimal_memory_format(device: torch.device) -> torch.memory_format:
    """
    Auto-detects the optimal memory layout based on GPU architecture.

    Strategy:
    - Compute Capability >= 7.5 (Turing/Ampere/Hopper): Use 'channels_last' (NHWC) for Tensor Core optimization.
    - Compute Capability < 7.5 (Pascal/Volta): Use 'contiguous_format' (NCHW) for CUDA Core optimization.
    - CPU: Use 'contiguous_format'.
    """
    global _OPTIMAL_MEMORY_FORMAT

    if _OPTIMAL_MEMORY_FORMAT is not None:
        return _OPTIMAL_MEMORY_FORMAT

    target_format = torch.contiguous_format

    if device.type == "cuda":
        try:
            # Get Compute Capability (Major, Minor)
            major, minor = torch.cuda.get_device_capability(device)

            # Turing (7.5) and newer architectures favor channels_last
            if (major >= 7 and minor >= 5) or major >= 8:
                target_format = torch.channels_last
                logger.info(f"Auto-Tuning: Detected GPU Arch {major}.{minor}. Using 'channels_last' (NHWC).")
            else:
                logger.info(f"Auto-Tuning: Detected GPU Arch {major}.{minor}. Using 'contiguous_format' (NCHW).")

        except Exception as e:
            logger.warning(f"Failed to detect GPU capability: {e}. Fallback to NCHW.")

    _OPTIMAL_MEMORY_FORMAT = target_format
    return target_format
