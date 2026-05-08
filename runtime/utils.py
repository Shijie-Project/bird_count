import logging
import sys

import torch


logger = logging.getLogger(__name__)

# Global cache for memory format to avoid redundant detection
_OPTIMAL_MEMORY_FORMAT = None


def setup_logging(debug: bool = False) -> None:
    """
    Configure specific logging format for industrial monitoring.
    Includes process name/ID to debug multi-process issues.
    """
    level = logging.DEBUG if debug else logging.INFO
    log_fmt = "%(asctime)s.%(msecs)03d | %(levelname)-8s | PID:%(process)-5d | %(name)-25s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=log_fmt,
        datefmt=date_fmt,
        stream=sys.stdout,
    )

    # Suppress noisy libraries
    for logger_name in ["urllib3", "PIL", "matplotlib", "socket"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def setup_cuda() -> None:
    """
    Configures global CUDA and cuDNN settings for maximum throughput.
    Includes TF32 optimization for Ampere+ architectures.
    """
    if not torch.cuda.is_available():
        return

    # Enable cuDNN auto-tuner to find the best algorithms for the current hardware
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def get_optimal_memory_format(device: torch.device) -> torch.memory_format:
    """
    Auto-detects the optimal memory layout based on GPU architecture.
    Caches the result to prevent repeated capability checks.
    """
    global _OPTIMAL_MEMORY_FORMAT
    if _OPTIMAL_MEMORY_FORMAT is not None:
        return _OPTIMAL_MEMORY_FORMAT

    target_format = torch.contiguous_format  # Default NCHW

    if device.type == "cuda":
        try:
            major, minor = torch.cuda.get_device_capability(device)
            # Turing (7.5) and newer architectures favor 'channels_last' (NHWC)
            if (major >= 7 and minor >= 5) or major >= 8:
                target_format = torch.channels_last
                logger.info(f"Auto-Tuning: Arch {major}.{minor} detected. Using 'channels_last' (NHWC).")
            else:
                logger.info(f"Auto-Tuning: Arch {major}.{minor} detected. Using 'contiguous_format' (NCHW).")
        except Exception as e:
            logger.warning(f"Failed to detect GPU capability: {e}. Fallback to NCHW.")

    _OPTIMAL_MEMORY_FORMAT = target_format
    return target_format
