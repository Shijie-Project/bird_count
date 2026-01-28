import logging
import multiprocessing as mp
import warnings
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from inferencer.config import Config
from inferencer.scheduler import Scheduler


# --- Configuration & Env ---
class Envs(BaseSettings):
    debug: bool = False
    fps: int = 10
    num_streams: int = 22
    cuda_visible_devices: str = "0"
    num_workers_per_gpu: int = 1
    source: Literal["camera", "video", "rtsp"] = "camera"
    num_buffer: int = 4  # Increased buffer size to ensure safety with the locking mechanism

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


envs = Envs()

# Setup Logger
logging.basicConfig(
    level=logging.DEBUG if envs.debug else logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("InferenceEngine")

warnings.filterwarnings("ignore")


def main():
    cfg = Config.from_settings(envs)

    mp.set_start_method("spawn", force=True)

    scheduler = Scheduler(cfg)
    scheduler.run()


if __name__ == "__main__":
    main()
