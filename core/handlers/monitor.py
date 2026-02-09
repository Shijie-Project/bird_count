import logging
import math
import multiprocessing as mp
import platform
import queue
import time
from typing import Optional

import cv2
import numpy as np

from ..config import Config
from ..memory_manager import SharedMemoryClient, SharedMemoryConfig
from .base import BaseHandler, BatchInferenceResult, InferenceResult


logger = logging.getLogger(__name__)


class _InternalMonitorRenderer:
    """
    Actual rendering logic that runs in the dedicated DisplayProcess.
    """

    # --- Style Config (Static constants for cache efficiency) ---
    COLOR_GRID = (50, 50, 50)
    COLOR_BG_BAR = (0, 0, 0)
    COLOR_TEXT_NORMAL = (0, 255, 0)
    COLOR_TEXT_ALERT = (255, 255, 255)
    COLOR_BG_ALERT = (0, 0, 255)

    FONT = cv2.FONT_HERSHEY_DUPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 1

    # Target display resolution
    SCREEN_W, SCREEN_H = 1920, 1080

    def __init__(self, config: Config, window_name="Bird Detection Dashboard", name="Monitor"):
        self.rgb_to_bgr = config.envs.show_density_map
        self.window_name = window_name
        self.name = name
        self.num_streams = config.num_streams
        self.is_window_setup = False

        self._last_ui_update = 0
        self._ui_update_interval = 1.0 / 25.0

        # Layout calculation
        self.cols = int(math.ceil(math.sqrt(self.num_streams)))
        self.rows = int(math.ceil(self.num_streams / self.cols))
        self.tile_w = self.SCREEN_W // self.cols
        self.tile_h = self.SCREEN_H // self.rows

        self.canvas = np.zeros((self.rows * self.tile_h, self.cols * self.tile_w, 3), dtype=np.uint8)
        self.roi_map = [
            (
                (slice(r * self.tile_h, (r + 1) * self.tile_h), slice(c * self.tile_w, (c + 1) * self.tile_w)),
                (c * self.tile_w, r * self.tile_h, (c + 1) * self.tile_w, (r + 1) * self.tile_h),
            )
            for i in range(self.num_streams)
            for r, c in [divmod(i, self.cols)]
        ]

    def render_batch(self, batch: BatchInferenceResult, shm_client: SharedMemoryClient):
        now = time.time()
        # Update canvas
        for res in batch.results:
            sid = res.stream_id
            if sid >= len(self.roi_map):
                continue

            slices, coords = self.roi_map[sid]
            # Zero-copy fetch from SHM
            frame = shm_client.frames[sid, res.buffer_idx]
            resized = cv2.resize(frame, (self.tile_w, self.tile_h))
            if self.rgb_to_bgr:
                resized = resized[..., ::-1]

            # Fast RGB to BGR flip
            self.canvas[slices[0], slices[1]] = resized
            self._draw_overlay(res, coords)

        # Throttled Window Refresh
        if now - self._last_ui_update >= self._ui_update_interval:
            if not self.is_window_setup:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self._disable_window_close_button()
                self.is_window_setup = True

            cv2.imshow(self.window_name, self.canvas)
            cv2.waitKey(1)
            self._last_ui_update = now

    def _draw_overlay(self, res: InferenceResult, coords: tuple):
        x1, y1, x2, y2 = coords
        flash_on = int(time.time() * 4) & 1 == 0

        if res.alert_flag and flash_on:
            bg_color, text_color, border_color, thick = (
                self.COLOR_BG_ALERT,
                self.COLOR_TEXT_ALERT,
                self.COLOR_BG_ALERT,
                3,
            )
        else:
            bg_color, text_color, border_color, thick = self.COLOR_BG_BAR, self.COLOR_TEXT_NORMAL, self.COLOR_GRID, 1

        # Header bar
        cv2.rectangle(self.canvas, (x1, y1), (x1 + 120, y1 + 30), bg_color, -1)

        # Label Text
        label = f"CAM {res.stream_id}: {int(res.count)}"
        cv2.putText(
            self.canvas, label, (x1 + 10, y1 + 15), self.FONT, self.FONT_SCALE, text_color, self.FONT_THICKNESS
        )

        # Tile Border
        cv2.rectangle(self.canvas, (x1, y1), (x2, y2), border_color, thick)

    def _disable_window_close_button(self):
        if platform.system() != "Windows":
            return

        try:
            import ctypes

            hwnd = ctypes.windll.user32.FindWindowW(None, self.window_name)  # noqa

            if hwnd:
                hmenu = ctypes.windll.user32.GetSystemMenu(hwnd, False)  # noqa
                ctypes.windll.user32.RemoveMenu(hmenu, 0xF060, 0x00000000)  # noqa
                logger.info(f"[Monitor] Close button (X) disabled for window: {self.window_name}")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to disable close button: {e}")


class DisplayProcess(mp.Process):
    def __init__(self, config, shm_config, display_queue):
        super().__init__(name="Monitor", daemon=True)
        self.config = config
        self.shm_config = shm_config
        self.display_queue = display_queue

    def run(self):
        # Only import and connect within the child process
        shm_client = SharedMemoryClient(self.shm_config)
        shm_client.connect()
        renderer = _InternalMonitorRenderer(self.config)
        logger.info(f"[{self.name}] Display Process started.")

        while True:
            try:
                batch = self.display_queue.get(timeout=1.0)
                renderer.render_batch(batch, shm_client)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[{self.name}] Render error: {e}")


class MonitorHandler(BaseHandler):
    """
    Proxy Handler that looks like a normal handler but runs
    the heavy UI in a separate process.
    """

    def __init__(self, config: Config, shm_config: SharedMemoryConfig, name="Monitor"):
        super().__init__(name=name, needs_frames=False)
        self.config = config
        self.shm_config = shm_config
        self.display_queue: Optional[mp.Queue] = None
        self.proc: Optional[mp.Process] = None

    def start(self):
        self.display_queue = mp.Queue(maxsize=5)

        self.proc = DisplayProcess(self.config, self.shm_config, self.display_queue)
        self.proc.start()

        logger.info("[Monitor] Background display process started.")

    def handle(self, result: InferenceResult, frame: Optional[np.ndarray]):
        # This is skipped because we override handle_batch
        pass

    def handle_batch(self, batch_result: BatchInferenceResult, shm_client: SharedMemoryClient):
        """Simply forwards the metadata. High speed, no blocking."""
        if self.display_queue:
            try:
                self.display_queue.put_nowait(batch_result)
            except queue.Full:
                pass

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc.join()
