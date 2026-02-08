import logging
import math
import platform
import time
from typing import Optional

import cv2
import numpy as np

from ..config import Config
from .base import BaseHandler, BatchInferenceResult, InferenceResult, SharedMemoryClient


logger = logging.getLogger(__name__)


class MonitorHandler(BaseHandler):
    """
    Real-time Visualization Dashboard.
    Updates individual tiles on a unified canvas as results arrive.
    """

    # --- Style Config ---
    COLOR_GRID = (50, 50, 50)
    COLOR_BG_BAR = (0, 0, 0)
    COLOR_TEXT_NORMAL = (0, 255, 0)  # Green
    COLOR_TEXT_ALERT = (255, 255, 255)  # White
    COLOR_BG_ALERT = (0, 0, 255)  # Red

    FONT = cv2.FONT_HERSHEY_DUPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 1

    # Target display resolution (Full HD)
    SCREEN_W, SCREEN_H = 1920, 1080

    def __init__(self, config: Config):
        self.window_name = "Bird Detection Dashboard"
        self.num_streams = config.num_streams
        self.is_window_setup = False

        # 1. Threshold Configuration (Zone-based)
        self.stream_thresholds = {}

        # Parse Zones to build the map
        for zone in getattr(config, "zones", []):
            # Get threshold for this zone (fallback to global if not set in zone)
            for cam_ip in getattr(zone, "cameras", []):
                self.stream_thresholds[config.ip2id[cam_ip]] = zone.threshold

        # 2. Prepare UI Layout
        self._setup_layout()
        self._init_canvas()

    def start(self):
        logger.info("Monitor Initialized.")

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
            logger.warning(f"[Monitor] Failed to disable close button: {e}")

    def _setup_layout(self):
        """Calculate grid positions, tile sizes and ROI map."""
        # Calculate optimal grid (rows x cols)
        if self.num_streams == 0:
            self.cols, self.rows = 1, 1
        else:
            self.cols = int(math.ceil(math.sqrt(self.num_streams)))
            self.rows = int(math.ceil(self.num_streams / self.cols))

        self.tile_w = self.SCREEN_W // self.cols
        self.tile_h = self.SCREEN_H // self.rows

        # Pre-calculate Slice objects for fast numpy indexing
        # ROI structure: List of (slice_y, slice_x, (x1, y1, x2, y2))
        self.roi_map = []
        for i in range(self.num_streams):
            r, c = divmod(i, self.cols)
            x1, y1 = c * self.tile_w, r * self.tile_h
            x2, y2 = x1 + self.tile_w, y1 + self.tile_h
            self.roi_map.append((slice(y1, y2), slice(x1, x2), (x1, y1, x2, y2)))

    def _init_canvas(self):
        """Create the black background canvas."""
        # Pre-allocate memory for the dashboard
        self.canvas = np.zeros((self.rows * self.tile_h, self.cols * self.tile_w, 3), dtype=np.uint8)

    def handle_batch(self, batch_result: Optional[BatchInferenceResult], shm_client: SharedMemoryClient):
        """
        [Optimized] Process updates in bulk and refresh window ONCE.
        """
        updated = False

        # 1. Update Canvas (Memory Operations)
        for result in batch_result.results:
            # Direct read from SHM
            raw_frame = shm_client.frames[result.stream_id, result.buffer_idx]

            # Update the tile
            success = self.handle(result, raw_frame)
            if success:
                updated = True

        # 2. Refresh Window (GUI Operation - Expensive)
        # Only do this once per batch!
        if updated:
            if not self.is_window_setup:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self._disable_window_close_button()
                self.is_window_setup = True

            cv2.imshow(self.window_name, self.canvas)
            cv2.waitKey(1)

    def handle(self, result: InferenceResult, frame: np.ndarray) -> bool:
        """
        Updates the specific tile for the incoming stream ID.
        Returns True if update was successful.
        """
        sid = result.stream_id
        if sid >= len(self.roi_map):
            return False

        # 1. Get ROI coordinates
        y_slice, x_slice, coords = self.roi_map[sid]

        # 2. Process Image (Resize & Copy)
        try:
            # Resize directly into the canvas slot
            # Note: frame is usually RGB (since we converted it in InferenceProcess or it came as RGB).
            # But OpenCV imshow expects BGR.
            # If your Inference Engine writes RGB overlay, convert here.
            # If your Inference Engine writes BGR, remove conversion.
            # Assuming standard pipeline: Inference writes BGR or RGB based on previous steps.
            # Let's assume input 'frame' is BGR for OpenCV compatibility (standard).

            resized_frame = cv2.resize(frame, (self.tile_w, self.tile_h), interpolation=cv2.INTER_LINEAR)

            # If the source is RGB (e.g. from PIL/Torch), swap to BGR for imshow
            self.canvas[y_slice, x_slice] = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)

        except Exception as e:
            logger.error(f"[Monitor] Resize failed for stream {sid}: {e}")
            return False

        # 3. Draw Overlay (In-place on canvas)
        self._draw_overlay(sid, result.count, coords)

        return True

    def _draw_overlay(self, sid: int, count: float, coords: tuple):
        """Draws the header bar, text, and alerts based on Zone Thresholds."""
        x1, y1, x2, y2 = coords

        # 1. Determine Threshold for this specific camera
        threshold = self.stream_thresholds[sid]

        # 2. Check Alert Status
        is_alert = count >= threshold
        flash_on = int(time.time() * 4) % 2 == 0

        if is_alert and flash_on:
            bg_color = self.COLOR_BG_ALERT
            text_color = self.COLOR_TEXT_ALERT
            border_color = self.COLOR_BG_ALERT
            border_thick = 3
        else:
            bg_color = self.COLOR_BG_BAR
            text_color = self.COLOR_TEXT_NORMAL
            border_color = self.COLOR_GRID
            border_thick = 1

        # 3. Format Label (Show current / threshold)
        # Example: "CAM 0: 15 / 10"
        label = f"CAM {sid}: {int(count)} / {int(threshold)}"

        # 4. Draw UI Elements
        # Header Background
        # Adjust bar width based on text length estimation
        bar_width = 220
        cv2.rectangle(self.canvas, (x1, y1), (x1 + bar_width, y1 + 30), bg_color, -1)

        # Text
        cv2.putText(
            self.canvas, label, (x1 + 10, y1 + 22), self.FONT, self.FONT_SCALE, text_color, self.FONT_THICKNESS
        )

        # Border
        cv2.rectangle(self.canvas, (x1, y1), (x2, y2), border_color, border_thick)

    def stop(self):
        cv2.destroyAllWindows()
