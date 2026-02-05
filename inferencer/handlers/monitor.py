import logging
import math

import cv2
import numpy as np

from .base import BaseResultHandler, ResultItem


logger = logging.getLogger(__name__)


class MonitorHandler(BaseResultHandler):
    # --- Style Config ---
    COLOR_GRID = (50, 50, 50)
    COLOR_BG_BAR = (0, 0, 0)
    COLOR_ACTIVE = (0, 255, 0)
    FONT = cv2.FONT_HERSHEY_DUPLEX

    def __init__(self, cfg):
        self.enable = cfg.stream.enable_monitor
        if not self.enable:
            return

        self.window_name = "Bird Detection Monitor"
        self.num_streams = cfg.stream.num_streams
        self.is_window_setup = False

        # 1. Prepare UI Layout
        self._setup_layout()
        self._init_canvas()

    def _setup_layout(self):
        """Calculate grid positions and tile sizes."""
        self.cols = int(math.ceil(math.sqrt(self.num_streams)))
        self.rows = int(math.ceil(self.num_streams / self.cols))

        # Display resolution target (1080p area)
        screen_w, screen_h = 1920, 1080
        self.tile_w = screen_w // self.cols
        self.tile_h = screen_h // self.rows

        # Pre-calculate ROI map for performance
        self.roi_map = []
        for i in range(self.num_streams):
            r, c = divmod(i, self.cols)
            x1, y1 = c * self.tile_w, r * self.tile_h
            x2, y2 = x1 + self.tile_w, y1 + self.tile_h
            self.roi_map.append((slice(y1, y2), slice(x1, x2), (x1, y1, x2, y2)))

    def _init_canvas(self):
        """Create the black background canvas."""
        self.canvas = np.zeros((self.rows * self.tile_h, self.cols * self.tile_w, 3), dtype=np.uint8)

    def _update_tiles(self, sids, images):
        """Resize and place processed frames onto the canvas."""
        for i, sid in enumerate(sids):
            if sid >= self.num_streams:
                continue

            y_slice, x_slice, _ = self.roi_map[sid]
            # Convert RGB (from pipeline) to BGR (for OpenCV)
            bgr_frame = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
            self.canvas[y_slice, x_slice] = cv2.resize(bgr_frame, (self.tile_w, self.tile_h))

    def _draw_overlays(self, sids, counts):
        """Draw text labels and grid lines."""
        for i, sid in enumerate(sids):
            if sid >= self.num_streams:
                continue

            _, _, (x1, y1, x2, y2) = self.roi_map[sid]
            label = f"CH-{sid:02d}: {int(counts[i])}"

            # Draw header bar and text
            cv2.rectangle(self.canvas, (x1, y1), (x1 + 120, y1 + 30), self.COLOR_BG_BAR, -1)
            cv2.putText(self.canvas, label, (x1 + 5, y1 + 22), self.FONT, 0.6, self.COLOR_ACTIVE, 1)
            # Draw border
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), self.COLOR_GRID, 1)

    def handle(self, results: ResultItem):
        """Main entry for Monitor rendering."""
        if not self.enable or not results.images:
            return

        self._update_tiles(results.sids, results.images)
        self._draw_overlays(results.sids, results.counts)

        if not self.is_window_setup:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self.is_window_setup = True

        cv2.imshow(self.window_name, self.canvas)
        cv2.waitKey(1)

    def cleanup(self):
        if self.enable:
            cv2.destroyAllWindows()
