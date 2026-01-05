import logging
import math

import cv2
import numpy as np


logger = logging.getLogger("Handler")


class BaseResultHandler:
    def setup(self):
        pass

    def handle(self, sids, processed_images, timestamp):
        pass

    def cleanup(self):
        pass


class VisualizationHandler(BaseResultHandler):
    def __init__(self, cfg):
        self.enable = cfg.ENABLE_MONITOR
        if not self.enable:
            return

        logger.info("[Monitor] Dashboard initialized. Press 'q' to exit.")

        # --- 1. Layout Calculation (One-off) ---
        self.num_streams = cfg.NUM_STREAMS

        # Calculate optimal grid (e.g., 22 streams -> 5x5)
        self.cols = int(math.ceil(math.sqrt(self.num_streams)))
        self.rows = int(math.ceil(self.num_streams / self.cols))

        # --- 2. Screen Adaptation (Key Optimization) ---
        # Assume target display resolution is 1920x1080 (adjust based on your actual screen)
        target_w, target_h = 1920, 1080

        # Calculate the size of each tile
        self.tile_w = target_w // self.cols
        self.tile_h = target_h // self.rows

        # Note: We fill the tile directly here.
        # For strict aspect ratio maintenance, additional padding logic would be needed.

        # Actual canvas size
        self.canvas_w = self.tile_w * self.cols
        self.canvas_h = self.tile_h * self.rows

        # --- 3. Pre-allocate Memory ---
        # Initialize with a black background
        self.canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        # Pre-calculate coordinate slices for each stream ID to avoid arithmetic in the loop
        self.roi_map = {}
        for sid in range(self.num_streams):
            r = sid // self.cols
            c = sid % self.cols
            x1 = c * self.tile_w
            y1 = r * self.tile_h
            x2 = x1 + self.tile_w
            y2 = y1 + self.tile_h
            self.roi_map[sid] = (slice(y1, y2), slice(x1, x2))

            # Pre-write ID on canvas so we can identify slots even when empty
            cv2.putText(
                self.canvas, f"ID:{sid}", (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

    def handle(self, sids, processed_images, timestamp):
        """
        sids: list[int] -> IDs for this batch
        processed_images: np.ndarray (B, H, W, 3) -> Heatmaps already drawn/rendered by GPU Worker
        """
        if not self.enable:
            return

        # --- 4. Partial Update ---
        # We only update the incoming batch of sids.
        # Other sids on the canvas remain as the previous frame (Visual Persistence).
        # This saves significant CPU compared to redrawing the entire mosaic every time.

        for i, sid in enumerate(sids):
            if sid >= self.num_streams:
                continue  # Prevent out of bounds

            # Get current frame (Native Resolution, e.g., 512x512)
            img = processed_images[i]

            # Resize to fit the tile (Resize is fast enough for small batches)
            # Interpolation can be LINEAR or NEAREST; NEAREST is fastest
            img_resized = cv2.resize(img, (self.tile_w, self.tile_h), interpolation=cv2.INTER_LINEAR)

            # Fill ID info (optional, for debugging or clarity)
            cv2.putText(img_resized, f"S:{sid}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Direct memory copy to the corresponding position on canvas
            y_slice, x_slice = self.roi_map[sid]
            self.canvas[y_slice, x_slice] = img_resized

        # --- 5. Display ---
        # Note: imshow is quite time-consuming as it communicates with the Window Manager
        cv2.imshow("Bird Count Monitor", self.canvas)

        # 1ms delay to refresh UI events, press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Exit signal received.")
            raise KeyboardInterrupt("UI Exit")

    def cleanup(self):
        if self.enable:
            cv2.destroyAllWindows()
