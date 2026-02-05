import logging
import math
import platform

import cv2
import numpy as np

from .base import BaseResultHandler


logger = logging.getLogger(__name__)


class MonitorHandler(BaseResultHandler):
    COLOR_TEXT_ID = (0, 255, 255)
    COLOR_TEXT_INFO = (200, 200, 200)
    COLOR_GRID = (50, 50, 50)
    COLOR_BG_BAR = (0, 0, 0)
    COLOR_ACTIVE = (0, 255, 0)

    LINE_THICKNESS = 4

    def __init__(self, cfg):
        self.enable = cfg.stream.enable_monitor
        if not self.enable:
            return

        self.window_name = "Bird Count Monitor"
        self.is_window_setup = False

        # --- 1. Layout Calculation (One-off) ---
        self.num_streams = cfg.stream.num_streams

        # Calculate optimal grid (e.g., 22 streams -> 5x5)
        self.cols = int(math.ceil(math.sqrt(self.num_streams)))
        self.rows = int(math.ceil(self.num_streams / self.cols))

        # --- 2. Screen Adaptation (Key Optimization) ---
        # Assume target display resolution is 1920x1080 (adjust based on your actual screen)
        target_w, target_h = 1920, 1080

        # Calculate the size of each tile
        self.tile_w = target_w // self.cols
        self.tile_h = target_h // self.rows

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
            self.roi_map[sid] = (slice(y1, y2), slice(x1, x2), (x1, y1, x2, y2))

            # Pre-write ID on canvas so we can identify slots even when empty
            self._draw_placeholder(x1, y1)

        self._draw_grid_lines()

        logger.info("[Monitor] Dashboard initialized.")

    def _draw_placeholder(self, x, y):
        cv2.putText(
            self.canvas,
            "Wait Signal...",
            (x + 30, y + self.tile_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (80, 80, 80),
            1,
        )

    def _draw_grid_lines(self):
        for r in range(1, self.rows):
            y = r * self.tile_h
            cv2.line(self.canvas, (0, y), (self.canvas_w, y), self.COLOR_GRID, self.LINE_THICKNESS)
        for c in range(1, self.cols):
            x = c * self.tile_w
            cv2.line(self.canvas, (x, 0), (x, self.canvas_h), self.COLOR_GRID, self.LINE_THICKNESS)

    def _disable_window_close_button(self):
        if platform.system() != "Windows":
            return

        try:
            import ctypes

            # 1. 查找窗口句柄 (HWND)
            # 注意：必须在 cv2.imshow 之后调用，否则窗口还没创建找不到
            hwnd = ctypes.windll.user32.FindWindowW(None, self.window_name)

            if hwnd:
                # 2. 获取系统菜单句柄
                hmenu = ctypes.windll.user32.GetSystemMenu(hwnd, False)

                # 3. 移除“关闭”菜单项 (SC_CLOSE = 0xF060)
                # 移除后，右上角的 X 按钮会自动变灰禁用
                ctypes.windll.user32.RemoveMenu(hmenu, 0xF060, 0x00000000)

                logger.info(f"[Monitor] Close button (X) disabled for window: {self.window_name}")
        except Exception as e:
            logger.warning(f"[Monitor] Failed to disable close button: {e}")

    def _display(self):
        if not self.is_window_setup:
            # 第一次显示窗口
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(self.window_name, self.canvas)
            cv2.waitKey(1)  # 必须先刷新一次，让窗口句柄生成

            # 调用 Windows API 禁用 X 按钮
            self._disable_window_close_button()

            self.is_window_setup = True
        else:
            # 常规刷新
            cv2.imshow(self.window_name, self.canvas)
            cv2.waitKey(1)

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            if platform.system() != "Windows":
                self.is_window_setup = False

    def handle(self, sids, processed_images, counts, timestamp):
        """
        sids: list[int] -> IDs for this batch
        processed_images: np.ndarray (B, H, W, 3) -> RGB Images ready for display
        """
        if not self.enable:
            return

        # --- 4. Partial Update ---
        # We only update the incoming batch of sids.
        # Other sids on the canvas remain as the previous frame (Visual Persistence).

        for i, sid in enumerate(sids):
            if sid >= self.num_streams:
                continue  # Prevent out of bounds

            y_slice, x_slice, coords = self.roi_map[sid]
            x1, y1, x2, y2 = coords

            # Get current frame (RGB)
            img_rgb = processed_images[i]

            # [Fix Color] OpenCV imshow expects BGR, but our pipeline uses RGB.
            # Convert here to ensure correct color display.
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Resize to fit the tile (Resize is fast enough for small batches)
            # Interpolation can be LINEAR or NEAREST; NEAREST is fastest
            img_resized = cv2.resize(img_bgr, (self.tile_w, self.tile_h), interpolation=cv2.INTER_LINEAR)
            self.canvas[y_slice, x_slice] = img_resized

            label = f"CH-{sid:02d}: {int(counts[i])}"

            cv2.rectangle(self.canvas, (x1, y1), (x1 + 120, y1 + 28), self.COLOR_BG_BAR, -1)
            cv2.putText(self.canvas, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.COLOR_TEXT_ID, 1)

            cv2.rectangle(self.canvas, (x1, y1), (x2 - 1, y2 - 1), self.COLOR_GRID, 1)

        # --- 5. Display ---
        # Note: imshow is quite time-consuming as it communicates with the Window Manager
        self._draw_grid_lines()
        self._display()

    def cleanup(self):
        if self.enable:
            cv2.destroyAllWindows()
