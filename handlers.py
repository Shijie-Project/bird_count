import logging
import math
import platform

import cv2
import numpy as np


logger = logging.getLogger("Handler")


class BaseResultHandler:
    """Base class for result handlers."""

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

        self.window_name = "Bird Count Monitor"  # 窗口标题固定
        self.is_window_setup = False  # 标记窗口是否已初始化

        self.COLOR_TEXT_ID = (0, 255, 255)
        self.COLOR_TEXT_INFO = (200, 200, 200)
        self.COLOR_GRID = (50, 50, 50)
        self.COLOR_BG_BAR = (0, 0, 0)
        self.COLOR_ACTIVE = (0, 255, 0)

        self.LINE_THICKNESS = 4

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
            self._draw_placeholder(sid, x1, y1)

        self._draw_grid_lines()

    def _draw_grid_lines(self):
        for r in range(1, self.rows):
            y = r * self.tile_h
            cv2.line(self.canvas, (0, y), (self.canvas_w, y), self.COLOR_GRID, self.LINE_THICKNESS)
        for c in range(1, self.cols):
            x = c * self.tile_w
            cv2.line(self.canvas, (x, 0), (x, self.canvas_h), self.COLOR_GRID, self.LINE_THICKNESS)

    def _draw_placeholder(self, sid, x, y):
        cv2.putText(
            self.canvas,
            "Wait Signal...",
            (x + 30, y + self.tile_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (80, 80, 80),
            1,
        )

    def _disable_window_close_button(self):
        """
        [黑科技] 仅限 Windows:
        获取窗口句柄，修改系统菜单，移除 '关闭' 选项 (SC_CLOSE)。
        效果：右上角的 X 变灰不可点，但 _ 最小化依然可用。
        """
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

                logger.info(f"[UI] Close button (X) disabled for window: {self.window_name}")
        except Exception as e:
            logger.warning(f"[UI] Failed to disable close button: {e}")

    def handle(self, sids, processed_images, timestamp):
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

            # Get current frame (RGB)
            img_rgb = processed_images[i]

            # [Fix Color] OpenCV imshow expects BGR, but our pipeline uses RGB.
            # Convert here to ensure correct color display.
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Resize to fit the tile (Resize is fast enough for small batches)
            # Interpolation can be LINEAR or NEAREST; NEAREST is fastest
            img_resized = cv2.resize(img_bgr, (self.tile_w, self.tile_h), interpolation=cv2.INTER_LINEAR)

            y_slice, x_slice, coords = self.roi_map[sid]
            x1, y1, x2, y2 = coords

            self.canvas[y_slice, x_slice] = img_resized

            bar_height = 28
            label = f"CH-{sid:02d}"

            cv2.rectangle(self.canvas, (x1, y1), (x1 + 90, y1 + bar_height), self.COLOR_BG_BAR, -1)
            cv2.putText(self.canvas, label, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, self.COLOR_TEXT_ID, 1)

            cv2.rectangle(self.canvas, (x1, y1), (x2 - 1, y2 - 1), self.COLOR_GRID, 1)

        # --- 5. Display ---
        # Note: imshow is quite time-consuming as it communicates with the Window Manager
        self._draw_grid_lines()

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

        cv2.imshow("Bird Count Monitor", self.canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Exit signal received.")
            raise KeyboardInterrupt("UI Exit")

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            if platform.system() != "Windows":  # Windows 下 X 已经被禁用了，不需要这个逻辑
                # 如果发现窗口不可见（被关了），下次循环重新创建
                self.is_window_setup = False

    def cleanup(self):
        if self.enable:
            cv2.destroyAllWindows()
