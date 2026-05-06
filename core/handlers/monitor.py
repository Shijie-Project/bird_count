import logging
import math
import multiprocessing as mp
import platform
import queue
import time
from typing import Optional

import cv2
import numpy as np

from ..audit import AuditLog
from ..config import Config
from ..memory_manager import SharedMemoryClient, SharedMemoryConfig
from .base import BaseHandler, BatchInferenceResult, InferenceResult


try:
    import tkinter as tk

    root = tk.Tk()
    SCREEN_W, SCREEN_H = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
except Exception:
    SCREEN_W, SCREEN_H = 1920, 1080  # fallback


logger = logging.getLogger(__name__)


class _InternalMonitorRenderer:
    """
    Actual rendering logic that runs in the dedicated DisplayProcess.
    """

    # --- Style Config (Static constants for cache efficiency) ---
    COLOR_GRID = (50, 50, 50)
    COLOR_BG_BAR = (0, 0, 0)
    COLOR_BG_ALERT = (0, 0, 255)
    COLOR_TEXT_NORMAL = (0, 255, 0)
    COLOR_TEXT_ALERT = (255, 255, 255)

    FONT = cv2.FONT_HERSHEY_DUPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 1

    # Reserve room for the OS title bar / taskbar so the bottom row of tiles
    # doesn't get clipped off the visible desktop.
    CHROME_MARGIN_W = 20
    CHROME_MARGIN_H = 80

    def __init__(self, config: Config, window_name="Bird Detection Dashboard", name="Monitor"):
        self.rgb_to_bgr = config.envs.show_density_map
        self.window_name = window_name
        self.name = name
        self.num_streams = config.num_streams
        self.is_window_setup = False

        self._last_ui_update = 0
        self._ui_update_interval = 1.0 / 25.0

        # Layout: square-ish grid, fit inside the visible desktop area.
        usable_w = max(320, SCREEN_W - self.CHROME_MARGIN_W)
        usable_h = max(240, SCREEN_H - self.CHROME_MARGIN_H)
        self.cols = int(math.ceil(math.sqrt(self.num_streams)))
        self.rows = int(math.ceil(self.num_streams / self.cols))
        self.tile_w = usable_w // self.cols
        self.tile_h = usable_h // self.rows

        # Initial window dimensions (the user can still drag-resize at runtime).
        self.window_w = self.cols * self.tile_w
        self.window_h = self.rows * self.tile_h

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
            # Zero-copy fetch from SHM (safe because ResultProcess holds the buffer
            # in READING state until we ack on the ack_queue).
            frame = shm_client.frames[sid, res.buffer_idx]
            resized = cv2.resize(frame, (self.tile_w, self.tile_h))
            if self.rgb_to_bgr:
                resized = resized[..., ::-1]

            self.canvas[slices[0], slices[1]] = resized
            self._draw_overlay(res, coords)

        # Throttled Window Refresh
        if now - self._last_ui_update >= self._ui_update_interval:
            if not self.is_window_setup:
                # WINDOW_NORMAL = user can drag-resize; WINDOW_KEEPRATIO preserves
                # aspect ratio so tiles don't stretch when the user resizes.
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(self.window_name, self.window_w, self.window_h)
                cv2.moveWindow(self.window_name, 0, 0)
                self._disable_window_close_button()
                # Pop to the front once on first show so the user notices the
                # monitor turned on; afterwards it behaves like a normal window
                # and can be hidden behind other apps.
                self._force_foreground()
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

    def _force_foreground(self):
        """Pop the dashboard to the front *once* (Windows only).

        We briefly mark the window TOPMOST to push it above everything, then
        immediately clear that flag so it behaves like an ordinary window and
        the user can cover it with other apps afterwards.
        """
        if platform.system() != "Windows":
            return
        try:
            import ctypes

            user32 = ctypes.windll.user32
            hwnd = user32.FindWindowW(None, self.window_name)  # noqa
            if not hwnd:
                return
            HWND_TOPMOST = -1
            HWND_NOTOPMOST = -2
            SWP_NOMOVE = 0x0002
            SWP_NOSIZE = 0x0001
            SWP_SHOWWINDOW = 0x0040
            flags = SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE in case it was minimized
            # Flash to the top of Z-order, then drop topmost so it doesn't stay pinned.
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, flags)
            user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, flags)
            user32.SetForegroundWindow(hwnd)
        except Exception as e:
            logger.debug(f"[{self.name}] force_foreground failed: {e}")


class DisplayProcess(mp.Process):
    def __init__(self, config, shm_config, display_queue, ack_queue):
        super().__init__(name="Monitor", daemon=True)
        self.config = config
        self.shm_config = shm_config
        self.display_queue = display_queue
        self.ack_queue = ack_queue

    def _ack(self, pairs):
        """Send buffer indices back to ResultProcess so it can release SHM."""
        if not pairs or self.ack_queue is None:
            return
        try:
            self.ack_queue.put_nowait(pairs)
        except queue.Full:
            # Drop the ack — ResultProcess will force-release after timeout.
            logger.debug(f"[{self.name}] ack_queue full; relying on stale-ack sweep.")

    def run(self):
        shm_client = SharedMemoryClient(self.shm_config)
        shm_client.connect()
        renderer = _InternalMonitorRenderer(self.config)
        logger.info(f"[{self.name}] Display Process started.")

        while True:
            try:
                batch = self.display_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[{self.name}] Queue error: {e}")
                continue

            pairs = list(zip(batch.stream_ids, batch.buffer_indices))
            try:
                renderer.render_batch(batch, shm_client)
            except Exception as e:
                logger.error(f"[{self.name}] Render error: {e}")
            finally:
                # CRITICAL: always ack so the SHM buffer can be reclaimed,
                # otherwise stale-ack sweep has to clean up.
                self._ack(pairs)


class MonitorHandler(BaseHandler):
    """
    Proxy Handler that looks like a normal handler but runs the heavy UI in a
    separate process.

    Runtime-toggleable: enable() / disable() spawn or terminate the DisplayProcess
    on demand. The initial state is taken from config.envs.enable_monitor; after
    that, the GUI controls the state.
    """

    def __init__(
        self,
        config: Config,
        shm_config: SharedMemoryConfig,
        ack_queue: Optional[mp.Queue] = None,
        name="Monitor",
    ):
        # needs_frames=False because the *handler* itself does not touch SHM in-process.
        # The DisplayProcess is what reads SHM, and we now coordinate that via ack_queue.
        super().__init__(name=name, needs_frames=False)
        self.config = config
        self.shm_config = shm_config
        self.ack_queue = ack_queue
        self.display_queue: Optional[mp.Queue] = None
        self.proc: Optional[mp.Process] = None
        self._enabled = bool(getattr(config.envs, "enable_monitor", True))
        self._started = False

        # Per-handler audit log (mirrors SmartPlugHandler / SpeakerHandler pattern).
        # Opened in start() since file handles can't be pickled across spawn.
        self._audit_log_path = config.envs.audit_log_path
        self.audit: Optional[AuditLog] = None

    def _spawn_display(self):
        """Start the DisplayProcess if it isn't already running."""
        if self.proc is not None and self.proc.is_alive():
            return
        self.display_queue = mp.Queue(maxsize=5)
        self.proc = DisplayProcess(self.config, self.shm_config, self.display_queue, self.ack_queue)
        self.proc.start()
        logger.info("[Monitor] Background display process started.")

    def _terminate_display(self):
        """Tear the DisplayProcess down. Safe to call when nothing is running."""
        proc = self.proc
        self.proc = None
        self.display_queue = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=0.5)
            logger.info("[Monitor] Background display process stopped.")
        except Exception as e:
            logger.error(f"[Monitor] Error terminating display: {e}")

    def start(self):
        """Lifecycle hook fired when ResultProcess starts up."""
        super().start()
        self.audit = AuditLog(self._audit_log_path, name=self.name)
        if self.audit:
            self.audit.log("handler.start", handler=self.name, initial_enabled=self._enabled)
        self._started = True
        if self._enabled:
            self._spawn_display()

    def enable(self) -> bool:
        """Turn the monitor ON. Spawns the DisplayProcess if needed."""
        if self._enabled:
            return True
        self._enabled = True
        if self._started:
            self._spawn_display()
        if self.audit:
            self.audit.log("monitor.enable")
        logger.info(f"[{self.name}] Monitor turned ON.")
        return True

    def disable(self) -> bool:
        """Turn the monitor OFF. Terminates the DisplayProcess if running."""
        if not self._enabled:
            return False
        self._enabled = False
        self._terminate_display()
        if self.audit:
            self.audit.log("monitor.disable")
        logger.info(f"[{self.name}] Monitor turned OFF.")
        return False

    def toggle(self) -> bool:
        """One-shot GUI callback: flip on/off. Returns the new enabled state."""
        return self.disable() if self._enabled else self.enable()

    def is_enabled(self) -> bool:
        return self._enabled

    def handle(self, result: InferenceResult, frame: Optional[np.ndarray]):
        # Overridden by handle_batch
        pass

    def handle_batch(self, batch_result: BatchInferenceResult, shm_client: SharedMemoryClient) -> set[tuple[int, int]]:
        """
        Forward metadata to DisplayProcess. If we successfully enqueue, claim the
        (sid, buffer_idx) pairs so ResultProcess defers the SHM release until
        DisplayProcess acks. On Queue.Full or when disabled we do NOT claim —
        caller releases immediately and the frame is simply dropped.
        """
        if not self._enabled or self.display_queue is None or not batch_result.results:
            return set()
        try:
            self.display_queue.put_nowait(batch_result)
        except queue.Full:
            return set()
        return set(zip(batch_result.stream_ids, batch_result.buffer_indices))

    def stop(self):
        self._terminate_display()
        if self.audit:
            self.audit.log("handler.stop", handler=self.name)
            self.audit.close()
