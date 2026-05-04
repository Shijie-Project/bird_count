import logging
import time
import tkinter as tk
from tkinter import messagebox
from typing import Callable, Optional

from .config import Config


logger = logging.getLogger(__name__)


# Style palette
_BG_PAGE = "#f0f0f0"
_BG_HEADER = "#2c3e50"
_BG_CANCEL = "#c0392b"
_BG_CANCEL_HOVER = "#e74c3c"
_DOT_IDLE = "#bdc3c7"
_DOT_HIJACK = "#f39c12"
_DOT_ACTIVE = "#e74c3c"


class ManualTriggerGUI:
    """
    Manual Trigger Panel with hijack toggles, live device-status indicators,
    and a confirmed 'Cancel All' button.
    """

    def __init__(
        self,
        config: Config,
        on_trigger_callback: Callable[[int], None],
        on_cancel_all_callback: Callable[[], None],
        status_provider: Optional[Callable[[], dict]] = None,
        name: str = "TriggerGUI",
    ):
        self.name = name
        self.config = config
        self.on_trigger_callback = on_trigger_callback
        self.on_cancel_all_callback = on_cancel_all_callback
        self.status_provider = status_provider

        self.root: Optional[tk.Tk] = None
        self.status_label: Optional[tk.Label] = None

        self.hijack_states: dict[int, bool] = dict.fromkeys(config.sid_to_ip.keys(), False)
        self.buttons: dict[int, tk.Button] = {}
        self.dots: dict[int, tk.Label] = {}

        # Resolve which speakers/plugs belong to each stream so we can light up dots
        # based on per-zone device activity rather than per-stream.
        self.stream_to_devices: dict[int, set[str]] = {}
        for sid, zone in config.sid_to_zone.items():
            ips: set[str] = set()
            ips.update(getattr(zone, "speakers", []) or [])
            ips.update(getattr(zone, "smart_plugs", []) or [])
            self.stream_to_devices[sid] = ips

        self._last_status_refresh = 0.0
        self._status_refresh_interval = 0.25  # 4 Hz, plenty for human eyes

    def setup(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as e:
            logger.error(f"[{self.name}] Tkinter unavailable: {e}")
            self.root = None
            return

        self.root.title("System Control - Manual Hijack")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, True)
        self.root.configure(bg=_BG_PAGE)

        # --- 1. Header ---
        header_frame = tk.Frame(self.root, bg=_BG_HEADER, pady=10)
        header_frame.pack(fill="x")
        tk.Label(
            header_frame,
            text="PRO-INFERENCE HIJACK CONTROL",
            fg="white",
            bg=_BG_HEADER,
            font=("Segoe UI", 12, "bold"),
        ).pack()

        # --- 2. Cancel All button ---
        cancel_frame = tk.Frame(self.root, bg=_BG_PAGE, padx=10, pady=8)
        cancel_frame.pack(fill="x")
        cancel_btn = tk.Button(
            cancel_frame,
            text="CANCEL ALL ALERTS",
            bg=_BG_CANCEL,
            fg="white",
            activebackground=_BG_CANCEL_HOVER,
            activeforeground="white",
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            height=2,
            command=self._handle_cancel_all_click,
        )
        cancel_btn.pack(fill="x")
        cancel_btn.bind("<Enter>", lambda e: cancel_btn.config(bg=_BG_CANCEL_HOVER))
        cancel_btn.bind("<Leave>", lambda e: cancel_btn.config(bg=_BG_CANCEL))

        # --- 3. Scrollable button grid ---
        grid_container = tk.Frame(self.root, bg=_BG_PAGE)
        grid_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        canvas = tk.Canvas(grid_container, bg=_BG_PAGE, highlightthickness=0, height=420)
        scrollbar = tk.Scrollbar(grid_container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        main_frame = tk.Frame(canvas, bg=_BG_PAGE)
        canvas.create_window((0, 0), window=main_frame, anchor="nw")
        main_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        max_cols = 4
        for i, (stream_id, ip) in enumerate(self.config.sid_to_ip.items()):
            row, col = divmod(i, max_cols)
            cell = tk.Frame(main_frame, bg=_BG_PAGE, padx=5, pady=5)
            cell.grid(row=row, column=col)

            # Status dot
            dot = tk.Label(cell, text="●", fg=_DOT_IDLE, bg=_BG_PAGE, font=("Segoe UI", 10))
            dot.pack()
            self.dots[stream_id] = dot

            btn = tk.Button(
                cell,
                text=f"STREAM {stream_id}\n{ip}",
                width=16,
                height=3,
                bg="#ffffff",
                fg="#2c3e50",
                relief="flat",
                highlightthickness=1,
                highlightbackground="#cccccc",
                font=("Segoe UI", 9),
                command=lambda s=stream_id: self._handle_click(s),
            )
            btn.pack()
            self.buttons[stream_id] = btn

            btn.bind("<Enter>", lambda e, b=btn: b.config(highlightbackground="#3498db"))
            btn.bind("<Leave>", lambda e, b=btn, s=stream_id: self._refresh_button_style(s))

        # --- 4. Status Bar ---
        self.status_label = tk.Label(
            self.root,
            text="Ready...",
            bd=1,
            relief="sunken",
            anchor="w",
            font=("Segoe UI", 8),
            padx=10,
            pady=5,
        )
        self.status_label.pack(fill="x")

    def _refresh_button_style(self, stream_id: int):
        """Updates button color based on hijack state."""
        btn = self.buttons.get(stream_id)
        if not btn:
            return

        if self.hijack_states.get(stream_id, False):
            btn.config(bg="#f39c12", fg="white", highlightbackground="#e67e22")
        else:
            btn.config(bg="#ffffff", fg="#2c3e50", highlightbackground="#cccccc")

    def _handle_click(self, stream_id: int):
        """Toggles hijack state and triggers callback."""
        self.hijack_states[stream_id] = not self.hijack_states[stream_id]
        self._refresh_button_style(stream_id)
        self.on_trigger_callback(stream_id)

        if self.status_label:
            state_text = "ENABLED" if self.hijack_states[stream_id] else "DISABLED"
            self.status_label.config(
                text=f"Stream {stream_id} Hijack {state_text} at {time.strftime('%H:%M:%S')}",
                fg="#e67e22" if self.hijack_states[stream_id] else "#2980b9",
            )

    def _handle_cancel_all_click(self):
        """Confirm, then dispatch cancel-all and reset every visual hijack indicator."""
        if not self.root:
            return
        confirmed = messagebox.askyesno(
            title="Confirm Cancel All",
            message="Cancel all active alerts and clear all manual hijacks?\n\n"
            "This will turn off every active speaker/plug and reset hold-down timers.",
            parent=self.root,
        )
        if not confirmed:
            return

        try:
            self.on_cancel_all_callback()
        except Exception as e:
            logger.error(f"[{self.name}] cancel_all callback failed: {e}")

        # Reset every per-stream hijack indicator visually.
        for sid in list(self.hijack_states.keys()):
            self.hijack_states[sid] = False
            self._refresh_button_style(sid)

        if self.status_label:
            self.status_label.config(
                text=f"All alerts cancelled at {time.strftime('%H:%M:%S')}",
                fg="#c0392b",
            )

    def _refresh_status_dots(self):
        """Pull active-device snapshot and recolor dots for streams in alerting zones."""
        if not self.status_provider:
            return
        try:
            snapshot = self.status_provider() or {}
        except Exception as e:
            logger.debug(f"[{self.name}] status_provider error: {e}")
            return

        active_ips: set[str] = set()
        for ips in snapshot.values():
            if ips:
                active_ips.update(ips)

        for sid, dot in self.dots.items():
            if self.hijack_states.get(sid, False):
                color = _DOT_HIJACK
            elif self.stream_to_devices.get(sid, set()) & active_ips:
                color = _DOT_ACTIVE
            else:
                color = _DOT_IDLE
            dot.config(fg=color)

    def update(self):
        if not self.root:
            return
        try:
            now = time.time()
            if now - self._last_status_refresh >= self._status_refresh_interval:
                self._refresh_status_dots()
                self._last_status_refresh = now
            self.root.update()
        except tk.TclError:
            self.root = None

    def destroy(self):
        """Explicitly destroy the GUI and release Tcl interpreter resources."""
        if self.root:
            try:
                logger.info(f"[{self.name}] Destroying manual trigger GUI...")
                self.root.destroy()
                self.root = None
            except Exception as e:
                logger.error(f"[{self.name}] Error during destroy: {e}")
