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
_BG_MONITOR_ON = "#27ae60"
_BG_MONITOR_ON_HOVER = "#2ecc71"
_BG_MONITOR_OFF = "#7f8c8d"
_BG_MONITOR_OFF_HOVER = "#95a5a6"
_DOT_IDLE = "#bdc3c7"
_DOT_HIJACK = "#f39c12"
_DOT_ACTIVE = "#e74c3c"


class InteractionGUI:
    """
    User-facing control panel.

    Two operator actions:
      1. CANCEL ALL ALERTS  - turns off active devices and clears hijacks/hold-down.
      2. MONITOR ON/OFF     - toggles the visualization DisplayProcess at runtime.

    Designed to stay visible during normal operation; lightweight (no per-stream grid).
    """

    def __init__(
        self,
        config: Config,
        on_cancel_all_callback: Callable[[], None],
        on_toggle_monitor_callback: Callable[[], bool],
        monitor_status_provider: Callable[[], bool],
        status_provider: Optional[Callable[[], dict]] = None,
        name: str = "InteractionGUI",
    ):
        self.name = name
        self.config = config
        self.on_cancel_all_callback = on_cancel_all_callback
        self.on_toggle_monitor_callback = on_toggle_monitor_callback
        self.monitor_status_provider = monitor_status_provider
        self.status_provider = status_provider

        self.root: Optional[tk.Tk] = None
        self.monitor_btn: Optional[tk.Button] = None
        self.activity_label: Optional[tk.Label] = None
        self.status_label: Optional[tk.Label] = None

        self._last_refresh = 0.0
        self._refresh_interval = 0.25  # 4 Hz

    def setup(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as e:
            logger.error(f"[{self.name}] Tkinter unavailable: {e}")
            self.root = None
            return

        self.root.title("Bird Count - Operator Control")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)
        self.root.configure(bg=_BG_PAGE)

        # Place top-right of screen so it doesn't collide with the manual hijack panel.
        try:
            sw = self.root.winfo_screenwidth()
            self.root.geometry(f"360x260+{max(0, sw - 380)}+20")
        except tk.TclError:
            pass

        # 1. Header
        header = tk.Frame(self.root, bg=_BG_HEADER)
        header.pack(pady=10, fill="x")
        tk.Label(
            header,
            text="OPERATOR CONTROL",
            fg="white",
            bg=_BG_HEADER,
            font=("Segoe UI", 12, "bold"),
        ).pack()

        # 2. Cancel All
        cancel_frame = tk.Frame(self.root, bg=_BG_PAGE)
        cancel_frame.pack(padx=12, pady=(12, 6), fill="x")  # tuple 在这里才有效
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

        # 3. Monitor Toggle
        monitor_frame = tk.Frame(self.root, bg=_BG_PAGE)
        monitor_frame.pack(padx=12, pady=6, fill="x")
        self.monitor_btn = tk.Button(
            monitor_frame,
            text="MONITOR: ...",
            fg="white",
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            height=2,
            command=self._handle_toggle_monitor_click,
        )
        self.monitor_btn.pack(fill="x")
        self._refresh_monitor_button()

        # 4. Activity summary + status bar
        self.activity_label = tk.Label(
            self.root,
            text="Active devices: 0",
            bg=_BG_PAGE,
            fg="#2c3e50",
            font=("Segoe UI", 9),
            pady=4,
        )
        self.activity_label.pack(fill="x")

        self.status_label = tk.Label(
            self.root,
            text="Ready...",
            bd=1,
            relief="sunken",
            anchor="w",
            font=("Segoe UI", 8),
            padx=10,
            pady=4,
        )
        self.status_label.pack(fill="x", side="bottom")

    def _handle_cancel_all_click(self):
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

        if self.status_label:
            self.status_label.config(
                text=f"All alerts cancelled at {time.strftime('%H:%M:%S')}",
                fg="#c0392b",
            )

    def _handle_toggle_monitor_click(self):
        try:
            new_state = self.on_toggle_monitor_callback()
        except Exception as e:
            logger.error(f"[{self.name}] toggle_monitor callback failed: {e}")
            return
        self._refresh_monitor_button(new_state)
        if self.status_label:
            label = "ON" if new_state else "OFF"
            self.status_label.config(
                text=f"Monitor turned {label} at {time.strftime('%H:%M:%S')}",
                fg="#27ae60" if new_state else "#7f8c8d",
            )

    def _refresh_monitor_button(self, state: Optional[bool] = None):
        if not self.monitor_btn:
            return
        if state is None:
            try:
                state = bool(self.monitor_status_provider())
            except Exception:
                state = False
        if state:
            self.monitor_btn.config(
                text="MONITOR: ON  (click to turn OFF)",
                bg=_BG_MONITOR_ON,
                activebackground=_BG_MONITOR_ON_HOVER,
            )
        else:
            self.monitor_btn.config(
                text="MONITOR: OFF  (click to turn ON)",
                bg=_BG_MONITOR_OFF,
                activebackground=_BG_MONITOR_OFF_HOVER,
            )

    def _refresh_activity(self):
        if not self.activity_label or not self.status_provider:
            return
        try:
            snapshot = self.status_provider() or {}
        except Exception:
            return
        total = sum(len(v) for v in snapshot.values() if v)
        parts = [f"{k}={len(v)}" for k, v in snapshot.items() if v]
        suffix = f"  ({', '.join(parts)})" if parts else ""
        self.activity_label.config(text=f"Active devices: {total}{suffix}")

    def update(self):
        if not self.root:
            return
        try:
            now = time.time()
            if now - self._last_refresh >= self._refresh_interval:
                self._refresh_monitor_button()
                self._refresh_activity()
                self._last_refresh = now
            self.root.update()
        except tk.TclError:
            self.root = None

    def destroy(self):
        if self.root:
            try:
                logger.info(f"[{self.name}] Destroying interaction GUI...")
                self.root.destroy()
                self.root = None
            except Exception as e:
                logger.error(f"[{self.name}] Error during destroy: {e}")


class ManualTriggerGUI:
    """
    Debug-only manual hijack panel.

    A grid of per-stream toggle buttons. Each click flips that stream's hijack state.
    No 'Cancel All' button - that lives on the operator-facing InteractionGUI.

    If `master` is provided this attaches as a Toplevel (recommended when an
    InteractionGUI already owns a Tk root in the same process).
    """

    def __init__(
        self,
        config: Config,
        on_trigger_callback: Callable[[int], None],
        status_provider: Optional[Callable[[], dict]] = None,
        master: Optional[tk.Misc] = None,
        name: str = "TriggerGUI",
    ):
        self.name = name
        self.config = config
        self.on_trigger_callback = on_trigger_callback
        self.status_provider = status_provider
        self.master = master

        self.root: Optional[tk.Misc] = None
        self.status_label: Optional[tk.Label] = None

        self.hijack_states: dict[int, bool] = dict.fromkeys(config.sid_to_ip.keys(), False)
        self.buttons: dict[int, tk.Button] = {}
        self.dots: dict[int, tk.Label] = {}

        self.stream_to_devices: dict[int, set[str]] = {}
        for sid, zone in config.sid_to_zone.items():
            ips: set[str] = set()
            ips.update(getattr(zone, "speakers", []) or [])
            ips.update(getattr(zone, "smart_plugs", []) or [])
            self.stream_to_devices[sid] = ips

        self._last_status_refresh = 0.0
        self._status_refresh_interval = 0.25

    def setup(self):
        try:
            if self.master is not None:
                self.root = tk.Toplevel(self.master)
            else:
                self.root = tk.Tk()
        except tk.TclError as e:
            logger.error(f"[{self.name}] Tkinter unavailable: {e}")
            self.root = None
            return

        self.root.title("DEBUG - Manual Hijack")
        try:
            self.root.attributes("-topmost", True)
        except tk.TclError:
            pass
        self.root.configure(bg=_BG_PAGE)

        header = tk.Frame(self.root, bg=_BG_HEADER, pady=10)
        header.pack(fill="x")
        tk.Label(
            header,
            text="DEBUG - MANUAL HIJACK",
            fg="white",
            bg=_BG_HEADER,
            font=("Segoe UI", 12, "bold"),
        ).pack()

        grid_container = tk.Frame(self.root, bg=_BG_PAGE)
        grid_container.pack(fill="both", expand=True, padx=10, pady=10)

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
        btn = self.buttons.get(stream_id)
        if not btn:
            return
        if self.hijack_states.get(stream_id, False):
            btn.config(bg="#f39c12", fg="white", highlightbackground="#e67e22")
        else:
            btn.config(bg="#ffffff", fg="#2c3e50", highlightbackground="#cccccc")

    def _handle_click(self, stream_id: int):
        self.hijack_states[stream_id] = not self.hijack_states[stream_id]
        self._refresh_button_style(stream_id)
        self.on_trigger_callback(stream_id)

        if self.status_label:
            state_text = "ENABLED" if self.hijack_states[stream_id] else "DISABLED"
            self.status_label.config(
                text=f"Stream {stream_id} Hijack {state_text} at {time.strftime('%H:%M:%S')}",
                fg="#e67e22" if self.hijack_states[stream_id] else "#2980b9",
            )

    def reset_all_hijacks(self):
        """Visually clear every hijack indicator (e.g., after a Cancel All from another GUI)."""
        for sid in list(self.hijack_states.keys()):
            self.hijack_states[sid] = False
            self._refresh_button_style(sid)

    def _refresh_status_dots(self):
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
            # Toplevel children are pumped by the master Tk's update(); only call
            # update() on our own root when we own the Tk.
            if self.master is None:
                self.root.update()
        except tk.TclError:
            self.root = None

    def destroy(self):
        if self.root:
            try:
                logger.info(f"[{self.name}] Destroying manual trigger GUI...")
                self.root.destroy()
                self.root = None
            except Exception as e:
                logger.error(f"[{self.name}] Error during destroy: {e}")
