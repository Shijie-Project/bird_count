import logging
import time
import tkinter as tk
from abc import ABC, abstractmethod
from tkinter import messagebox
from typing import Callable, Optional, Protocol

from .config import Config


class _MonitorTogglable(Protocol):
    """Duck-typed contract the operator-panel factory needs from a monitor handler."""

    def toggle(self) -> bool: ...

    def is_enabled(self) -> bool: ...


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


# ----------------------------------------------------------------------------
# Component framework
# ----------------------------------------------------------------------------

# Type alias: callable used to push a message into the host GUI's status bar.
StatusSetter = Callable[..., None]


class GuiComponent(ABC):
    """
    A self-contained widget that mounts itself into a parent Tk frame and
    optionally refreshes on a periodic tick.

    Each component owns its own state, click handler, and visual refresh,
    so adding a new control = a new class + appending it to the GUI's
    component list. The host GUI knows nothing about the control type.
    """

    @abstractmethod
    def mount(self, parent: tk.Misc, set_status: Optional[StatusSetter] = None) -> None:
        """Build and pack this component into `parent`."""

    def refresh(self) -> None:
        """Optional periodic hook (called from the host GUI's update loop)."""
        return None


# ----------------------------------------------------------------------------
# Concrete components
# ----------------------------------------------------------------------------


class CancelAllButton(GuiComponent):
    """Red button with confirmation dialog. Fires `on_cancel_all` on confirm."""

    def __init__(self, on_cancel_all: Callable[[], None]):
        self.on_cancel_all = on_cancel_all
        self.btn: Optional[tk.Button] = None
        self.set_status: Optional[StatusSetter] = None

    def mount(self, parent: tk.Misc, set_status: Optional[StatusSetter] = None) -> None:
        self.set_status = set_status
        frame = tk.Frame(parent, bg=_BG_PAGE)
        frame.pack(padx=12, pady=(12, 6), fill="x")
        self.btn = tk.Button(
            frame,
            text="CANCEL ALL ALERTS",
            bg=_BG_CANCEL,
            fg="white",
            activebackground=_BG_CANCEL_HOVER,
            activeforeground="white",
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            height=2,
            command=self._on_click,
        )
        self.btn.pack(fill="x")
        self.btn.bind("<Enter>", lambda e: self.btn.config(bg=_BG_CANCEL_HOVER))
        self.btn.bind("<Leave>", lambda e: self.btn.config(bg=_BG_CANCEL))

    def _on_click(self) -> None:
        if not self.btn:
            return
        confirmed = messagebox.askyesno(
            title="Confirm Cancel All",
            message="Cancel all active alerts and clear all manual hijacks?\n\n"
            "This will turn off every active speaker/plug and reset hold-down timers.",
            parent=self.btn.winfo_toplevel(),
        )
        if not confirmed:
            return
        try:
            self.on_cancel_all()
        except Exception as e:
            logger.error(f"[CancelAllButton] callback failed: {e}")
        if self.set_status:
            self.set_status(f"All alerts cancelled at {time.strftime('%H:%M:%S')}", fg="#c0392b")


class MonitorToggleButton(GuiComponent):
    """
    Bicolor ON/OFF button driven by a state provider. Calls `on_toggle()`
    (which itself returns the new state) and re-syncs visually each tick so
    external state changes reflect even when the user didn't click.
    """

    def __init__(self, on_toggle: Callable[[], bool], status_provider: Callable[[], bool]):
        self.on_toggle = on_toggle
        self.status_provider = status_provider
        self.btn: Optional[tk.Button] = None
        self.set_status: Optional[StatusSetter] = None

    def mount(self, parent: tk.Misc, set_status: Optional[StatusSetter] = None) -> None:
        self.set_status = set_status
        frame = tk.Frame(parent, bg=_BG_PAGE)
        frame.pack(padx=12, pady=6, fill="x")
        self.btn = tk.Button(
            frame,
            text="MONITOR: ...",
            fg="white",
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            height=2,
            command=self._on_click,
        )
        self.btn.pack(fill="x")
        self._apply_appearance()

    def _on_click(self) -> None:
        try:
            new_state = bool(self.on_toggle())
        except Exception as e:
            logger.error(f"[MonitorToggleButton] toggle failed: {e}")
            return
        self._apply_appearance(new_state)
        if self.set_status:
            label = "ON" if new_state else "OFF"
            self.set_status(
                f"Monitor turned {label} at {time.strftime('%H:%M:%S')}",
                fg="#27ae60" if new_state else "#7f8c8d",
            )

    def _apply_appearance(self, state: Optional[bool] = None) -> None:
        if not self.btn:
            return
        if state is None:
            try:
                state = bool(self.status_provider())
            except Exception:
                state = False
        if state:
            self.btn.config(
                text="MONITOR: ON  (click to turn OFF)",
                bg=_BG_MONITOR_ON,
                activebackground=_BG_MONITOR_ON_HOVER,
            )
        else:
            self.btn.config(
                text="MONITOR: OFF  (click to turn ON)",
                bg=_BG_MONITOR_OFF,
                activebackground=_BG_MONITOR_OFF_HOVER,
            )

    def refresh(self) -> None:
        self._apply_appearance()


class ActivityLabel(GuiComponent):
    """Read-only label showing total + per-handler active device counts."""

    def __init__(self, status_provider: Callable[[], dict]):
        self.status_provider = status_provider
        self.label: Optional[tk.Label] = None

    def mount(self, parent: tk.Misc, set_status: Optional[StatusSetter] = None) -> None:
        self.label = tk.Label(
            parent,
            text="Active devices: 0",
            bg=_BG_PAGE,
            fg="#2c3e50",
            font=("Segoe UI", 9),
            pady=4,
        )
        self.label.pack(fill="x")

    def refresh(self) -> None:
        if not self.label:
            return
        try:
            snapshot = self.status_provider() or {}
        except Exception:
            return
        total = sum(len(v) for v in snapshot.values() if v)
        parts = [f"{k}={len(v)}" for k, v in snapshot.items() if v]
        suffix = f"  ({', '.join(parts)})" if parts else ""
        try:
            self.label.config(text=f"Active devices: {total}{suffix}")
        except tk.TclError:
            pass


# ----------------------------------------------------------------------------
# Host GUIs
# ----------------------------------------------------------------------------


class InteractionGUI:
    """
    Thin operator-facing shell. Owns the Tk root, header, and status bar; the
    body is a vertical stack of GuiComponents passed in by the caller. Adding a
    new control = define a new GuiComponent and append it to the components list.
    """

    def __init__(
        self,
        components: list[GuiComponent],
        title: str = "Bird Count - Operator Control",
        header_text: str = "OPERATOR CONTROL",
        name: str = "InteractionGUI",
    ):
        self.components = components
        self.title = title
        self.header_text = header_text
        self.name = name

        self.root: Optional[tk.Tk] = None
        self.status_label: Optional[tk.Label] = None

        self._last_refresh = 0.0
        self._refresh_interval = 0.25  # 4 Hz

    def setup(self) -> None:
        try:
            self.root = tk.Tk()
        except tk.TclError as e:
            logger.error(f"[{self.name}] Tkinter unavailable: {e}")
            self.root = None
            return

        self.root.title(self.title)
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)
        self.root.configure(bg=_BG_PAGE)

        # Place top-right of screen so it doesn't collide with the manual hijack panel.
        try:
            sw = self.root.winfo_screenwidth()
            self.root.geometry(f"360x320+{max(0, sw - 380)}+20")
        except tk.TclError:
            pass

        # Header
        header = tk.Frame(self.root, bg=_BG_HEADER)
        header.pack(pady=10, fill="x")
        tk.Label(
            header,
            text=self.header_text,
            fg="white",
            bg=_BG_HEADER,
            font=("Segoe UI", 12, "bold"),
        ).pack()

        # Status bar (created early so set_status works during component mount).
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

        # Body — components stack vertically, each one self-mounts.
        body = tk.Frame(self.root, bg=_BG_PAGE)
        body.pack(fill="both", expand=True)

        for comp in self.components:
            try:
                comp.mount(body, set_status=self.set_status)
            except Exception as e:
                logger.error(f"[{self.name}] {type(comp).__name__} mount failed: {e}")

    def set_status(self, text: str, fg: Optional[str] = None) -> None:
        """Pushed down into components so each one can update the shared status bar."""
        if not self.status_label:
            return
        try:
            kwargs: dict = {"text": text}
            if fg is not None:
                kwargs["fg"] = fg
            self.status_label.config(**kwargs)
        except tk.TclError:
            pass

    def update(self) -> None:
        if not self.root:
            return
        try:
            now = time.time()
            if now - self._last_refresh >= self._refresh_interval:
                for comp in self.components:
                    try:
                        comp.refresh()
                    except Exception as e:
                        logger.debug(f"[{self.name}] {type(comp).__name__} refresh error: {e}")
                self._last_refresh = now
            self.root.update()
        except tk.TclError:
            self.root = None

    def destroy(self) -> None:
        if self.root:
            try:
                logger.info(f"[{self.name}] Destroying interaction GUI...")
                self.root.destroy()
                self.root = None
            except Exception as e:
                logger.error(f"[{self.name}] Error during destroy: {e}")

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def operator_panel(
        cls,
        *,
        on_cancel_all: Callable[[], None],
        monitor_handler: Optional[_MonitorTogglable] = None,
        active_devices_provider: Optional[Callable[[], dict]] = None,
    ) -> "InteractionGUI":
        """
        Assemble the standard operator panel: Cancel All + (optional) Monitor toggle
        + (optional) active-devices summary. Caller only imports InteractionGUI.
        """
        components: list[GuiComponent] = [CancelAllButton(on_cancel_all=on_cancel_all)]
        if monitor_handler is not None:
            components.append(
                MonitorToggleButton(
                    on_toggle=monitor_handler.toggle,
                    status_provider=monitor_handler.is_enabled,
                )
            )
        if active_devices_provider is not None:
            components.append(ActivityLabel(status_provider=active_devices_provider))
        return cls(components)


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
