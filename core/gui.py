import logging
import time
import tkinter as tk
from typing import Callable, Optional

from .config import Config


logger = logging.getLogger(__name__)


class ManualTriggerGUI:
    """
    Improved Manual Trigger Panel with visible borders and Hijack state feedback.
    """

    def __init__(self, config: Config, on_trigger_callback: Callable[[int], None], name: str = "TriggerGUI"):
        self.name = name
        self.config = config
        self.on_trigger_callback = on_trigger_callback
        self.root: Optional[tk.Tk] = None
        self.status_label: Optional[tk.Label] = None

        # Track which streams are currently "Hijacked" to update button colors
        self.hijack_states: dict[int, bool] = dict.fromkeys(config.sid_to_ip.keys(), False)
        self.buttons: dict[int, tk.Button] = {}

    def setup(self):
        if not tk:
            logger.error(f"[{self.name}] Tkinter not found.")
            return

        self.root = tk.Tk()
        self.root.title("System Control - Manual Hijack")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)
        self.root.configure(bg="#f0f0f0")

        # --- 1. Header ---
        header_frame = tk.Frame(self.root, bg="#2c3e50", pady=10)
        header_frame.pack(fill="x")
        tk.Label(
            header_frame, text="PRO-INFERENCE HIJACK CONTROL", fg="white", bg="#2c3e50", font=("Segoe UI", 12, "bold")
        ).pack()

        # --- 2. Button Grid ---
        main_frame = tk.Frame(self.root, bg="#f0f0f0", padx=20, pady=20)
        main_frame.pack()

        max_cols = 4
        for i, (stream_id, ip) in enumerate(self.config.sid_to_ip.items()):
            row, col = divmod(i, max_cols)
            btn_frame = tk.Frame(main_frame, bg="#f0f0f0", padx=5, pady=5)
            btn_frame.grid(row=row, column=col)

            # OPTIMIZATION: Added highlightthickness and highlightbackground for a clear border
            # Changed bg to white for better contrast against gray background
            btn = tk.Button(
                btn_frame,
                text=f"STREAM {stream_id}\n{ip}",
                width=16,
                height=3,
                bg="#ffffff",  # Clear white background
                fg="#2c3e50",  # Dark text
                relief="flat",  # Keep flat but use highlight for border
                highlightthickness=1,  # Force a 1px border
                highlightbackground="#cccccc",  # Light gray border line
                font=("Segoe UI", 9),
                command=lambda s=stream_id: self._handle_click(s),
            )
            btn.pack()

            # Save reference for state updates
            self.buttons[stream_id] = btn

            # Simple Hover Effect: Change border color on hover
            btn.bind("<Enter>", lambda e, b=btn: b.config(highlightbackground="#3498db"))
            btn.bind("<Leave>", lambda e, b=btn, s=stream_id: self._refresh_button_style(s))

        # --- 3. Status Bar ---
        self.status_label = tk.Label(
            self.root, text="Ready...", bd=1, relief="sunken", anchor="w", font=("Segoe UI", 8), padx=10, pady=5
        )
        self.status_label.pack(fill="x")

    def _refresh_button_style(self, stream_id: int):
        """Updates button color based on whether it is hijacked or not."""
        btn = self.buttons.get(stream_id)
        if not btn:
            return

        if self.hijack_states[stream_id]:
            # Hijacked State: Warning Orange/Yellow
            btn.config(bg="#f39c12", fg="white", highlightbackground="#e67e22")
        else:
            # Normal State: Clean White
            btn.config(bg="#ffffff", fg="#2c3e50", highlightbackground="#cccccc")

    def _handle_click(self, stream_id: int):
        """Toggles state and triggers callback."""
        # 1. Toggle local state
        self.hijack_states[stream_id] = not self.hijack_states[stream_id]

        # 2. Update visual style immediately
        self._refresh_button_style(stream_id)

        # 3. Call the ResultProcess hijack logic
        self.on_trigger_callback(stream_id)

        # 4. Update status text
        if self.status_label:
            state_text = "ENABLED" if self.hijack_states[stream_id] else "DISABLED"
            self.status_label.config(
                text=f"Stream {stream_id} Hijack {state_text} at {time.strftime('%H:%M:%S')}",
                fg="#e67e22" if self.hijack_states[stream_id] else "#2980b9",
            )

    def update(self):
        if self.root:
            try:
                self.root.update()
            except tk.TclError:
                self.root = None

    def destroy(self):
        """
        Explicitly destroy the GUI and release Tcl interpreter resources.
        """
        if self.root:
            try:
                logger.info(f"[{self.name}] Destroying manual trigger GUI...")
                self.root.destroy()
                self.root = None
            except Exception as e:
                logger.error(f"[{self.name}] Error during destroy: {e}")
