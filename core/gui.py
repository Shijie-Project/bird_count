import logging
from typing import Callable, Optional


# Standard GUI library
try:
    import tkinter as tk
except ImportError:
    tk = None

from .config import Config


logger = logging.getLogger(__name__)


class ManualTriggerGUI:
    """
    A standalone class to manage the manual trigger simulation window.
    """

    def __init__(self, config: Config, on_trigger_callback: Callable[[int], None]):
        self.config = config
        self.on_trigger_callback = on_trigger_callback
        self.root: Optional[tk.Tk] = None

    def setup(self):
        """Initializes the window and creates buttons for each stream."""
        if not tk:
            logger.error("Tkinter not found. GUI cannot be started.")
            return

        self.root = tk.Tk()
        self.root.title("Bird Count - Manual Trigger")
        self.root.attributes("-topmost", True)  # Always stay on top

        tk.Label(self.root, text="Manual Trigger Panel", font=("Arial", 12, "bold"), pady=10).pack()
        tk.Label(self.root, text="Simulate a detection for a specific stream:").pack(padx=20, pady=(0, 10))

        # Create buttons for each configured stream
        for stream_id, ip in self.config.sid_to_ip.items():
            btn_text = f"Stream {stream_id} ({ip})"
            # Use default argument 's=stream_id' to freeze the value in the lambda
            btn = tk.Button(
                self.root, text=btn_text, width=35, pady=5, command=lambda s=stream_id: self.on_trigger_callback(s)
            )
            btn.pack(padx=20, pady=5)

    def update(self):
        """Processes GUI events without blocking. Call this in a loop."""
        if self.root:
            try:
                self.root.update()
            except tk.TclError:
                # Handle case where user closes the window manually
                self.root = None
                logger.info("Manual Trigger GUI closed by user.")

    def destroy(self):
        """Clean up the GUI resources."""
        if self.root:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None
