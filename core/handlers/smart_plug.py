import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

from ..config import Config
from .base import BaseHandler, InferenceResult


# Make this optional to prevent crash if library missing
try:
    from tapo import ApiClient
except ImportError:
    ApiClient = None

logger = logging.getLogger(__name__)


class SmartPlugHandler(BaseHandler):
    """
    Optimized Industrial Handler for IoT Plugs (TP-Link Tapo).

    Trigger Logic:
    - Responds directly to 'alert_flag' from InferenceResult.
    - Turns ON assigned plugs when an alert is detected.
    - Stays in 'ACTIVE' state until the plug is MANUALLY turned off OR cancel_all() is called.
    """

    def __init__(self, config: Config, name="SmartPlug", max_idle_time=3600):
        super().__init__(name=name, needs_frames=False)
        self.enable = config.envs.enable_smart_plug
        if not self.enable or ApiClient is None:
            if ApiClient is None and self.enable:
                logger.error(f"[{self.name}] 'tapo' library not installed. SmartPlugHandler disabled.")
            self.enable = False
            return

        self.max_idle_time = max_idle_time

        self.auth_email = config.plug_auth.email
        self.auth_password = config.plug_auth.password

        self.stream_to_plugs: dict[int, set[str]] = {sid: zone.smart_plugs for sid, zone in config.sid_to_zone.items()}

        all_unique_ips = {ip for ips in self.stream_to_plugs.values() for ip in ips}
        self.device_states: dict[str, str] = dict.fromkeys(all_unique_ips, "READY")

        # Cancellation primitives — one Event per device, all guarded by a single lock.
        self._state_lock = threading.Lock()
        self.cancel_events: dict[str, threading.Event] = {ip: threading.Event() for ip in all_unique_ips}

        self._executor = None
        logger.info(f"[{self.name}] Initialized with {len(all_unique_ips)} unique devices.")

    @property
    def executor(self):
        """Initialize the thread pool only after the process starts."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=max(1, len(self.device_states)), thread_name_prefix=self.name
            )
        return self._executor

    def handle(self, result: InferenceResult, frame: Optional[np.ndarray]):
        """Processes a single result. Frame is always None (needs_frames=False)."""
        if not result.alert_flag:
            return

        target_ips = self.stream_to_plugs.get(result.stream_id, [])
        for ip in target_ips:
            # Atomic READY -> ACTIVE transition
            with self._state_lock:
                if self.device_states.get(ip) != "READY":
                    continue
                self.device_states[ip] = "ACTIVE"
                self.cancel_events[ip].clear()
            self.executor.submit(self._run_lifecycle, ip)

    def _run_lifecycle(self, ip: str):
        """Runs the async monitoring task in a dedicated thread."""
        try:
            asyncio.run(self._async_task(ip))
        except Exception as e:
            logger.error(f"[{self.name}] Task failed for {ip}: {e}")
        finally:
            with self._state_lock:
                self.device_states[ip] = "READY"
                self.cancel_events[ip].clear()
            logger.debug(f"[{self.name}] {ip} reset to READY state.")

    async def _async_task(self, ip: str):
        """
        Async Logic: Turn ON -> wait for manual hardware off OR cancel signal -> turn OFF on cancel -> Exit.
        """
        client = ApiClient(self.auth_email, self.auth_password)
        cancel_event = self.cancel_events[ip]
        loop = asyncio.get_running_loop()

        try:
            device = await client.p100(ip)

            # Step A: Turn ON
            await asyncio.wait_for(device.on(), timeout=10.0)
            logger.info(f"[{self.name}] {ip} is now ON.")

            # Step B: Wait for either manual hardware shutdown or a cancel signal.
            start_time = time.time()

            while (time.time() - start_time) < self.max_idle_time:
                # Wake up either every 5s (to poll device state) or immediately on cancel.
                cancelled = await loop.run_in_executor(None, cancel_event.wait, 5.0)
                if cancelled:
                    logger.info(f"[{self.name}] Cancel signal received for {ip}. Turning OFF.")
                    try:
                        await asyncio.wait_for(device.off(), timeout=10.0)
                    except Exception as e:
                        logger.error(f"[{self.name}] Failed to issue OFF on cancel for {ip}: {e}")
                    return

                try:
                    info = await asyncio.wait_for(device.get_device_info(), timeout=5.0)
                    if not info.device_on:
                        logger.info(f"[{self.name}] Manual shutdown detected on {ip}. Task complete.")
                        return
                except Exception:
                    continue  # Ignore transient network issues during monitoring

        except Exception as e:
            logger.error(f"[{self.name}] Async error on {ip}: {e}")

    def cancel_all(self):
        """Signal every active device to abort, turn off, and return to READY."""
        if not self.enable:
            return
        with self._state_lock:
            active_ips = [ip for ip, st in self.device_states.items() if st == "ACTIVE"]
            for ip in active_ips:
                self.cancel_events[ip].set()
        if active_ips:
            logger.info(f"[{self.name}] cancel_all() signalled {len(active_ips)} active plug(s): {active_ips}")

    def get_active_devices(self) -> set[str]:
        if not self.enable:
            return set()
        with self._state_lock:
            return {ip for ip, st in self.device_states.items() if st == "ACTIVE"}

    def stop(self):
        """Shuts down the worker pool."""
        if self.enable:
            self.cancel_all()
            self.executor.shutdown(wait=False)
            logger.info(f"[{self.name}] Handler stopped.")
