import asyncio
import logging
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
    - Stays in 'ACTIVE' state until the plug is MANUALLY turned off.
    """

    def __init__(self, config: Config, name="SmartPlug"):
        super().__init__(name=name, needs_frames=False)
        self.enable = config.envs.enable_smart_plug
        if not self.enable or ApiClient is None:
            if ApiClient is None and self.enable:
                logger.error(f"[{self.name}] 'tapo' library not installed. SmartPlugHandler disabled.")
            self.enable = False
            return

        self.auth_email = config.plug_auth.email
        self.auth_password = config.plug_auth.password

        # 1. Map Stream ID -> {IPs, Threshold}
        # Note: One stream can trigger MULTIPLE plugs if configured in the zone.
        self.stream_to_plugs: dict[int, set[str]] = {sid: zone.smart_plugs for sid, zone in config.sid_to_zone.items()}

        all_unique_ips = {ip for ips in self.stream_to_plugs.values() for ip in ips}
        self.device_states: dict[str, str] = dict.fromkeys(all_unique_ips, "READY")

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
        """
        Processes a single result.
        Note: Frame is always None here because needs_frames=False.
        """
        # OPTIMIZATION 2: Directly trust the alert_flag calculated by ResultProcess.
        if not result.alert_flag:
            return

        target_ips = self.stream_to_plugs.get(result.stream_id, [])
        for ip in target_ips:
            # Quick check without lock to reduce overhead
            if self.device_states.get(ip) == "READY":
                self.device_states[ip] = "ACTIVE"
                self.executor.submit(self._run_lifecycle, ip)

    def _run_lifecycle(self, ip: str):
        """Runs the async monitoring task in a dedicated thread."""
        try:
            asyncio.run(self._async_task(ip))
        except Exception as e:
            logger.error(f"[{self.name}] Task failed for {ip}: {e}")
        finally:
            # Reset state back to READY so it can be triggered again later
            self.device_states[ip] = "READY"
            logger.debug(f"[{self.name}] {ip} reset to READY state.")

    async def _async_task(self, ip: str):
        """
        Async Logic: Turn ON -> Poll until MANUALLY turned off -> Exit.
        """
        client = ApiClient(self.auth_email, self.auth_password)
        try:
            device = await client.p100(ip)

            # Step A: Turn ON the device
            await asyncio.wait_for(device.on(), timeout=10.0)
            logger.info(f"[{self.name}] {ip} is now ON.")

            # Step B: Wait for manual shutdown
            # We check the state periodically. The loop exits when a human turns it off.
            max_idle_time = 3600  # Safety timeout (1 hour)
            start_time = time.time()

            while (time.time() - start_time) < max_idle_time:
                await asyncio.sleep(5.0)  # Check every 5 seconds

                try:
                    info = await asyncio.wait_for(device.get_device_info(), timeout=5.0)
                    if not info.device_on:
                        logger.info(f"[{self.name}] Manual shutdown detected on {ip}. Task complete.")
                        break
                except Exception:
                    continue  # Ignore transient network issues during monitoring

        except Exception as e:
            logger.error(f"[{self.name}] Async error on {ip}: {e}")

    def stop(self):
        """Shuts down the worker pool."""
        if self.enable:
            self.executor.shutdown(wait=False)
            logger.info(f"[{self.name}] Handler stopped.")
