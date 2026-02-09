import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx  # High-performance async HTTP client

from ..config import Config
from .base import BaseHandler, InferenceResult


logger = logging.getLogger(__name__)


class SpeakerHandler(BaseHandler):
    """
    High-performance Async Speaker Handler using httpx.

    Behavior Logic:
    - Sets needs_frames to False to eliminate SHM overhead.
    - Uses asyncio + httpx to handle non-blocking audio broadcasts.
    - Implements a "Trigger-and-Loop" pattern until manual stop.
    """

    def __init__(self, config: Config, name: str = "Speaker"):
        super().__init__(name=name, needs_frames=False)
        self.enable = config.envs.enable_speaker
        if not self.enable:
            return

        # 1. Simplified Mapping: Stream ID -> Set of associated Speaker IPs
        self.stream_to_speakers: dict[int, set[str]] = {sid: zone.speakers for sid, zone in config.sid_to_zone.items()}

        # 2. State Machine: READY (Idle), ACTIVE (Broadcasting Loop)
        all_unique_ips = {ip for ips in self.stream_to_speakers.values() for ip in ips}
        self.device_states: dict[str, str] = dict.fromkeys(all_unique_ips, "READY")

        # 3. ThreadPool for running the asyncio event loops per device
        self._executor = None
        logger.info(f"[{self.name}] Async Handler initialized for {len(all_unique_ips)} devices.")

    @property
    def executor(self):
        """Initialize the thread pool only after the process starts."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=max(1, len(self.device_states)), thread_name_prefix=self.name
            )
        return self._executor

    def handle(self, result: InferenceResult, frame: Optional[object]):
        """
        Entry point for inference results. Frame is None.
        """
        # OPTIMIZATION: Respond directly to pre-calculated alert flags.
        if not result.alert_flag:
            return

        target_ips = self.stream_to_speakers.get(result.stream_id, [])
        for ip in target_ips:
            # Double-check state before entering lock to reduce contention
            if self.device_states.get(ip) == "READY":
                self.device_states[ip] = "ACTIVE"
                self.executor.submit(self._run_async_lifecycle, ip)

    def _run_async_lifecycle(self, ip: str):
        """Bridge between ThreadPool and Asyncio loop."""
        try:
            asyncio.run(self._async_broadcast_task(ip))
        except Exception as e:
            logger.error(f"[{self.name}] Async lifecycle error on {ip}: {e}")
        finally:
            self.device_states[ip] = "READY"
            logger.info(f"[{self.name}] {ip} cycle finished. Returned to READY state.")

    async def _async_broadcast_task(self, ip: str):
        """
        Native Async Loop using httpx.
        Loops until manual stop or safety timeout.
        """
        # API credentials and URL (assuming default admin:admin for industrial speakers)
        auth = httpx.BasicAuth("admin", "admin")
        url = f"http://{ip}/cgi-bin/audio_play?name=7MB.wav&action=start&time=1"

        # Use a safety duration to prevent infinite loops (1 hour)
        start_time = time.time()
        max_duration = 3600

        # httpx AsyncClient handles connection pooling automatically
        async with httpx.AsyncClient(auth=auth, timeout=5.0) as client:
            while (time.time() - start_time) < max_duration:
                try:
                    # A. Trigger Playback
                    response = await client.get(url)
                    response.raise_for_status()

                    # B. Polling Interval
                    # We wait 10 seconds between play commands.
                    # This is also where you'd check for a 'manual stop' signal if available.
                    await asyncio.sleep(10.0)

                    # C. [OPTIONAL] Check for manual stop signal via a status API
                    # status_resp = await client.get(f"http://{ip}/cgi-bin/audio_get_status")
                    # if "stopped" in status_resp.text: break

                except httpx.HTTPError as e:
                    logger.warning(f"[{self.name}] Network error on {ip}: {e}. Retrying in 10s...")
                    await asyncio.sleep(10.0)

    def stop(self):
        """Cleanup handler resources."""
        if self.enable:
            self.executor.shutdown(wait=False)
