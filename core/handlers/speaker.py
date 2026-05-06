import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx  # High-performance async HTTP client

from ..audit import AuditLog
from ..config import Config
from .base import BaseHandler, InferenceResult


logger = logging.getLogger(__name__)


class SpeakerClient:
    """Custom Client for Industrial Speakers."""

    def __init__(self, ip: str):
        self.ip = ip
        self.auth = httpx.BasicAuth("admin", "admin")
        self.base_url = f"http://{ip}/cgi-bin"

    async def play_audio(self, client: httpx.AsyncClient, filename: str = "7MB.wav"):
        url = f"{self.base_url}/audio_play?name={filename}&action=start&time=1"
        response = await client.get(url)
        response.raise_for_status()
        return response

    async def stop_audio(self, client: httpx.AsyncClient, filename: str = "7MB.wav"):
        url = f"{self.base_url}/audio_play?name={filename}&action=stop&time=1"
        response = await client.get(url)
        response.raise_for_status()
        return response

    async def get_status(self, client: httpx.AsyncClient) -> str:
        url = f"{self.base_url}/audio_get_status"
        response = await client.get(url)
        return response.text


class SpeakerHandler(BaseHandler):
    """
    High-performance Async Speaker Handler using httpx.

    Behavior Logic:
    - Sets needs_frames to False to eliminate SHM overhead.
    - Uses asyncio + httpx to handle non-blocking audio broadcasts.
    - Implements a "Trigger-and-Loop" pattern until manual stop OR cancel_all().
    """

    def __init__(self, config: Config, name: str = "Speaker"):
        super().__init__(name=name, needs_frames=False)
        self._audit_log_path = config.envs.audit_log_path
        self.audit: Optional[AuditLog] = None

        self.enable = config.envs.enable_speaker
        if not self.enable:
            return

        self.stream_to_speakers: dict[int, set[str]] = {sid: zone.speakers for sid, zone in config.sid_to_zone.items()}

        self._all_unique_ips: list[str] = sorted({ip for ips in self.stream_to_speakers.values() for ip in ips})
        self.device_states: dict[str, str] = dict.fromkeys(self._all_unique_ips, "READY")

        # Threading primitives MUST NOT exist before pickling — _thread.lock and
        # threading.Event aren't picklable, which breaks spawn-based mp on Windows.
        # The properties below lazy-create them on first access in the child process.
        self._state_lock: Optional[threading.Lock] = None
        self._cancel_events: Optional[dict[str, threading.Event]] = None

        self._executor = None
        logger.info(f"[{self.name}] Async Handler initialized for {len(self._all_unique_ips)} devices.")

    @property
    def state_lock(self) -> threading.Lock:
        """Lazy-init lock on first access (post-spawn)."""
        if self._state_lock is None:
            self._state_lock = threading.Lock()
        return self._state_lock

    @property
    def cancel_events(self) -> dict[str, threading.Event]:
        """Lazy-init per-device cancel events on first access (post-spawn)."""
        if self._cancel_events is None:
            self._cancel_events = {ip: threading.Event() for ip in self._all_unique_ips}
        return self._cancel_events

    @property
    def executor(self):
        """Initialize the thread pool only after the process starts."""
        if self._executor is None:
            num_threads = max(1, len(self.device_states))
            self._executor = ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix=self.name)
        return self._executor

    def start(self):
        super().start()
        self.audit = AuditLog(self._audit_log_path, name=self.name)
        if self.enable and self.audit:
            self.audit.log("handler.start", handler=self.name, devices=sorted(self.device_states.keys()))

    def handle(self, result: InferenceResult, frame: Optional[object]):
        """Entry point for inference results. Frame is None."""
        if not result.alert_flag:
            return

        target_ips = self.stream_to_speakers.get(result.stream_id, [])
        for ip in target_ips:
            with self.state_lock:
                if self.device_states.get(ip) != "READY":
                    continue
                self.device_states[ip] = "ACTIVE"
                self.cancel_events[ip].clear()
            if self.audit:
                self.audit.log(
                    "device.activate",
                    handler=self.name,
                    ip=ip,
                    stream_id=result.stream_id,
                    count=result.count,
                )
            self.executor.submit(self._run_async_lifecycle, ip)

    def _run_async_lifecycle(self, ip: str):
        """Bridge between ThreadPool and Asyncio loop."""
        error_msg = None
        try:
            asyncio.run(self._async_broadcast_task(ip))
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[{self.name}] Async lifecycle error on {ip}: {e}")
        finally:
            with self.state_lock:
                self.device_states[ip] = "READY"
                self.cancel_events[ip].clear()
            if self.audit:
                self.audit.log("device.deactivate", handler=self.name, ip=ip, error=error_msg)
            logger.info(f"[{self.name}] {ip} cycle finished. Returned to READY state.")

    async def _async_broadcast_task(self, ip: str):
        """
        Native Async Loop using httpx. Loops until manual stop, cancel signal, or safety timeout.
        """
        speaker = SpeakerClient(ip)
        cancel_event = self.cancel_events[ip]
        loop = asyncio.get_running_loop()

        start_time = time.time()
        max_duration = 3600

        async with httpx.AsyncClient(auth=speaker.auth, timeout=5.0) as client:
            while (time.time() - start_time) < max_duration:
                # Check cancel before issuing a play.
                if cancel_event.is_set():
                    break

                try:
                    await speaker.play_audio(client)
                except httpx.HTTPError as e:
                    logger.warning(f"[{self.name}] Network error on {ip}: {e}. Retrying in 10s...")

                # Sleep up to 10s but wake immediately on cancel.
                cancelled = await loop.run_in_executor(None, cancel_event.wait, 10.0)
                if cancelled:
                    break

                # Optional: detect a manual stop via the speaker's status endpoint.
                try:
                    status = await speaker.get_status(client)
                    if "stopped" in status.lower() or "idle" in status.lower():
                        logger.info(f"[{self.name}] Manual stop detected on {ip}. Exiting broadcast.")
                        return
                except Exception:
                    pass  # Status endpoint optional; ignore failures.

            # Always issue an explicit stop on the way out (cancel or timeout).
            try:
                await speaker.stop_audio(client)
                logger.info(f"[{self.name}] {ip} broadcast stopped.")
            except Exception as e:
                logger.error(f"[{self.name}] Failed to stop audio on {ip}: {e}")

    def cancel_all(self):
        """Signal every active speaker to abort and stop broadcasting."""
        if not self.enable:
            return
        with self.state_lock:
            active_ips = [ip for ip, st in self.device_states.items() if st == "ACTIVE"]
            for ip in active_ips:
                self.cancel_events[ip].set()
        if active_ips:
            logger.info(f"[{self.name}] cancel_all() signalled {len(active_ips)} active speaker(s): {active_ips}")
            if self.audit:
                self.audit.log("handler.cancel_all", handler=self.name, devices=active_ips)

    def get_active_devices(self) -> set[str]:
        if not self.enable:
            return set()
        with self.state_lock:
            return {ip for ip, st in self.device_states.items() if st == "ACTIVE"}

    def stop(self):
        """Cleanup handler resources."""
        if self.enable:
            self.cancel_all()
            self.executor.shutdown(wait=False)
        if self.audit:
            self.audit.log("handler.stop", handler=self.name)
            self.audit.close()
