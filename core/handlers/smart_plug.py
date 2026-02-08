import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from ..config import Config
from .base import BaseHandler, BatchInferenceResult


# Make this optional to prevent crash if library missing
try:
    from tapo import ApiClient
except ImportError:
    ApiClient = None

logger = logging.getLogger(__name__)


class SmartPlugHandler(BaseHandler):
    """
    Industrial Handler for IoT Plugs (TP-Link Tapo) based on Zone Config.
    Trigger logic:
    If ANY camera in a Zone exceeds the Zone's threshold -> Turn ON that Zone's Plug.
    """

    def __init__(self, config: Config):
        self.enable = config.envs.enable_smart_plug
        if not self.enable:
            return

        if ApiClient is None:
            logger.error("'tapo' library not installed. Handler disabled.")
            self.enable = False
            return

        self.auth_email = config.plug_auth.email
        self.auth_password = config.plug_auth.password

        # 1. Map Stream ID -> {IPs, Threshold}
        # Note: One stream can trigger MULTIPLE plugs if configured in the zone.
        self.stream_map = {}
        unique_ips = set()

        for zone in getattr(config, "zones", []):
            plug_ips = getattr(zone, "smart_plugs", [])
            if not plug_ips:
                continue

            # Ensure we track all unique IPs for state management
            unique_ips |= set(plug_ips)

            # Map each camera in this zone to these plugs
            for cam_ip in getattr(zone, "cameras", []):
                sid = config.ip2id[cam_ip]
                self.stream_map[sid] = {"ips": plug_ips, "threshold": zone.threshold}

        if not self.stream_map:
            logger.warning("[SmartPlug] Enabled but no smart plugs configured in Zones.")
            self.enable = False
            return

        # 2. State Management
        # Device States: 'READY', 'ACTIVE'
        self.device_states = dict.fromkeys(unique_ips, "READY")
        self.state_lock = threading.Lock()

        # 3. Worker Pool
        self.executor = ThreadPoolExecutor(max_workers=max(1, len(unique_ips)), thread_name_prefix="PlugWorker")
        logger.info(f"[SmartPlug] Initialized. Controlling {len(unique_ips)} devices.")

    def handle_batch(self, batch_result: BatchInferenceResult, shm_client=None):
        """
        Check batch results against zone thresholds.
        """
        if not self.enable or batch_result is None:
            return

        ips_to_trigger = set()

        for res in batch_result.results:
            sid = res.stream_id

            if sid not in self.stream_map:
                continue

            zone_info = self.stream_map[sid]

            # Check Threshold
            if res.count >= zone_info["threshold"]:
                for ip in zone_info["ips"]:
                    # Optimization: Peek state without lock first
                    if self.device_states.get(ip) == "READY":
                        ips_to_trigger.add(ip)
                        # logger.debug(f"Trigger condition met on Stream {sid} for Plug {ip}")

        # Execute triggers
        for ip in ips_to_trigger:
            self._trigger_if_ready(ip)

    def handle(self, result, frame):
        """Legacy interface stub."""
        pass

    def _trigger_if_ready(self, ip: str):
        """Thread-safe trigger logic."""
        trigger = False
        with self.state_lock:
            if self.device_states[ip] == "READY":
                self.device_states[ip] = "ACTIVE"
                trigger = True

        if trigger:
            logger.info(f"[SmartPlug] Triggering sequence for {ip}...")
            self.executor.submit(self._run_lifecycle, ip)

    def _run_lifecycle(self, ip: str):
        """Wrapper to run async code in thread."""
        try:
            asyncio.run(self._async_task(ip))
        except Exception as e:
            logger.error(f"[SmartPlug] Lifecycle failed for {ip}: {e}")
        finally:
            # Always reset state when done (even if error)
            with self.state_lock:
                self.device_states[ip] = "READY"
                logger.info(f"[SmartPlug] {ip} cycle finished. Reset to READY.")

    async def _async_task(self, ip: str):
        """Async sequence: ON -> Monitor -> Finish"""
        client = ApiClient(self.auth_email, self.auth_password)
        try:
            device = await client.p100(ip)

            # A. Turn ON
            # logger.info(f"[SmartPlug] Sending ON command to {ip}...")
            await asyncio.wait_for(device.on(), timeout=10.0)

            # B. Monitoring Loop (Wait for manual turn off)
            logger.info(f"[SmartPlug] {ip} is ON. Monitoring for manual shutdown...")

            monitor_start = time.time()
            max_monitor_time = 3600  # 1 Hour Safety Timeout

            while (time.time() - monitor_start) < max_monitor_time:
                await asyncio.sleep(5.0)  # Check every 5s

                try:
                    info = await asyncio.wait_for(device.get_device_info(), timeout=5.0)
                    if not info.device_on:
                        logger.info(f"[SmartPlug] Manual shutdown detected on {ip}.")
                        break
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.warning(f"[SmartPlug] Status check warning {ip}: {e}")
                    await asyncio.sleep(5.0)

        except Exception as e:
            logger.error(f"[SmartPlug] Async error on {ip}: {e}")

    def stop(self):
        if self.enable:
            self.executor.shutdown(wait=False)
