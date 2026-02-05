import asyncio
import logging
import threading

from tapo import ApiClient

from .base import BaseResultHandler, ResultItem


logger = logging.getLogger(__name__)


class SmartPlugHandler(BaseResultHandler):
    def __init__(self, cfg):
        self.enable = cfg.smart_plug.enable
        if not self.enable:
            return

        self.tapo_email = cfg.smart_plug.email
        self.tapo_password = cfg.smart_plug.password
        self.threshold = cfg.smart_plug.alert_threshold
        self.timeout = cfg.smart_plug.timeout_seconds

        # Interval (seconds) to check if the plug was manually turned off
        self.check_interval = 10

        self.ip_to_sids = cfg.smart_plug.device_maps
        # States: 'READY' (can trigger), 'ACTIVE' (on and waiting for manual off)
        self.device_states = dict.fromkeys(self.ip_to_sids, "READY")
        self.state_lock = threading.Lock()

    def _get_ips_needing_activation(self, results: ResultItem):
        """Check thresholds and return IPs that are currently READY."""
        count_map = dict(zip(results.sids, results.counts))
        ips_to_trigger = []

        with self.state_lock:
            for ip, sids in self.ip_to_sids.items():
                if self.device_states[ip] == "READY":
                    max_count = max([count_map.get(s, 0) for s in sids])
                    if max_count >= self.threshold:
                        # Lock it to ACTIVE immediately to prevent spawning multiple threads
                        self.device_states[ip] = "ACTIVE"
                        ips_to_trigger.append(ip)
        return ips_to_trigger

    def handle(self, results: ResultItem):
        """Main entry: Identifies targets and starts their lifecycle threads."""
        if not self.enable or not results.counts:
            return

        targets = self._get_ips_needing_activation(results)
        for ip in targets:
            # Each IP gets its own lifecycle thread
            threading.Thread(target=self._run_device_lifecycle, args=(ip,), daemon=True).start()

    def _run_device_lifecycle(self, ip):
        """Thread worker: Handles ON command and subsequent manual-off detection."""
        asyncio.run(self._async_lifecycle_task(ip))

    async def _async_lifecycle_task(self, ip):
        try:
            client = ApiClient(self.tapo_email, self.tapo_password)
            device = await client.p100(ip)

            # 1. Perform Turn ON
            logger.info(f"[SmartPlug] ALERT: Count exceeded. Turning ON {ip}")
            await asyncio.wait_for(device.on(), timeout=self.timeout)

            # 2. Wait until detected as OFF (the manual-off detection phase)
            logger.info(f"[SmartPlug] {ip} is now ON. Entering manual-off monitoring mode.")
            while True:
                await asyncio.sleep(self.check_interval)

                # Low-frequency status check
                info = await asyncio.wait_for(device.get_device_info(), timeout=self.timeout)
                if not info.device_on:
                    logger.info(f"[SmartPlug] Manual shutdown detected for {ip}. Resetting to READY state.")
                    break

            # 3. Return to READY state
            with self.state_lock:
                self.device_states[ip] = "READY"

        except Exception as e:
            logger.error(f"[SmartPlug] Error in lifecycle for {ip}: {e}")
            # Reset on failure to allow retry
            with self.state_lock:
                self.device_states[ip] = "READY"
