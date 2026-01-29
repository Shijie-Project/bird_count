import asyncio
import logging
import threading
from typing import Optional

from tapo import ApiClient

from .base import BaseResultHandler


logger = logging.getLogger(__name__)


class SmartPlugHandler(BaseResultHandler):
    """
    Independent Multi-Channel Smart Plug Handler.

    Features:
    - **Independent Control**: Controls devices individually without blocking others.
    - **Process-Safe**: safe to be instantiated in parent process and passed to child workers (lazy locking).
    - **Non-blocking**: Uses background threads for slow Network I/O.
    - **Threshold Logic**: Switches ON if bird count > threshold, otherwise switches OFF.
    """

    def __init__(self, cfg):
        self.enable = cfg.smart_plug.enable
        if not self.enable:
            return

        # Credentials
        self.tapo_email = cfg.smart_plug.email
        self.tapo_password = cfg.smart_plug.password

        self.timeout_seconds = cfg.smart_plug.timeout_seconds
        self.alert_threshold = cfg.smart_plug.alert_threshold

        # Device Mapping: {'zone1': '192.168.1.100', 'zone2': '192.168.1.101'}
        self.device_ips: list[str] = cfg.smart_plug.devices
        self.sid_to_ip: dict[int, str] = {}
        for idx, ip in enumerate(self.device_ips):
            self.sid_to_ip[idx] = ip

        # 2. State Management
        # Stores the last known state of the plug (True=ON, False=OFF, None=Unknown)
        self.current_states: dict[str, Optional[bool]] = dict.fromkeys(self.device_ips)

        # 3. Concurrency Control
        # A set to track devices currently performing network I/O
        self.busy_ips: set[str] = set()

        # Initialize internal lock variable as None (Lazy Loading)
        self._lock: Optional[threading.Lock] = None

        logger.info(f"[Plug] Initialized. Monitoring {len(self.device_ips)} devices.")

        # Reset all devices to OFF on startup
        self._reset_all_devices_off()

    @property
    def lock(self) -> threading.Lock:
        """
        Lazy-loaded lock property.
        Creates a new lock if one does not exist (e.g., after process forking).
        """
        if self._lock is None:
            self._lock = threading.Lock()
        return self._lock

    def __getstate__(self):
        """Pickling support: drop the lock."""
        state = self.__dict__.copy()
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        """Unpickling support: restore state and reset lock."""
        self.__dict__.update(state)
        self._lock = None

    def _reset_all_devices_off(self):
        """
        Spawns background threads to force all devices OFF during initialization.
        """
        if not self.enable:
            return

        logger.info("[Plug] Resetting all devices to OFF...")
        for ip in self.device_ips:
            threading.Thread(
                target=self._control_one_device_task,
                args=(ip, False),  # False = OFF
                daemon=True,
                name=f"Init-Reset-{ip}",
            ).start()

    def handle(self, sids, processed_images, counts, timestamp):
        """
        Core logic for triggering the plugs.
        """
        if not self.enable:
            return

        # Step 1: Determine Target States for each IP
        # Default all to False (OFF)
        target_states = dict.fromkeys(self.device_ips, False)

        # Iterate through detected streams
        for sid, count in zip(sids, counts):
            # Logic: Turn ON if count > threshold
            if count > self.alert_threshold:
                # Retrieve the IP associated with this stream ID
                if sid in self.sid_to_ip:
                    target_ip = self.sid_to_ip[sid]

                    # Verify this IP is managed by us
                    if target_ip in target_states:
                        target_states[target_ip] = True

        # Step 2: Actuate Devices (Independent Threads)
        for ip in self.device_ips:
            should_be_on = target_states[ip]

            # A. Check if IP is busy (Thread-safe)
            is_busy = False
            with self.lock:
                if ip in self.busy_ips:
                    is_busy = True

            if is_busy:
                continue

            # B. Check if state change is needed
            if self.current_states.get(ip) != should_be_on:
                threading.Thread(
                    target=self._control_one_device_task,
                    args=(ip, should_be_on),
                    daemon=True,
                    name=f"Thread-Plug-{ip}",
                ).start()

    async def _control_one_device_task(self, ip: str, turn_on: bool):
        """
        Orchestrator: Wraps the execution with timeout protection and error handling.
        """
        # Ensure timeout is defined (default to 3.0s if not set in __init__)
        try:
            # CALL CHANGE: Directly await the new separate method with a timeout
            await asyncio.wait_for(self._execute_device_command(ip, turn_on), timeout=self.timeout_seconds)

            # Update state cache upon success
            self.current_states[ip] = turn_on

            action = "ON" if turn_on else "OFF"
            logger.info(f"[Plug] Device {ip} switched {action}")

        except asyncio.TimeoutError:
            # Warning level to avoid spamming logs on network timeout
            logger.warning(f"[Plug] Timeout: Device {ip} unresponsive for {self.timeout_seconds}s.")

        except Exception as e:
            logger.warning(f"[Plug] Failed to control {ip}: {e}")

    async def _execute_device_command(self, ip: str, turn_on: bool):
        """
        Worker: Performs the actual Network I/O (Connect & Switch).
        No try-except here; let errors bubble up to the caller.
        """
        # Create fresh client for thread safety (Crucial for asyncio isolation)
        client = ApiClient(self.tapo_email, self.tapo_password)

        # Connect to device (This step is usually where timeouts happen)
        device = await client.p100(ip)

        # Execute the switch command
        if turn_on:
            await device.on()
        else:
            await device.off()
