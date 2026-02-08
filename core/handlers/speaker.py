import logging
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..config import Config
from ..inference_process import InferenceResult
from .base import BaseHandler, BatchInferenceResult


logger = logging.getLogger(__name__)


class SpeakerHandler(BaseHandler):
    """
    Audio Deterrent Handler (Zone Compatible).
    Trigger logic: Zone Threshold Violation -> Check Cooldown -> CURL Trigger.
    """

    def __init__(self, config: Config):
        self.enable = config.envs.enable_speaker
        if not self.enable:
            return

        self.cooldown_seconds = 5

        # --- 1. 构建映射: Stream ID -> {IP, Threshold} ---
        self.stream_map = {}

        # 收集所有涉及的 IP 用于状态管理
        unique_ips = set()

        for zone in getattr(config, "zones", []):
            # 检查该 Zone 是否启用了 Speaker
            if not getattr(zone, "speakers", False):
                continue

            ip = getattr(zone, "speaker_ip", None)
            if not ip:
                continue

            unique_ips.add(ip)

            for sid in getattr(zone, "cameras", []):
                self.stream_map[sid] = {"ip": ip, "threshold": zone.threshold}

        if not self.stream_map:
            logger.warning("[Speaker] Enabled but no speakers configured in Zones.")
            self.enable = False
            return

        logger.info(f"[Speaker] Configured {len(unique_ips)} speakers for {len(self.stream_map)} streams.")

        # --- 2. 状态管理 (Debouncing) ---
        # 记录每个 IP 上次触发的时间
        self.last_trigger_times = dict.fromkeys(unique_ips, 0.0)
        self.state_lock = threading.Lock()

        # --- 3. 工作线程池 ---
        # 根据 Speaker 数量分配线程，防止单个网络请求阻塞所有设备
        self.executor = ThreadPoolExecutor(max_workers=max(1, len(unique_ips)), thread_name_prefix="SpeakerWorker")

    def handle_batch(self, batch_result: BatchInferenceResult, shm_client=None):
        """
        处理一批结果。Speaker 不需要图像数据，所以忽略 shm_client。
        """
        if not self.enable:
            return

        # 找出本批次需要触发的 IP 集合 (避免同一批次对同一 IP 重复提交任务)
        trigger_tasks = {}  # Key: IP, Value: StreamID (for logging)

        current_time = time.time()

        for res in batch_result.results:
            sid = res.stream_id

            if sid not in self.stream_map:
                continue

            zone_info = self.stream_map[sid]
            target_ip = zone_info["ip"]
            threshold = zone_info["threshold"]

            if res.count < threshold:
                continue

            last_time = self.last_trigger_times.get(target_ip, 0.0)
            if current_time - last_time < self.cooldown_seconds:
                continue

            # 记录待触发任务 (同批次去重，取第一个触发的 SID 即可)
            if target_ip not in trigger_tasks:
                trigger_tasks[target_ip] = sid

        # 4. 执行触发 (带锁更新时间状态)
        for ip, sid in trigger_tasks.items():
            should_fire = False
            with self.state_lock:
                # 二次检查 (Double-check locking)
                last_time = self.last_trigger_times.get(ip, 0.0)
                if current_time - last_time > self.cooldown_seconds:
                    self.last_trigger_times[ip] = current_time
                    should_fire = True

            if should_fire:
                self.executor.submit(self._trigger_curl, ip, sid)

    def handle(self, result: InferenceResult, frame: np.ndarray):
        pass

    def _trigger_curl(self, ip: str, sid: int):
        """执行 CURL 命令调用远程 API"""
        # API 格式: http://{ip}/cgi-bin/audio_play?name=7MB.wav&action=start&time=1
        url = f"http://{ip}/cgi-bin/audio_play?name=7MB.wav&action=start&time=1"

        cmd = [
            "curl",
            "-s",  # Silent mode
            "-u",
            "admin:admin",  # 假设所有设备使用默认密码，如需不同密码需在 zone config 中扩展
            "--max-time",
            "5",  # 硬超时，防止卡死 Worker 线程
            url,
        ]

        try:
            logger.info(f"[Speaker] Alerting Zone/Cam {sid} (IP: {ip})...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                logger.error(f"[Speaker] Curl failed for {ip}: {result.stderr}")
            else:
                # logger.debug(f"[Speaker] Trigger success: {ip}")
                pass

        except Exception as e:
            logger.error(f"[Speaker] Exception triggering {ip}: {e}")

    def stop(self):
        if self.enable:
            self.executor.shutdown(wait=False)
