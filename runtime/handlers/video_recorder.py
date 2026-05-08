import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..audit import AuditLog
from ..config import Config
from ..memory_manager import SharedMemoryClient
from .base import BaseHandler, BatchInferenceResult, InferenceResult


logger = logging.getLogger(__name__)


class VideoRecorderHandler(BaseHandler):
    """
    Continuously records selected streams to disk as fixed-length segments.

    Recording state is per-stream: the GUI flips individual streams on/off and
    the master button is just `enable() = enable every known stream`. The initial
    state for every stream comes from config.envs.enable_video_recorder.

    Each enabled stream owns its own cv2.VideoWriter; once a segment exceeds
    `segment_seconds` of wall time, the writer is released and a new file is
    opened automatically. Disk I/O happens synchronously inside handle() — the
    same place the SHM frame view is still valid (ResultProcess only releases
    the buffer after the handler chain returns), so no copy is required.
    """

    FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

    def __init__(self, config: Config, name: str = "VideoRecorder"):
        super().__init__(name=name, needs_frames=True)
        self.fps = float(config.fps)
        self.frame_size = (config.shm.width, config.shm.height)  # (W, H)
        self.segment_seconds = float(getattr(config.envs, "video_segment_seconds", 300.0))
        self.output_dir = Path(getattr(config.envs, "video_record_dir", "recordings"))

        self._all_stream_ids: set[int] = set(config.sid_to_ip.keys())
        initial_on = bool(getattr(config.envs, "enable_video_recorder", False))
        self._enabled_streams: set[int] = set(self._all_stream_ids) if initial_on else set()

        # Threading primitives MUST NOT exist before pickling — _thread.lock
        # isn't picklable, which breaks spawn-based mp on Windows. Lazy-create
        # in the child process on first access.
        self._lock_obj: Optional[threading.Lock] = None
        self._writers: dict[int, cv2.VideoWriter] = {}
        self._segment_start: dict[int, float] = {}

        self._audit_log_path = config.envs.audit_log_path
        self.audit: Optional[AuditLog] = None

    @property
    def _lock(self) -> threading.Lock:
        if self._lock_obj is None:
            self._lock_obj = threading.Lock()
        return self._lock_obj

    def start(self):
        super().start()
        self.audit = AuditLog(self._audit_log_path, name=self.name)
        if self.audit:
            self.audit.log(
                "handler.start",
                handler=self.name,
                initial_streams=sorted(self._enabled_streams),
                output_dir=str(self.output_dir),
                segment_seconds=self.segment_seconds,
            )

    def stop(self):
        with self._lock:
            self._close_all_writers_locked()
        if self.audit:
            self.audit.log("handler.stop", handler=self.name)
            self.audit.close()

    # ------------------------------------------------------------------
    # Per-stream control (driven by per-stream GUI buttons)
    # ------------------------------------------------------------------

    def enable_stream(self, sid: int) -> bool:
        if sid in self._enabled_streams:
            return True
        self._enabled_streams.add(sid)
        if self.audit:
            self.audit.log("recorder.stream_enable", stream_id=sid)
        logger.info(f"[{self.name}] Recording ON for stream {sid}.")
        return True

    def disable_stream(self, sid: int) -> bool:
        if sid not in self._enabled_streams:
            return False
        self._enabled_streams.discard(sid)
        with self._lock:
            writer = self._writers.pop(sid, None)
            self._segment_start.pop(sid, None)
        if writer is not None:
            self._release_writer(sid, writer)
        if self.audit:
            self.audit.log("recorder.stream_disable", stream_id=sid)
        logger.info(f"[{self.name}] Recording OFF for stream {sid}.")
        return False

    def toggle_stream(self, sid: int) -> bool:
        if sid in self._enabled_streams:
            self.disable_stream(sid)
            return False
        self.enable_stream(sid)
        return True

    def is_stream_enabled(self, sid: int) -> bool:
        return sid in self._enabled_streams

    # ------------------------------------------------------------------
    # Aggregate control (driven by the master "RECORDING ALL" button)
    # ------------------------------------------------------------------

    def enable(self) -> bool:
        for sid in self._all_stream_ids:
            self.enable_stream(sid)
        return True

    def disable(self) -> bool:
        if not self._enabled_streams:
            return False
        for sid in list(self._enabled_streams):
            self.disable_stream(sid)
        return False

    def toggle(self) -> bool:
        """If anything is recording, stop everything; otherwise start everything."""
        if self._enabled_streams:
            self.disable()
            return False
        self.enable()
        return True

    def is_enabled(self) -> bool:
        """True iff at least one stream is currently being recorded."""
        return bool(self._enabled_streams)

    # ------------------------------------------------------------------
    # Frame ingest
    # ------------------------------------------------------------------

    def handle_batch(self, batch_result: BatchInferenceResult, shm_client: SharedMemoryClient) -> set[tuple[int, int]]:
        # Skip the SHM frame indexing entirely when nothing is being recorded.
        if not self._enabled_streams:
            return set()
        return super().handle_batch(batch_result, shm_client)

    def handle(self, result: InferenceResult, frame: Optional[np.ndarray]):
        sid = result.stream_id
        if sid not in self._enabled_streams or frame is None:
            return
        now = time.time()
        with self._lock:
            writer = self._writers.get(sid)
            seg_start = self._segment_start.get(sid, 0.0)
            if writer is None or (now - seg_start) >= self.segment_seconds:
                if writer is not None:
                    self._release_writer(sid, writer)
                writer = self._open_new_writer(sid, now)
                if writer is None:
                    return
                self._writers[sid] = writer
                self._segment_start[sid] = now
            try:
                writer.write(frame)
            except Exception as e:
                logger.error(f"[{self.name}] write failed on stream {sid}: {e}")

    # ------------------------------------------------------------------
    # Writer lifecycle
    # ------------------------------------------------------------------

    def _open_new_writer(self, sid: int, now: float) -> Optional[cv2.VideoWriter]:
        # Per-stream subfolder so all segments for a given camera live together,
        # e.g. recordings/stream_00/20260508_123456.mp4.
        stream_dir = self.output_dir / f"stream_{sid:02d}"
        try:
            stream_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"[{self.name}] Failed to create output dir {stream_dir}: {e}")
            return None

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        path = stream_dir / f"{ts}.mp4"
        writer = cv2.VideoWriter(str(path), self.FOURCC, self.fps, self.frame_size)
        if not writer.isOpened():
            logger.error(f"[{self.name}] Failed to open writer for {path}")
            return None
        logger.info(f"[{self.name}] Recording stream {sid} -> {path}")
        if self.audit:
            self.audit.log("recorder.segment_open", stream_id=sid, path=str(path))
        return writer

    def _release_writer(self, sid: int, writer: cv2.VideoWriter):
        try:
            writer.release()
        except Exception as e:
            logger.error(f"[{self.name}] Error releasing writer for stream {sid}: {e}")
            return
        if self.audit:
            self.audit.log("recorder.segment_close", stream_id=sid)

    def _close_all_writers_locked(self):
        for sid, writer in list(self._writers.items()):
            self._release_writer(sid, writer)
        self._writers.clear()
        self._segment_start.clear()
