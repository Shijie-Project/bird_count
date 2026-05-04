import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


class AuditLog:
    """
    Append-only JSONL audit log.

    Thread-safe within a process; cross-process append is also safe because
    each emitted record fits well under PIPE_BUF on every supported platform
    and the file is opened in O_APPEND mode (each write atomically seeks-end).

    Disabled if `path` is falsy — log() then becomes a cheap no-op.
    """

    def __init__(self, path: Optional[str], name: str = "audit"):
        self.name = name
        self.enabled: bool = bool(path)
        self.path: Optional[Path] = Path(path) if path else None
        self._lock = threading.Lock()
        self._fh = None

        if not self.enabled:
            return

        try:
            assert self.path is not None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
            logger.info(f"[{self.name}] Audit log opened: {self.path}")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to open audit log at {self.path}: {e}")
            self.enabled = False
            self._fh = None

    def log(self, event: str, **fields: Any) -> None:
        if not self.enabled or self._fh is None:
            return
        record = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "pid": os.getpid(),
            "event": event,
            **fields,
        }
        try:
            line = json.dumps(record, default=str)
            with self._lock:
                self._fh.write(line + "\n")
        except Exception as e:
            logger.debug(f"[{self.name}] audit write failed: {e}")

    def close(self) -> None:
        if self._fh is not None:
            try:
                with self._lock:
                    self._fh.close()
            except Exception:
                pass
            self._fh = None
