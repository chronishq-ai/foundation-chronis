"""
Append-only JSON logging of every request/response pair, as required by
the Pod B spec ("Save every request and response into a log file").
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional

_lock = threading.Lock()


class EventLogger:
    def __init__(self, log_path: str):
        self._path = log_path
        directory = os.path.dirname(self._path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self._path):
            with open(self._path, "w") as f:
                json.dump([], f)

    def log(
        self,
        event: str,
        raw_response: str,
        parsed: Optional[dict],
        error: Optional[str] = None,
    ) -> None:
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "raw_response": raw_response,
            "parsed": parsed,
            "error": error,
        }
        with _lock:
            with open(self._path, "r+") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
                data.append(entry)
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
