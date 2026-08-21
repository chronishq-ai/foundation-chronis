"""Privacy-minimised structured observability for the Pod E pipeline."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LOG_PATH = Path("logs/integration-events.jsonl")


def input_reference(text: str) -> str:
    """Stable input reference without copying a pilot's raw wording into logs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def log_event(*, pilot_id: str | None, input_id: str | None, component: str,
              event_type: str, status: str, stage: str, message: str,
              error_code: str | None = None, trace_id: str | None = None,
              text: str | None = None, event_id: int | None = None,
              path: Path = DEFAULT_LOG_PATH) -> str:
    trace_id = trace_id or str(uuid.uuid4())
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pilot_id": pilot_id or "unattributed",
        "input_id": input_id or input_reference(text or ""),
        "event_id": event_id,
        "component": component,
        "event_type": event_type,
        "status": status,
        "error_code": error_code,
        "message": message,
        "trace_id": trace_id,
        "pipeline_stage": stage,
        "input_reference": input_reference(text or ""),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")
    return trace_id
