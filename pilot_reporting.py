"""Read-only pilot coverage and Demo 2 evidence reporting."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.update_state import update_state
from event_store import initialize_database

STARTING_STATE = {
    "mood": 5.0, "focus": 5.0, "stress": 5.0, "confidence": 5.0,
    "trust": 5.0, "motivation": 5.0, "social_engagement": 5.0,
}


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_roster(path: str | Path) -> list[dict[str, str]]:
    """Parse operational roster rows without accepting names as an identity contract."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] in {"Pilot ID", "---"} or not cells[0]:
            continue
        if cells[1] not in {"PENDING", "ACTIVE", "COMPLETE"}:
            continue
        rows.append({"pilot_id": cells[0], "status": cells[1]})
    return rows


def _load_records(db_path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    initialize_database(db_path)
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT id, description, happened_at, change_data, pilot_id, input_id, data_kind FROM events ORDER BY id"
        ).fetchall()
    finally:
        connection.close()
    records, malformed = [], []
    for row in rows:
        record = dict(row)
        errors = []
        try:
            record["timestamp"] = _parse_timestamp(record["happened_at"])
        except (TypeError, ValueError):
            errors.append("invalid_timestamp")
        try:
            record["signal"] = json.loads(record["change_data"])
            if not isinstance(record["signal"], dict):
                errors.append("invalid_signal")
        except (TypeError, json.JSONDecodeError):
            errors.append("malformed_signal")
        if record["data_kind"] == "real" and not record["pilot_id"]:
            errors.append("missing_pilot_id")
        if record["data_kind"] == "real" and not record["input_id"]:
            errors.append("missing_input_id")
        if errors:
            malformed.append({"event_id": record["id"], "errors": errors})
        records.append(record)
    return records, malformed


def build_report(db_path: str | Path, *, start: date | None = None) -> dict[str, Any]:
    records, malformed = _load_records(db_path)
    real = [r for r in records if r["data_kind"] == "real" and r["pilot_id"]]
    synthetic = [r for r in records if r["data_kind"] == "synthetic"]
    unattributed = [r for r in records if r["data_kind"] == "unattributed"]
    pilot_ids = sorted({record["pilot_id"] for record in real})
    dated_real = [record for record in real if "timestamp" in record]
    if start is None and dated_real:
        start = min(r["timestamp"].date() for r in dated_real)

    coverage: dict[str, dict[str, Any]] = {}
    per_day: dict[str, Counter[str]] = defaultdict(Counter)
    for record in real:
        if "timestamp" in record:
            per_day[record["pilot_id"]][record["timestamp"].date().isoformat()] += 1
    for pilot_id in pilot_ids:
        days = per_day[pilot_id]
        expected = [] if start is None else [(start + timedelta(days=offset)).isoformat() for offset in range(4)]
        present = [day for day in expected if days[day] > 0]
        missing = [day for day in expected if days[day] == 0]
        status = "COMPLETE" if len(present) == 4 else "PARTIAL" if present else "MISSING"
        coverage[pilot_id] = {"status": status, "events_by_day": dict(days), "missing_days": missing}

    duplicate_index: dict[tuple[str, str], list[int]] = defaultdict(list)
    for record in real:
        if record["input_id"]:
            duplicate_index[(record["pilot_id"], record["input_id"])].append(record["id"])
    duplicates = [
        {"pilot_id": pilot_id, "input_id": input_id, "event_ids": event_ids}
        for (pilot_id, input_id), event_ids in duplicate_index.items() if len(event_ids) > 1
    ]

    candidates = _candidate_patterns(real)
    return {
        "classification": {
            "real_pilot_events": len(real), "synthetic_events": len(synthetic),
            "unattributed_legacy_events": len(unattributed),
        },
        "real_pilots": pilot_ids,
        "coverage_start": start.isoformat() if start else None,
        "coverage": coverage,
        "malformed_events": malformed,
        "duplicates": duplicates,
        "candidate_patterns": candidates,
        "demo2_status": "QUALIFYING_PATTERN_FOUND" if candidates else "NO_QUALIFYING_PATTERN",
    }


def _candidate_patterns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pilot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if "timestamp" in record and "signal" in record:
            by_pilot[record["pilot_id"]].append(record)
    candidates = []
    for pilot_id, events in by_pilot.items():
        events.sort(key=lambda item: (item["timestamp"], item["id"]))
        for index, pivot in enumerate(events[:-1]):
            try:
                then_state = dict(STARTING_STATE)
                for event in events[:index + 1]:
                    then_state = update_state(then_state, event["signal"])
                now_state = dict(then_state)
                for event in events[index + 1:]:
                    now_state = update_state(now_state, event["signal"])
            except (TypeError, ValueError, KeyError):
                continue
            divergence = round(sum(abs(now_state[key] - then_state[key]) for key in STARTING_STATE), 2)
            if divergence > 0.5:
                candidates.append({
                    "pilot_id": pilot_id,
                    "earlier_observation_ref": f"event-{pivot['id']}",
                    "later_observation_refs": [f"event-{event['id']}" for event in events[index + 1:]],
                    "earlier_timestamp": pivot["timestamp"].isoformat(),
                    "later_timestamps": [event["timestamp"].isoformat() for event in events[index + 1:]],
                    "pattern_type": "then_now_state_divergence",
                    "divergence": divergence,
                    "pipeline_output": {"then": then_state, "now": now_state},
                })
    return candidates
