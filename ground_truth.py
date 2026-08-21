"""
Human ground-truth collection for Pod B v0.2 calibration validation.

Per spec section 4: for each event, a human (not the model) rates how much
that event should change a given state variable, blind to what the LLM or
baseline predicted. This file only handles collecting and storing that
judgment - it never imports analyzer.py or baseline.py, so there's no way
for a rater's own terminal session to accidentally leak a model prediction
before they rate.

Storage mirrors logger.py's append-only JSON pattern (same read-modify-
write-under-lock shape) but is a separate file/schema, since this is a
fundamentally different kind of record (human judgment, not model I/O).
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional, TypedDict

from schemas import VALID_VARIABLES

_GROUND_TRUTH_PATH = "logs/ground_truth.json"
_lock = threading.Lock()


class GroundTruthRating(TypedDict):
    event_id: str
    variable: str
    human_rating: float  # 0-1, normalized same scale as model confidence
    rater_id: str
    timestamp: str


def _ensure_store(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump([], f)


def record_rating(
    event_id: str,
    variable: str,
    human_rating: float,
    rater_id: str,
    path: Optional[str] = None,
) -> None:
    """
    Append one blind human rating. Not idempotent by design - re-rating the
    same (event_id, variable) by a different rater is valid data, not a
    duplicate to be overwritten. Same append-only philosophy as logger.py
    and Pod C's events table.
    """
    path = path or _GROUND_TRUTH_PATH
    if variable.lower() not in VALID_VARIABLES:
        raise ValueError(f"Unknown variable: {variable!r}")
    if not (0.0 <= human_rating <= 1.0):
        raise ValueError(f"human_rating must be 0-1, got {human_rating}")

    entry: GroundTruthRating = {
        "event_id": event_id,
        "variable": variable.lower(),
        "human_rating": human_rating,
        "rater_id": rater_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _ensure_store(path)
    with _lock:
        with open(path, "r+") as f:
            try:
                data: list[dict[str, Any]] = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()


def load_ratings(path: Optional[str] = None) -> list[GroundTruthRating]:
    path = path or _GROUND_TRUTH_PATH
    _ensure_store(path)
    with open(path, "r") as f:
        return json.load(f)


def get_rating(
    event_id: str, variable: str, path: Optional[str] = None
) -> Optional[float]:
    """
    Latest human rating for (event_id, variable), or None if never rated.
    If multiple raters rated the same pair, returns the most recent -
    evaluate.py can call load_ratings() directly for full multi-rater detail.
    """
    path = path or _GROUND_TRUTH_PATH
    matches = [
        r for r in load_ratings(path)
        if r["event_id"] == event_id and r["variable"] == variable.lower()
    ]
    if not matches:
        return None
    return matches[-1]["human_rating"]


def _collect_cli() -> None:
    """
    Minimal terminal collector: walks the dataset, asks a rater to score
    each (event, variable) pair blind - only the sentence and variable name
    are shown, never a model or baseline prediction.
    """
    from dataset import DATASET

    print("Blind ground-truth collection.")
    print("For each event + variable, rate 0-1: how much should this event")
    print("change that variable? 0 = not at all, 1 = extremely.")
    print("Press Enter with no input to skip a variable as 'not applicable'.\n")

    rater_id = input("Rater name/id: ").strip() or "anonymous"

    for entry in DATASET:
        print(f"\nEvent [{entry['id']}]: {entry['text']}")
        for variable in sorted(VALID_VARIABLES):
            raw = input(f"  {variable} (0-1, blank=skip): ").strip()
            if not raw:
                continue
            try:
                rating = float(raw)
                record_rating(entry["id"], variable, rating, rater_id)
            except ValueError as exc:
                print(f"  skipped, invalid input: {exc}")

    print("\nDone. Ratings saved to", _GROUND_TRUTH_PATH)


if __name__ == "__main__":
    _collect_cli()