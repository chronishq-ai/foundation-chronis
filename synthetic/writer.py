"""Writes a stream's generated records to disk under the synthetic_root
folder layout (spec Section 3), so `tiles_participant_index.py` has real
files to discover and probe.

This is intentionally minimal — full dataset orchestration
(`generate_dataset()`, `_generator_manifest.json` at the dataset root)
is a later build step. This module only knows how to write ONE stream's
records for one participant/device + day.
"""

from __future__ import annotations

import csv
from pathlib import Path

from synthetic.generators.base import Record

# Where each registered stream's files live under synthetic_root,
# relative to the root. Data-driven — used by both the writer and the
# indexer's `infer_stream`, so the two can never silently drift apart.
STREAM_PATH_PREFIXES: dict[str, str] = {
    "fitbit.heart_rate": "fitbit/heart_rate",
    "fitbit.steps": "fitbit/steps",
    "fitbit.sleep": "fitbit/sleep",
    "chest_ecg.snippet": "chest_ecg",
    "audio_features.summary": "audio_features",
    "proximity.participant_beacon": "proximity/participant_beacon",
    "environment.device": "environment",
    "phone_events.interaction": "phone_events",
    "surveys.ema": "surveys",
}

# File granularity per stream (spec Section 3's granularity column).
# "participant" and "device" streams are written as one file per
# group_id per day; a future single-file or snippet-level split mode
# would need a different writer path, not yet built.
GRANULARITY_BY_STREAM: dict[str, str] = {
    "fitbit.heart_rate": "participant",
    "fitbit.steps": "participant",
    "fitbit.sleep": "participant",
    "chest_ecg.snippet": "participant",
    "audio_features.summary": "participant",
    "proximity.participant_beacon": "participant",
    "environment.device": "device",
    "phone_events.interaction": "participant",
    "surveys.ema": "participant",
}


def write_records(
    root: Path,
    stream_name: str,
    group_id: str,
    day_isoformat: str,
    records: list[Record],
) -> Path:
    """Write one stream's records for one participant/device + day as a
    CSV file, creating parent directories as needed.

    Column order is deterministic: the union of every record's keys,
    sorted alphabetically — necessary because corruption injection
    (`malformed_row`, `schema_version_bump`) can make different records
    within the same file have different keys (extra/renamed/dropped
    columns), so a naive "use the first record's keys" approach would
    silently drop or misalign data.
    """

    if stream_name not in STREAM_PATH_PREFIXES:
        raise ValueError(f"unknown stream_name {stream_name!r}; not in STREAM_PATH_PREFIXES")

    stream_dir = root / STREAM_PATH_PREFIXES[stream_name] / group_id
    stream_dir.mkdir(parents=True, exist_ok=True)

    file_path = stream_dir / f"{day_isoformat}.csv"

    all_columns: list[str] = []
    seen = set()
    for record in records:
        for key in record.keys():
            if key not in seen:
                seen.add(key)
                all_columns.append(key)
    all_columns.sort()

    with file_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_columns, restval="")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    return file_path
