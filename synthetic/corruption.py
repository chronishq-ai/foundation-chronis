"""Corruption injection — spec Section 6.

Applies configurable, low-probability "ugly but realistic" defects to a
freshly-generated stream's records, so the ingestion pipeline is
exercised against messy data on every run, not just clean output.

Two of the spec's 8 corruption modes are intentionally NOT applied here:
  - `participant_dropout` is implemented at the roster level
    (`synthetic.config.build_roster`), truncating a participant's
    active-day calendar. It doesn't make sense as a per-record/per-file
    transform, since it's about which days exist at all, not about
    corrupting records within a day.
  - `leaky_fixture` is not a probabilistic corruption mode. It is a
    fixed, deterministic, always-on regression record for the isolation
    test suite (spec Section 8), built separately and always included
    (not sampled) — see the isolation-suite build step.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from random import Random

from synthetic.config import CorruptionConfig
from synthetic.generators.base import Record

# Which field(s) on a stream's records represent time, and therefore
# need to move together under clock_drift. Streams not listed default
# to a single "timestamp" field.
TIME_FIELDS_BY_STREAM: dict[str, tuple[str, ...]] = {
    "fitbit.steps": ("window_start", "window_end"),
    "fitbit.sleep": ("start_time", "end_time"),
    "chest_ecg.snippet": ("start_time",),
}

DEFAULT_TIME_FIELDS: tuple[str, ...] = ("timestamp",)


def _time_fields(stream_name: str) -> tuple[str, ...]:
    return TIME_FIELDS_BY_STREAM.get(stream_name, DEFAULT_TIME_FIELDS)


def inject_corruption(
    records: list[Record],
    config: CorruptionConfig,
    rng: Random,
    *,
    stream_name: str,
) -> list[Record]:
    """Apply every applicable corruption mode to one stream's records,
    each independently gated by its own configured probability.

    Modes are applied in a fixed order so behavior is deterministic
    given the same rng state, and each operates on the *output* of the
    previous one — corruptions can compound (e.g. a malformed row could
    also end up duplicated), mirroring how real-world defects stack
    rather than occurring in isolation.
    """

    if not records:
        return records

    result = list(records)

    result = _maybe_missing_block(result, config.missing_block, rng)
    result = _maybe_duplicate_rows(result, config.duplicate_rows, rng)
    result = _maybe_out_of_order(result, config.out_of_order, rng)
    result = _maybe_clock_drift(result, config.clock_drift, rng, stream_name=stream_name)
    result = _maybe_malformed_row(result, config.malformed_row, rng)
    result = _maybe_schema_version_bump(result, config.schema_version_bump, rng)

    return result


def _maybe_missing_block(records: list[Record], probability: float, rng: Random) -> list[Record]:
    """Simulate a sensor going silent for a contiguous stretch: removes
    a contiguous run of records entirely. Never replaces them with
    zeros or interpolated values — the block is simply absent, exactly
    as a real dropout would leave nothing for the loader to see."""

    if rng.random() >= probability or len(records) < 4:
        return records

    block_fraction = rng.uniform(0.05, 0.25)
    block_length = max(1, int(len(records) * block_fraction))
    block_length = min(block_length, len(records) - 1)  # never remove everything

    start_index = rng.randint(0, len(records) - block_length)

    return records[:start_index] + records[start_index + block_length :]


def _maybe_duplicate_rows(records: list[Record], probability: float, rng: Random) -> list[Record]:
    """Simulate a retry bug on the real device: some records appear
    twice, verbatim, at arbitrary positions in the file."""

    if rng.random() >= probability:
        return records

    result = list(records)
    duplicate_count = max(1, int(len(records) * rng.uniform(0.02, 0.08)))

    for _ in range(duplicate_count):
        source_index = rng.randrange(len(records))
        insert_index = rng.randrange(len(result) + 1)
        result.insert(insert_index, dict(records[source_index]))

    return result


def _maybe_out_of_order(records: list[Record], probability: float, rng: Random) -> list[Record]:
    """Simulate records arriving with timestamps not in file order:
    swaps a handful of record positions without changing their
    contents."""

    if rng.random() >= probability or len(records) < 2:
        return records

    result = list(records)
    swap_count = max(1, int(len(records) * rng.uniform(0.02, 0.10)))

    for _ in range(swap_count):
        i = rng.randrange(len(result))
        j = rng.randrange(len(result))
        result[i], result[j] = result[j], result[i]

    return result


def _maybe_clock_drift(
    records: list[Record], probability: float, rng: Random, *, stream_name: str
) -> list[Record]:
    """Simulate one device's clock being off by a few minutes: shifts
    EVERY time field on EVERY record in this file by the same fixed
    random offset — a whole-file/whole-device drift, not per-record
    jitter."""

    if rng.random() >= probability:
        return records

    offset_minutes = rng.choice([m for m in range(-10, 11) if m != 0])
    offset = timedelta(minutes=offset_minutes)

    fields = _time_fields(stream_name)

    result = []
    for record in records:
        new_record = dict(record)
        for field_name in fields:
            raw_value = new_record.get(field_name)
            if isinstance(raw_value, str):
                try:
                    shifted = datetime.fromisoformat(raw_value) + offset
                    new_record[field_name] = shifted.isoformat()
                except ValueError:
                    pass  # not a parseable timestamp - leave untouched
        result.append(new_record)

    return result


_MALFORM_STRATEGIES = ("drop_field", "extra_column", "wrong_type")


def _maybe_malformed_row(records: list[Record], probability: float, rng: Random) -> list[Record]:
    """Simulate a row with a missing field, an extra unexpected column,
    or a bad type. Each row is independently at risk at a low rate, so
    most rows stay clean even when this mode is active."""

    result = []

    for record in records:
        if rng.random() < probability and record:
            new_record = dict(record)
            strategy = rng.choice(_MALFORM_STRATEGIES)

            if strategy == "drop_field" and len(new_record) > 1:
                key_to_drop = rng.choice(list(new_record.keys()))
                del new_record[key_to_drop]
            elif strategy == "extra_column":
                new_record["_unexpected_field"] = "unexpected_value"
            elif strategy == "wrong_type":
                key_to_corrupt = rng.choice(list(new_record.keys()))
                new_record[key_to_corrupt] = f"CORRUPT[{new_record[key_to_corrupt]!r}]"

            result.append(new_record)
        else:
            result.append(record)

    return result


def _maybe_schema_version_bump(
    records: list[Record], probability: float, rng: Random
) -> list[Record]:
    """Simulate a later batch of files shipping one renamed column:
    applied to the WHOLE file at once (not per-row), since a schema
    change affects every row in a batch together, never a scattered
    few rows within one file."""

    if rng.random() >= probability or not records:
        return records

    sample_keys = list(records[0].keys())
    candidate_keys = [k for k in sample_keys if k not in ("participant_id", "timestamp")]

    if not candidate_keys:
        return records

    renamed_key = rng.choice(candidate_keys)
    new_key = f"{renamed_key}_v2"

    result = []
    for record in records:
        new_record = dict(record)
        if renamed_key in new_record:
            new_record[new_key] = new_record.pop(renamed_key)
        result.append(new_record)

    return result
