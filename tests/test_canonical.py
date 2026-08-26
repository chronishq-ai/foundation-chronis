"""Tests for Step 7: synthetic.canonical adapters."""

from datetime import date
from pathlib import Path
from random import Random

import pytest
from synthetic.canonical import (
    CanonicalEventRecord,
    CanonicalIntervalRecord,
    CanonicalPointRecord,
    CanonicalSnippetRecord,
    MalformedRowError,
    SchemaDriftError,
    adapt_file,
)
from synthetic.config import Participant
from synthetic.registry import REGISTRY
from synthetic.writer import write_records

TEST_DAY = date(2026, 3, 1)


def make_participant() -> Participant:
    return Participant(
        participant_id="synthetic_p0001",
        enrollment_date=TEST_DAY,
        active_days=(TEST_DAY,),
        resting_heart_rate=68.0,
    )


def write_stream(
    tmp_path: Path, stream_name: str, *, seed: int = 1, group_id: str | None = None
) -> tuple[Path, str]:
    participant = make_participant()
    records = REGISTRY[stream_name].generate(participant, TEST_DAY, Random(seed))
    resolved_group_id = group_id or participant.participant_id
    path = write_records(tmp_path, stream_name, resolved_group_id, TEST_DAY.isoformat(), records)
    relative_path = str(path.relative_to(tmp_path))
    return path, relative_path


# --- Record kind separation (the core architectural rule) -------------------


def test_heart_rate_produces_point_records(tmp_path: Path) -> None:
    path, rel = write_stream(tmp_path, "fitbit.heart_rate")

    records = adapt_file(path, "fitbit.heart_rate", rel)

    assert all(isinstance(r, CanonicalPointRecord) for r in records)


def test_steps_produces_interval_records_not_point(tmp_path: Path) -> None:
    path, rel = write_stream(tmp_path, "fitbit.steps")

    records = adapt_file(path, "fitbit.steps", rel)

    assert all(isinstance(r, CanonicalIntervalRecord) for r in records)
    assert not any(isinstance(r, CanonicalPointRecord) for r in records)


def test_sleep_produces_interval_records() -> None:
    assert issubclass(CanonicalIntervalRecord, object)  # sanity import check


def test_phone_events_produces_event_records_not_point(tmp_path: Path) -> None:
    path, rel = write_stream(tmp_path, "phone_events.interaction")

    records = adapt_file(path, "phone_events.interaction", rel)

    assert all(isinstance(r, CanonicalEventRecord) for r in records)
    assert not any(isinstance(r, CanonicalPointRecord) for r in records)


def test_chest_ecg_produces_snippet_records_not_point(tmp_path: Path) -> None:
    path, rel = write_stream(tmp_path, "chest_ecg.snippet")

    records = adapt_file(path, "chest_ecg.snippet", rel)

    assert all(isinstance(r, CanonicalSnippetRecord) for r in records)
    assert not any(isinstance(r, CanonicalPointRecord) for r in records)


# --- Unit attachment -------------------------------------------------------


def test_every_point_and_interval_record_has_a_unit(tmp_path: Path) -> None:
    for stream_name in (
        "fitbit.heart_rate",
        "fitbit.steps",
        "audio_features.summary",
        "environment.device",
    ):
        path, rel = write_stream(tmp_path, stream_name, seed=hash(stream_name) % 1000)
        records = adapt_file(path, stream_name, rel)
        assert all(getattr(r, "unit", None) is not None for r in records), stream_name


# --- Source traceability -----------------------------------------------------


def test_every_record_traces_to_source_file_and_row(tmp_path: Path) -> None:
    path, rel = write_stream(tmp_path, "fitbit.heart_rate")

    records = adapt_file(path, "fitbit.heart_rate", rel)

    assert all(r.source.relative_path == rel for r in records)
    assert all(isinstance(r.source.row_index, int) for r in records)


# --- Deduplication -----------------------------------------------------------


def test_duplicate_rows_are_deduplicated(tmp_path: Path) -> None:
    participant = make_participant()
    base_records = REGISTRY["fitbit.heart_rate"].generate(participant, TEST_DAY, Random(1))

    # Manually inject an exact duplicate of the first record.
    duplicated = [base_records[0]] + base_records

    path = write_records(
        tmp_path, "fitbit.heart_rate", participant.participant_id, TEST_DAY.isoformat(), duplicated
    )
    rel = str(path.relative_to(tmp_path))

    result = adapt_file(path, "fitbit.heart_rate", rel)

    # unique timestamps should equal unique input timestamps, not the padded count
    unique_input_timestamps = {r["timestamp"] for r in base_records}
    result_timestamps = {r.timestamp.isoformat() for r in result}

    assert len(result) == len(unique_input_timestamps)
    assert len(result_timestamps) == len(unique_input_timestamps)


def test_dedup_result_is_order_independent(tmp_path: Path) -> None:
    """Feeding the same set of rows in a different order must produce
    an identical canonical result — proves dedup isn't 'first seen'."""

    participant = make_participant()
    records = REGISTRY["fitbit.heart_rate"].generate(participant, TEST_DAY, Random(1))[:5]

    forward_path = write_records(
        tmp_path / "forward",
        "fitbit.heart_rate",
        participant.participant_id,
        TEST_DAY.isoformat(),
        records,
    )
    reversed_path = write_records(
        tmp_path / "reversed",
        "fitbit.heart_rate",
        participant.participant_id,
        TEST_DAY.isoformat(),
        list(reversed(records)),
    )

    forward_result = adapt_file(forward_path, "fitbit.heart_rate", "forward.csv")
    reversed_result = adapt_file(reversed_path, "fitbit.heart_rate", "reversed.csv")

    forward_values = sorted((r.timestamp, r.value) for r in forward_result)
    reversed_values = sorted((r.timestamp, r.value) for r in reversed_result)

    assert forward_values == reversed_values


# --- Sorting despite out-of-order input --------------------------------------


def test_records_are_sorted_by_time_regardless_of_input_order(tmp_path: Path) -> None:
    participant = make_participant()
    records = REGISTRY["fitbit.heart_rate"].generate(participant, TEST_DAY, Random(1))
    shuffled = list(reversed(records))

    path = write_records(
        tmp_path, "fitbit.heart_rate", participant.participant_id, TEST_DAY.isoformat(), shuffled
    )
    rel = str(path.relative_to(tmp_path))

    result = adapt_file(path, "fitbit.heart_rate", rel)
    timestamps = [r.timestamp for r in result]

    assert timestamps == sorted(timestamps)


# --- Failure modes: schema drift vs malformed row ---------------------------


def test_missing_column_in_every_row_raises_schema_drift_error(tmp_path: Path) -> None:
    records = [
        {
            "timestamp": "2026-03-01T00:00:00+00:00",
            "participant_id": "p1",
        },  # heart_rate_bpm missing entirely
    ]
    path = write_records(tmp_path, "fitbit.heart_rate", "p1", TEST_DAY.isoformat(), records)
    rel = str(path.relative_to(tmp_path))

    with pytest.raises(SchemaDriftError, match="schema drift"):
        adapt_file(path, "fitbit.heart_rate", rel)


def test_missing_value_in_one_row_raises_malformed_row_error(tmp_path: Path) -> None:
    records = [
        {
            "timestamp": "2026-03-01T00:00:00+00:00",
            "participant_id": "p1",
            "heart_rate_bpm": "70.0",
        },
        {
            "timestamp": "2026-03-01T00:01:00+00:00",
            "participant_id": "p1",
            "heart_rate_bpm": "",
        },  # empty value
    ]
    path = write_records(tmp_path, "fitbit.heart_rate", "p1", TEST_DAY.isoformat(), records)
    rel = str(path.relative_to(tmp_path))

    with pytest.raises(MalformedRowError, match="malformed row"):
        adapt_file(path, "fitbit.heart_rate", rel)


def test_non_numeric_value_raises_malformed_row_error(tmp_path: Path) -> None:
    records = [
        {
            "timestamp": "2026-03-01T00:00:00+00:00",
            "participant_id": "p1",
            "heart_rate_bpm": "CORRUPT[70.0]",
        },
    ]
    path = write_records(tmp_path, "fitbit.heart_rate", "p1", TEST_DAY.isoformat(), records)
    rel = str(path.relative_to(tmp_path))

    with pytest.raises(MalformedRowError, match="not a valid number"):
        adapt_file(path, "fitbit.heart_rate", rel)


def test_corrupted_timestamp_value_raises_malformed_row_not_raw_exception(tmp_path: Path) -> None:
    """Regression test for a real bug found by the Step 9 stress test:
    a corrupted timestamp string (e.g. from malformed_row's wrong_type
    strategy, which can target ANY field including timestamp) must
    raise MalformedRowError, never let the underlying parser's raw
    exception type (e.g. pandas' DateParseError) propagate uncaught."""

    records = [
        {
            "timestamp": "CORRUPT['2026-03-01T00:00:00+00:00']",
            "participant_id": "p1",
            "heart_rate_bpm": "70.0",
        },
    ]
    path = write_records(tmp_path, "fitbit.heart_rate", "p1", TEST_DAY.isoformat(), records)
    rel = str(path.relative_to(tmp_path))

    with pytest.raises(MalformedRowError, match="could not be parsed as a timestamp"):
        adapt_file(path, "fitbit.heart_rate", rel)


def test_unrecognized_stream_name_fails_loudly(tmp_path: Path) -> None:
    path, rel = write_stream(tmp_path, "fitbit.heart_rate")

    with pytest.raises(Exception, match="unrecognized modality|no adapter registered"):
        adapt_file(path, "not_a_real_modality", rel)


# --- Timestamp preservation across streams -----------------------------------


def test_interval_record_preserves_distinct_start_and_end(tmp_path: Path) -> None:
    path, rel = write_stream(tmp_path, "fitbit.steps")

    records = adapt_file(path, "fitbit.steps", rel)

    assert all(r.end_time > r.start_time for r in records)


def test_snippet_record_preserves_all_samples(tmp_path: Path) -> None:
    from synthetic.generators.chest_ecg import ChestEcgGenerator

    participant = make_participant()
    small_generator = ChestEcgGenerator(duration_seconds=1.0, sampling_rate_hz=50.0)
    records = small_generator.generate(participant, TEST_DAY, Random(1))

    path = write_records(
        tmp_path, "chest_ecg.snippet", participant.participant_id, TEST_DAY.isoformat(), records
    )
    rel = str(path.relative_to(tmp_path))

    result = adapt_file(path, "chest_ecg.snippet", rel)

    assert all(len(r.samples) == r.sample_count for r in result)
    assert all(r.sample_count == 50 for r in result)  # 1.0s * 50Hz


# --- EMA dual-record behavior --------------------------------------------------


def test_ema_produces_both_event_and_point_records(tmp_path: Path) -> None:
    path, rel = write_stream(tmp_path, "surveys.ema")

    records = adapt_file(path, "surveys.ema", rel)

    event_records = [r for r in records if isinstance(r, CanonicalEventRecord)]
    point_records = [r for r in records if isinstance(r, CanonicalPointRecord)]

    assert event_records
    assert point_records
    assert all(r.feature_name == "ema_stress_score" for r in point_records)
