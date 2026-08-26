"""Tests for Step 4: corruption injection (spec Section 6)."""

from datetime import datetime
from random import Random

from synthetic.config import CorruptionConfig
from synthetic.corruption import inject_corruption


def make_records(count: int = 50) -> list[dict[str, object]]:
    return [
        {
            "timestamp": f"2026-03-01T{hour:02d}:00:00+00:00",
            "participant_id": "synthetic_p0001",
            "heart_rate_bpm": 70.0 + index,
        }
        for index, hour in zip(range(count), (h % 24 for h in range(count)), strict=False)
    ]


def zero_config(**overrides: float) -> CorruptionConfig:
    base = {
        "missing_block": 0.0,
        "duplicate_rows": 0.0,
        "out_of_order": 0.0,
        "clock_drift": 0.0,
        "malformed_row": 0.0,
        "participant_dropout": 0.0,
        "schema_version_bump": 0.0,
    }
    base.update(overrides)
    return CorruptionConfig(**base)  # type: ignore[arg-type]


# --- No-op baseline -----------------------------------------------------


def test_zero_probabilities_leave_records_unchanged() -> None:
    records = make_records()
    config = zero_config()

    result = inject_corruption(records, config, Random(1), stream_name="fitbit.heart_rate")

    assert result == records


def test_empty_records_returns_empty() -> None:
    config = CorruptionConfig()

    result = inject_corruption([], config, Random(1), stream_name="fitbit.heart_rate")

    assert result == []


# --- missing_block --------------------------------------------------------


def test_missing_block_removes_a_contiguous_run_when_forced() -> None:
    records = make_records(50)
    config = zero_config(missing_block=1.0)

    result = inject_corruption(records, config, Random(1), stream_name="fitbit.heart_rate")

    assert len(result) < len(records)
    # remaining records must still be a subsequence (order preserved, contiguous chunk gone)
    result_values = [r["heart_rate_bpm"] for r in result]
    original_values = [r["heart_rate_bpm"] for r in records]
    assert result_values == [v for v in original_values if v in set(result_values)]


def test_missing_block_never_removes_everything() -> None:
    records = make_records(10)
    config = zero_config(missing_block=1.0)

    result = inject_corruption(records, config, Random(3), stream_name="fitbit.heart_rate")

    assert len(result) >= 1


def test_missing_block_does_not_fire_at_zero_probability() -> None:
    records = make_records(50)
    config = zero_config(missing_block=0.0)

    result = inject_corruption(records, config, Random(1), stream_name="fitbit.heart_rate")

    assert len(result) == len(records)


# --- duplicate_rows --------------------------------------------------------


def test_duplicate_rows_increases_record_count_when_forced() -> None:
    records = make_records(50)
    config = zero_config(duplicate_rows=1.0)

    result = inject_corruption(records, config, Random(1), stream_name="fitbit.heart_rate")

    assert len(result) > len(records)


def test_duplicate_rows_are_exact_copies() -> None:
    records = make_records(20)
    config = zero_config(duplicate_rows=1.0)

    result = inject_corruption(records, config, Random(2), stream_name="fitbit.heart_rate")

    # every original record's content must still appear at least once
    original_signatures = [tuple(sorted(r.items())) for r in records]
    result_signatures = [tuple(sorted(r.items())) for r in result]
    for sig in original_signatures:
        assert sig in result_signatures


# --- out_of_order -----------------------------------------------------------


def test_out_of_order_changes_position_without_changing_content() -> None:
    records = make_records(50)
    config = zero_config(out_of_order=1.0)

    result = inject_corruption(records, config, Random(1), stream_name="fitbit.heart_rate")

    assert len(result) == len(records)  # same content, different arrangement
    assert sorted(r["heart_rate_bpm"] for r in result) == sorted(
        r["heart_rate_bpm"] for r in records
    )
    assert [r["heart_rate_bpm"] for r in result] != [r["heart_rate_bpm"] for r in records]


# --- clock_drift --------------------------------------------------------------


def test_clock_drift_shifts_every_timestamp_by_the_same_offset() -> None:
    records = make_records(10)
    config = zero_config(clock_drift=1.0)

    result = inject_corruption(records, config, Random(5), stream_name="fitbit.heart_rate")

    original_times = [datetime.fromisoformat(str(r["timestamp"])) for r in records]
    shifted_times = [datetime.fromisoformat(str(r["timestamp"])) for r in result]

    deltas = {
        (shifted - original)
        for original, shifted in zip(original_times, shifted_times, strict=False)
    }

    assert len(deltas) == 1  # exactly one uniform offset applied
    offset = next(iter(deltas))
    assert offset.total_seconds() != 0


def test_clock_drift_moves_multi_field_streams_together() -> None:
    """fitbit.steps has two time fields (window_start, window_end) -
    both must shift by the identical offset, preserving the window's
    internal consistency."""

    records = [
        {
            "window_start": "2026-03-01T09:00:00+00:00",
            "window_end": "2026-03-01T09:01:00+00:00",
            "participant_id": "synthetic_p0001",
            "step_count": 10,
        }
    ]
    config = zero_config(clock_drift=1.0)

    result = inject_corruption(records, config, Random(5), stream_name="fitbit.steps")

    start = datetime.fromisoformat(str(result[0]["window_start"]))
    end = datetime.fromisoformat(str(result[0]["window_end"]))

    assert (end - start).total_seconds() == 60.0  # window duration preserved


# --- malformed_row -----------------------------------------------------------


def test_malformed_row_alters_some_records_when_forced() -> None:
    records = make_records(50)
    config = zero_config(malformed_row=1.0)

    result = inject_corruption(records, config, Random(1), stream_name="fitbit.heart_rate")

    assert len(result) == len(records)
    # at 100% probability every record should have SOME structural change
    changes = sum(
        1 for original, mutated in zip(records, result, strict=False) if original != mutated
    )
    assert changes == len(records)


def test_malformed_row_does_not_fire_at_zero_probability() -> None:
    records = make_records(50)
    config = zero_config(malformed_row=0.0)

    result = inject_corruption(records, config, Random(1), stream_name="fitbit.heart_rate")

    assert result == records


# --- schema_version_bump -----------------------------------------------------


def test_schema_version_bump_renames_a_column_across_all_records() -> None:
    records = make_records(20)
    config = zero_config(schema_version_bump=1.0)

    result = inject_corruption(records, config, Random(9), stream_name="fitbit.heart_rate")

    # participant_id/timestamp are protected from renaming; heart_rate_bpm
    # is the only other candidate key in this fixture, so it must be the
    # one renamed.
    assert all("heart_rate_bpm" not in r for r in result)
    assert all("heart_rate_bpm_v2" in r for r in result)
    assert all("participant_id" in r for r in result)  # protected field untouched
    assert all("timestamp" in r for r in result)  # protected field untouched


def test_schema_version_bump_applies_uniformly_to_whole_batch() -> None:
    """A schema bump must never affect only some rows in a file - it's
    a whole-file change."""

    records = make_records(20)
    config = zero_config(schema_version_bump=1.0)

    result = inject_corruption(records, config, Random(9), stream_name="fitbit.heart_rate")

    renamed_key_present = ["heart_rate_bpm_v2" in r for r in result]
    assert all(renamed_key_present)  # uniform, not partial


# --- compounding --------------------------------------------------------------


def test_multiple_modes_can_compound() -> None:
    records = make_records(50)
    config = CorruptionConfig(
        missing_block=1.0,
        duplicate_rows=1.0,
        out_of_order=0.0,
        clock_drift=0.0,
        malformed_row=0.0,
        participant_dropout=0.0,
        schema_version_bump=0.0,
    )

    result = inject_corruption(records, config, Random(1), stream_name="fitbit.heart_rate")

    # missing_block shrinks, then duplicate_rows grows from the shrunk set -
    # net effect just needs to differ from the original in some way.
    assert result != records
