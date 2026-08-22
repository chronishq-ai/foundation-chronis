"""Tests for the TILES-2018 loader.

NOTE: the column names used in these fixtures (e.g. "date", "heart_rate")
are placeholders for testing the loader's mechanics. They are NOT
confirmed against the real TILES-2018 release manifest. Once the real
column names are confirmed, update TilesColumnMapping usage here (and
in production config) to match — the loader itself does not need to
change, since it takes the mapping as an explicit parameter.
"""

from pathlib import Path

import pandas as pd
import pytest

from chronis_ml.loaders.base import LoaderConfig
from chronis_ml.loaders.tiles import TilesColumnMapping, TilesLoader
from chronis_ml.schema.models import MeasurementStatus, MissingReason


def default_mapping() -> TilesColumnMapping:
    return TilesColumnMapping(
        timestamp_column="date",
        feature_columns=("heart_rate", "steps"),
        modality="fitbit",
    )


def test_tiles_missing_path_fails(tmp_path: Path) -> None:
    loader = TilesLoader(column_mapping=default_mapping())

    with pytest.raises(FileNotFoundError):
        loader.load(LoaderConfig(source_path=tmp_path / "missing"))


def test_tiles_loads_per_participant_files(tmp_path: Path) -> None:
    """Participant ID is derived from the filename, matching TILES-2018's
    per-participant-file layout (no participant_id_column configured)."""

    frame = pd.DataFrame(
        {
            "date": ["2026-08-16", "2026-08-17"],
            "heart_rate": [70.0, 72.0],
            "steps": [1000.0, 1500.0],
        }
    )
    frame.to_csv(tmp_path / "participant_001.csv", index=False)

    dataset = TilesLoader(column_mapping=default_mapping()).load(LoaderConfig(source_path=tmp_path))

    assert dataset.users == ("participant_001",)
    assert len(dataset.records) == 4
    assert all(record.status is MeasurementStatus.OBSERVED for record in dataset.records)


def test_tiles_handles_multiple_participants(tmp_path: Path) -> None:
    pd.DataFrame({"date": ["2026-08-16"], "heart_rate": [70.0], "steps": [1000.0]}).to_csv(
        tmp_path / "participant_001.csv", index=False
    )

    pd.DataFrame({"date": ["2026-08-16"], "heart_rate": [65.0], "steps": [800.0]}).to_csv(
        tmp_path / "participant_002.csv", index=False
    )

    dataset = TilesLoader(column_mapping=default_mapping()).load(LoaderConfig(source_path=tmp_path))

    assert dataset.users == ("participant_001", "participant_002")


def test_tiles_preserves_explicit_missing_value(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-08-16", "2026-08-17"],
            "heart_rate": [70.0, None],
            "steps": [1000.0, 1200.0],
        }
    )
    frame.to_csv(tmp_path / "participant_001.csv", index=False)

    dataset = TilesLoader(column_mapping=default_mapping()).load(LoaderConfig(source_path=tmp_path))

    missing = [
        record
        for record in dataset.records
        if record.feature_name == "heart_rate" and record.status is MeasurementStatus.MISSING
    ]

    assert len(missing) == 1
    assert missing[0].value is None
    assert missing[0].missing_reason is MissingReason.SENSOR_FAILURE


def test_tiles_fills_multi_day_gap_as_typed_missing(tmp_path: Path) -> None:
    """A participant with readings on day 1 and day 3, but nothing on day
    2, must get explicit MISSING records for day 2 rather than day 2
    simply not appearing in the dataset at all."""

    frame = pd.DataFrame(
        {
            "date": ["2026-08-16", "2026-08-18"],
            "heart_rate": [70.0, 71.0],
            "steps": [1000.0, 1100.0],
        }
    )
    frame.to_csv(tmp_path / "participant_001.csv", index=False)

    dataset = TilesLoader(column_mapping=default_mapping()).load(LoaderConfig(source_path=tmp_path))

    gap_day_records = [
        record for record in dataset.records if record.timestamp.date().isoformat() == "2026-08-17"
    ]

    assert len(gap_day_records) == 2  # heart_rate + steps
    assert all(record.status is MeasurementStatus.MISSING for record in gap_day_records)
    assert all(record.value is None for record in gap_day_records)
    assert all(record.missing_reason is MissingReason.SENSOR_FAILURE for record in gap_day_records)


def test_tiles_gap_reason_is_configurable(tmp_path: Path) -> None:
    """The gap missing-reason defaults to SENSOR_FAILURE but can be
    overridden once the not_worn detection rule is confirmed."""

    frame = pd.DataFrame(
        {
            "date": ["2026-08-16", "2026-08-18"],
            "heart_rate": [70.0, 71.0],
            "steps": [1000.0, 1100.0],
        }
    )
    frame.to_csv(tmp_path / "participant_001.csv", index=False)

    loader = TilesLoader(
        column_mapping=default_mapping(),
        gap_missing_reason=MissingReason.NOT_WORN,
    )
    dataset = loader.load(LoaderConfig(source_path=tmp_path))

    gap_day_records = [
        record for record in dataset.records if record.timestamp.date().isoformat() == "2026-08-17"
    ]

    assert all(record.missing_reason is MissingReason.NOT_WORN for record in gap_day_records)


def test_tiles_malformed_value_raises(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "date": ["2026-08-16"],
            "heart_rate": ["not_a_number"],
            "steps": [1000.0],
        }
    )
    frame.to_csv(tmp_path / "participant_001.csv", index=False)

    with pytest.raises(ValueError, match="non-numeric value"):
        TilesLoader(column_mapping=default_mapping()).load(LoaderConfig(source_path=tmp_path))


def test_tiles_missing_required_columns_raises(tmp_path: Path) -> None:
    frame = pd.DataFrame({"date": ["2026-08-16"], "heart_rate": [70.0]})
    frame.to_csv(tmp_path / "participant_001.csv", index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        TilesLoader(column_mapping=default_mapping()).load(LoaderConfig(source_path=tmp_path))


def test_tiles_respects_user_id_filter(tmp_path: Path) -> None:
    pd.DataFrame({"date": ["2026-08-16"], "heart_rate": [70.0], "steps": [1000.0]}).to_csv(
        tmp_path / "participant_001.csv", index=False
    )

    pd.DataFrame({"date": ["2026-08-16"], "heart_rate": [65.0], "steps": [800.0]}).to_csv(
        tmp_path / "participant_002.csv", index=False
    )

    dataset = TilesLoader(column_mapping=default_mapping()).load(
        LoaderConfig(source_path=tmp_path, user_ids=("participant_001",))
    )

    assert dataset.users == ("participant_001",)
