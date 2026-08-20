from pathlib import Path

import pandas as pd
import pytest

from chronis_ml.loaders.base import LoaderConfig
from chronis_ml.loaders.globem import GlobemLoader
from chronis_ml.schema.models import MeasurementStatus


def create_fixture(root: Path) -> Path:
    feature_data = root / "FeatureData"
    feature_data.mkdir()

    frame = pd.DataFrame(
        {
            "pid": ["user_001", "user_001"],
            "date": [
                "2026-08-16",
                "2026-08-17",
            ],
            "heart_rate": [70.0, 72.0],
            "step_count": [1000.0, None],
        }
    )

    frame.to_csv(
        feature_data / "rapids.csv",
        index=False,
    )

    return root


def test_globem_loader_returns_dataset(tmp_path: Path) -> None:
    source = create_fixture(tmp_path)

    dataset = GlobemLoader().load(LoaderConfig(source_path=source))

    assert len(dataset.records) == 4


def test_globem_preserves_missing_value(
    tmp_path: Path,
) -> None:
    source = create_fixture(tmp_path)

    dataset = GlobemLoader().load(LoaderConfig(source_path=source))

    missing = [
        record
        for record in dataset.records
        if record.feature_name == "step_count" and record.status is MeasurementStatus.MISSING
    ]

    assert len(missing) == 1
    assert missing[0].value is None


def test_globem_handles_non_identifier_column_names(tmp_path: Path) -> None:
    """Column names with spaces/slashes must not break row access.

    itertuples() renames these columns internally (e.g. to "_1"); the
    loader must not depend on that renamed form.
    """

    feature_data = tmp_path / "FeatureData"
    feature_data.mkdir()

    frame = pd.DataFrame(
        {
            "pid": ["user_001"],
            "date": ["2026-08-16"],
            "Heart Rate / Mean": [72.0],
        }
    )
    frame.to_csv(feature_data / "rapids.csv", index=False)

    dataset = GlobemLoader().load(LoaderConfig(source_path=tmp_path))

    assert dataset.records[0].feature_name == "heart_rate_mean"
    assert dataset.records[0].value == 72.0


def test_globem_missing_required_columns_raises(tmp_path: Path) -> None:
    feature_data = tmp_path / "FeatureData"
    feature_data.mkdir()

    frame = pd.DataFrame({"heart_rate": [72.0]})
    frame.to_csv(feature_data / "rapids.csv", index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        GlobemLoader().load(LoaderConfig(source_path=tmp_path))


def test_globem_malformed_value_raises(tmp_path: Path) -> None:
    """A present-but-non-numeric value is malformed data, not missing data,
    and must raise rather than being silently coerced or dropped."""

    feature_data = tmp_path / "FeatureData"
    feature_data.mkdir()

    frame = pd.DataFrame(
        {
            "pid": ["user_001"],
            "date": ["2026-08-16"],
            "heart_rate": ["not_a_number"],
        }
    )
    frame.to_csv(feature_data / "rapids.csv", index=False)

    with pytest.raises(ValueError, match="non-numeric value"):
        GlobemLoader().load(LoaderConfig(source_path=tmp_path))


def test_globem_skips_unsupported_or_missing_files(tmp_path: Path) -> None:
    """FeatureData directories that only contain some of the supported
    files should load cleanly rather than failing on the missing ones."""

    feature_data = tmp_path / "FeatureData"
    feature_data.mkdir()

    frame = pd.DataFrame(
        {
            "pid": ["user_001"],
            "date": ["2026-08-16"],
            "step_count": [500.0],
        }
    )
    frame.to_csv(feature_data / "steps.csv", index=False)

    dataset = GlobemLoader().load(LoaderConfig(source_path=tmp_path))

    assert len(dataset.records) == 1
    assert dataset.records[0].modality == "steps"
