from pathlib import Path

import pandas as pd

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

    dataset = GlobemLoader().load(
        LoaderConfig(source_path=source)
    )

    assert len(dataset.records) == 4


def test_globem_preserves_missing_value(
    tmp_path: Path,
) -> None:
    source = create_fixture(tmp_path)

    dataset = GlobemLoader().load(
        LoaderConfig(source_path=source)
    )

    missing = [
        record
        for record in dataset.records
        if record.feature_name == "step_count"
        and record.status is MeasurementStatus.MISSING
    ]

    assert len(missing) == 1
    assert missing[0].value is None