"""GLOBEM dataset loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from chronis_ml.loaders.base import LoaderConfig
from chronis_ml.loaders.utils import (
    build_missing_record,
    build_observed_record,
    normalize_feature_name,
    parse_timestamp,
)
from chronis_ml.schema.models import (
    ChronisDataset,
    FeatureMetadata,
    MissingReason,
)
from chronis_ml.schema.validation import validate_dataset


class GlobemLoader:
    """Load GLOBEM feature data into the Chronis schema."""

    SUPPORTED_FILES = (
        "rapids.csv",
        "location.csv",
        "screen.csv",
        "call.csv",
        "bluetooth.csv",
        "steps.csv",
        "sleep.csv",
        "wifi.csv",
    )

    MODALITY_MAP = {
        "rapids.csv": "rapids",
        "location.csv": "location",
        "screen.csv": "screen",
        "call.csv": "call",
        "bluetooth.csv": "bluetooth",
        "steps.csv": "steps",
        "sleep.csv": "sleep",
        "wifi.csv": "wifi",
    }

    @property
    def dataset_name(self) -> str:
        return "globem"

    def load(self, config: LoaderConfig) -> ChronisDataset:
        """Load all supported GLOBEM feature files."""

        feature_directory = self._resolve_feature_directory(
            config.source_path
        )

        records = []
        metadata = {}

        for filename in self.SUPPORTED_FILES:
            path = feature_directory / filename

            if not path.exists():
                continue

            frame = pd.read_csv(path)

            file_records, file_metadata = self._parse_file(
                frame,
                filename,
                config,
            )

            records.extend(file_records)

            for item in file_metadata:
                metadata[item.name] = item

        dataset = ChronisDataset.from_records(
            records,
            metadata.values(),
        )

        validate_dataset(dataset)

        return dataset

    @staticmethod
    def _resolve_feature_directory(
        source_path: Path,
    ) -> Path:
        """Locate the GLOBEM FeatureData directory."""

        if source_path.name == "FeatureData":
            return source_path

        candidate = source_path / "FeatureData"

        if candidate.is_dir():
            return candidate

        raise FileNotFoundError(
            f"FeatureData directory not found under {source_path}"
        )

    def _parse_file(
        self,
        frame: pd.DataFrame,
        filename: str,
        config: LoaderConfig,
    ):
        """Convert one GLOBEM feature file."""

        required_columns = {"pid", "date"}

        missing = required_columns - set(frame.columns)

        if missing:
            raise ValueError(
                f"{filename} is missing required columns: "
                f"{sorted(missing)}"
            )

        modality = self.MODALITY_MAP[filename]

        feature_columns = [
            column
            for column in frame.columns
            if column not in {"pid", "date"}
        ]

        metadata = [
            FeatureMetadata(
                name=normalize_feature_name(column),
                modality=modality,
                source_feature=column,
            )
            for column in feature_columns
        ]

        records = []

        for row in frame.itertuples(index=False):
            user_id = str(row.pid)

            if (
                config.user_ids is not None
                and user_id not in config.user_ids
            ):
                continue

            timestamp = parse_timestamp(row.date)

            for column in feature_columns:
                value = getattr(row, column)

                feature_name = normalize_feature_name(column)

                if pd.isna(value):
                    record = build_missing_record(
                        user_id=user_id,
                        timestamp=timestamp,
                        feature_name=feature_name,
                        modality=modality,
                        reason=MissingReason.SENSOR_FAILURE,
                        unit=None,
                        source="globem",
                    )
                else:
                    record = build_observed_record(
                        user_id=user_id,
                        timestamp=timestamp,
                        feature_name=feature_name,
                        value=float(value),
                        modality=modality,
                        unit=None,
                        source="globem",
                    )

                records.append(record)

        return records, metadata