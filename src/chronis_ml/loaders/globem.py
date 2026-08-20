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
    FeatureRecord,
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

        feature_directory = self._resolve_feature_directory(config.source_path)

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

        raise FileNotFoundError(f"FeatureData directory not found under {source_path}")

    def _parse_file(
        self,
        frame: pd.DataFrame,
        filename: str,
        config: LoaderConfig,
    ) -> tuple[list[FeatureRecord], list[FeatureMetadata]]:
        """Convert one GLOBEM feature file."""

        required_columns = {"pid", "date"}

        missing = required_columns - set(frame.columns)

        if missing:
            raise ValueError(f"{filename} is missing required columns: {sorted(missing)}")

        modality = self.MODALITY_MAP[filename]

        feature_columns = [column for column in frame.columns if column not in {"pid", "date"}]

        metadata = [
            FeatureMetadata(
                name=normalize_feature_name(column),
                modality=modality,
                source_feature=column,
            )
            for column in feature_columns
        ]

        records: list[FeatureRecord] = []

        # NOTE: iterate via to_dict(orient="records") rather than itertuples().
        # itertuples() silently renames any column that is not a valid Python
        # identifier (spaces, slashes, leading digits, etc.) to a positional
        # name like "_1", which breaks getattr(row, column) lookups without
        # raising an obvious error. Dict-based row access preserves the exact
        # source column name regardless of its shape.
        for row in frame.to_dict(orient="records"):
            user_id = str(row["pid"])

            if config.user_ids is not None and user_id not in config.user_ids:
                continue

            timestamp = parse_timestamp(row["date"])

            for column in feature_columns:
                value = row[column]

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
                    numeric_value = self._coerce_numeric(
                        value=value,
                        filename=filename,
                        column=column,
                        user_id=user_id,
                        date_value=row["date"],
                    )

                    record = build_observed_record(
                        user_id=user_id,
                        timestamp=timestamp,
                        feature_name=feature_name,
                        value=numeric_value,
                        modality=modality,
                        unit=None,
                        source="globem",
                    )

                records.append(record)

        return records, metadata

    @staticmethod
    def _coerce_numeric(
        *,
        value: object,
        filename: str,
        column: str,
        user_id: str,
        date_value: object,
    ) -> float:
        """Convert a non-missing cell to float, or raise a clear error.

        A value that is present but cannot be interpreted as numeric (a
        stray string, an unexpected type, etc.) is malformed data, not
        missing data. It must never be silently coerced or dropped.
        """

        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{filename}: column {column!r} contains a non-numeric value "
                f"{value!r} for pid={user_id!r}, date={date_value!r}"
            ) from exc
