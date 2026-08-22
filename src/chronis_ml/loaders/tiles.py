"""TILES-2018 dataset loader.

TILES-2018 ships as per-participant files rather than GLOBEM's single
consolidated-file-per-modality layout. The exact column names have not
been confirmed against the real TILES-2018 release manifest yet, so
this loader does NOT hardcode any guessed column names. Callers must
supply a `TilesColumnMapping` with the verbatim names from the actual
source files.

Open design question (tracked, not decided here): whether a multi-day
gap in a participant's readings should be represented as
`MissingReason.SENSOR_FAILURE` or `MissingReason.NOT_WORN`. This loader
defaults to SENSOR_FAILURE and exposes `gap_missing_reason` so the team
can change it once the detection rule is confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
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


@dataclass(frozen=True, slots=True)
class TilesColumnMapping:
    """Verbatim column names for a TILES-2018 per-participant source file.

    Nothing here has a guessed default. These values must come from the
    actual TILES-2018 release manifest/schema before this loader is run
    against real data.
    """

    timestamp_column: str
    feature_columns: tuple[str, ...]
    modality: str
    participant_id_column: str | None = None
    """If the source file has no participant-ID column (participant is
    identified purely by filename, as TILES-2018 is described), leave
    this as None and the participant ID is taken from the file stem."""


class TilesLoader:
    """Load TILES-2018 per-participant files into the Chronis schema."""

    def __init__(
        self,
        column_mapping: TilesColumnMapping,
        gap_missing_reason: MissingReason = MissingReason.SENSOR_FAILURE,
        gap_granularity: timedelta = timedelta(days=1),
        file_glob: str = "*.csv",
    ) -> None:
        self._column_mapping = column_mapping
        self._gap_missing_reason = gap_missing_reason
        self._gap_granularity = gap_granularity
        self._file_glob = file_glob

    @property
    def dataset_name(self) -> str:
        return "tiles_2018"

    def load(self, config: LoaderConfig) -> ChronisDataset:
        """Load all per-participant TILES-2018 files under source_path."""

        if not config.source_path.exists():
            raise FileNotFoundError(f"TILES source path does not exist: {config.source_path}")

        mapping = self._column_mapping

        metadata = {
            normalize_feature_name(column): FeatureMetadata(
                name=normalize_feature_name(column),
                modality=mapping.modality,
                source_feature=column,
            )
            for column in mapping.feature_columns
        }

        records: list[FeatureRecord] = []

        for file_path in sorted(config.source_path.glob(self._file_glob)):
            frame = pd.read_csv(file_path)

            file_records = self._parse_participant_file(frame, file_path, config)

            records.extend(file_records)

        dataset = ChronisDataset.from_records(records, metadata.values())

        validate_dataset(dataset)

        return dataset

    def _parse_participant_file(
        self,
        frame: pd.DataFrame,
        file_path: Path,
        config: LoaderConfig,
    ) -> list[FeatureRecord]:
        """Convert one participant's file, filling multi-day gaps."""

        mapping = self._column_mapping

        required_columns = {mapping.timestamp_column, *mapping.feature_columns}
        if mapping.participant_id_column is not None:
            required_columns.add(mapping.participant_id_column)

        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{file_path.name} is missing required columns: {sorted(missing_columns)}"
            )

        if mapping.participant_id_column is not None:
            participant_ids = frame[mapping.participant_id_column].astype(str)
            if participant_ids.nunique() != 1:
                raise ValueError(
                    f"{file_path.name} contains more than one participant ID; "
                    "expected exactly one per file"
                )
            user_id = participant_ids.iloc[0]
        else:
            user_id = file_path.stem

        if config.user_ids is not None and user_id not in config.user_ids:
            return []

        records: list[FeatureRecord] = []
        observed_dates: set[pd.Timestamp] = set()

        for row in frame.to_dict(orient="records"):
            timestamp = parse_timestamp(row[mapping.timestamp_column])
            observed_dates.add(pd.Timestamp(timestamp).normalize())

            for column in mapping.feature_columns:
                value = row[column]
                feature_name = normalize_feature_name(column)

                if pd.isna(value):
                    record = build_missing_record(
                        user_id=user_id,
                        timestamp=timestamp,
                        feature_name=feature_name,
                        modality=mapping.modality,
                        reason=MissingReason.SENSOR_FAILURE,
                        unit=None,
                        source="tiles_2018",
                    )
                else:
                    numeric_value = self._coerce_numeric(
                        value=value,
                        file_path=file_path,
                        column=column,
                        user_id=user_id,
                        row_timestamp=row[mapping.timestamp_column],
                    )
                    record = build_observed_record(
                        user_id=user_id,
                        timestamp=timestamp,
                        feature_name=feature_name,
                        value=numeric_value,
                        modality=mapping.modality,
                        unit=None,
                        source="tiles_2018",
                    )

                records.append(record)

        records.extend(
            self._build_gap_records(
                user_id=user_id,
                observed_dates=observed_dates,
                mapping=mapping,
            )
        )

        return records

    def _build_gap_records(
        self,
        *,
        user_id: str,
        observed_dates: set[pd.Timestamp],
        mapping: TilesColumnMapping,
    ) -> list[FeatureRecord]:
        """Represent multi-day dropout as explicit typed-missing records.

        Given the participant's earliest and latest observed date, any
        expected date in between with no row at all is filled in as a
        MISSING record per feature, rather than being silently absent
        from the dataset (which would be indistinguishable from "this
        participant simply has no data for that day" vs. "we expected
        a reading and never got one").
        """

        if not observed_dates:
            return []

        start = min(observed_dates)
        end = max(observed_dates)

        gap_records: list[FeatureRecord] = []
        current = start

        while current <= end:
            if current not in observed_dates:
                timestamp = current.to_pydatetime()

                for column in mapping.feature_columns:
                    feature_name = normalize_feature_name(column)

                    gap_records.append(
                        build_missing_record(
                            user_id=user_id,
                            timestamp=timestamp,
                            feature_name=feature_name,
                            modality=mapping.modality,
                            reason=self._gap_missing_reason,
                            unit=None,
                            source="tiles_2018",
                        )
                    )

            current = current + self._gap_granularity

        return gap_records

    @staticmethod
    def _coerce_numeric(
        *,
        value: object,
        file_path: Path,
        column: str,
        user_id: str,
        row_timestamp: object,
    ) -> float:
        """Convert a non-missing cell to float, or raise a clear error.

        A value that is present but cannot be interpreted as numeric is
        malformed data, not missing data, and must never be silently
        coerced or dropped.
        """

        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{file_path.name}: column {column!r} contains a non-numeric value "
                f"{value!r} for participant={user_id!r}, timestamp={row_timestamp!r}"
            ) from exc
