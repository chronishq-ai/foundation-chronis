"""Shared utilities for dataset loaders."""

from __future__ import annotations

from datetime import date, datetime
from typing import cast

import pandas as pd

from chronis_ml.schema.models import (
    FeatureRecord,
    MeasurementStatus,
    MissingReason,
)


def parse_timestamp(value: object) -> datetime:
    """Convert a source timestamp into a timezone-aware datetime."""

    timestamp_input = cast(
        str | float | int | date | datetime,
        value,
    )

    timestamp = pd.to_datetime(timestamp_input, utc=True)

    if not isinstance(timestamp, pd.Timestamp):
        raise ValueError(f"Unable to parse timestamp: {value!r}")

    return timestamp.to_pydatetime()


def normalize_feature_name(name: str) -> str:
    """Normalize source feature names."""

    normalized = name.strip().lower()

    for character in (" ", "-", ":", "/", "."):
        normalized = normalized.replace(character, "_")

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized.strip("_")


def build_observed_record(
    *,
    user_id: str,
    timestamp: datetime,
    feature_name: str,
    value: float,
    modality: str,
    unit: str | None,
    source: str,
) -> FeatureRecord:
    """Create an observed canonical feature record."""

    return FeatureRecord(
        user_id=user_id,
        timestamp=timestamp,
        feature_name=feature_name,
        value=value,
        modality=modality,
        status=MeasurementStatus.OBSERVED,
        unit=unit,
        source=source,
    )


def build_missing_record(
    *,
    user_id: str,
    timestamp: datetime,
    feature_name: str,
    modality: str,
    reason: MissingReason,
    unit: str | None,
    source: str,
) -> FeatureRecord:
    """Create a typed missing canonical feature record."""

    return FeatureRecord(
        user_id=user_id,
        timestamp=timestamp,
        feature_name=feature_name,
        value=None,
        modality=modality,
        status=MeasurementStatus.MISSING,
        missing_reason=reason,
        unit=unit,
        source=source,
    )