"""Validation rules for the canonical Chronis schema."""

from __future__ import annotations

import re

from .models import ChronisDataset, FeatureMetadata, FeatureRecord, MeasurementStatus

_FEATURE_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


class SchemaValidationError(ValueError):
    """Raised when canonical data violates the schema."""


def _valid_snake_case(value: str) -> bool:
    return bool(_FEATURE_NAME_RE.fullmatch(value))


def validate_record(record: FeatureRecord) -> None:
    """Validate one canonical feature record."""
    if not record.user_id.strip():
        raise SchemaValidationError("user_id must not be empty")

    if record.timestamp.tzinfo is None or record.timestamp.utcoffset() is None:
        raise SchemaValidationError("timestamp must be timezone-aware")

    if not _valid_snake_case(record.feature_name):
        raise SchemaValidationError("feature_name must use lowercase snake_case")

    if not _valid_snake_case(record.modality):
        raise SchemaValidationError("modality must use lowercase snake_case")

    if record.status is MeasurementStatus.OBSERVED:
        if record.value is None:
            raise SchemaValidationError("observed records must contain a value")
        if record.missing_reason is not None:
            raise SchemaValidationError("observed records cannot contain missing_reason")

    if record.status is MeasurementStatus.MISSING:
        if record.value is not None:
            raise SchemaValidationError("missing records must use value=None")
        if record.missing_reason is None:
            raise SchemaValidationError("missing records must contain a typed missing_reason")


def validate_metadata(metadata: FeatureMetadata) -> None:
    """Validate feature metadata."""
    if not _valid_snake_case(metadata.name):
        raise SchemaValidationError("metadata.name must use lowercase snake_case")
    if not _valid_snake_case(metadata.modality):
        raise SchemaValidationError("metadata.modality must use lowercase snake_case")


def validate_dataset(dataset: ChronisDataset) -> None:
    """Validate all records and feature metadata."""
    for record in dataset.records:
        validate_record(record)

    for metadata in dataset.features:
        validate_metadata(metadata)
