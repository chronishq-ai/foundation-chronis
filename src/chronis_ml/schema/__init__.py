"""Canonical data schema definitions for Chronis."""

from .models import (
    ChronisDataset,
    FeatureMetadata,
    FeatureRecord,
    MeasurementStatus,
    MissingnessSignals,
    MissingReason,
)
from .validation import (
    PHYSIOLOGICAL_RANGES,
    SUPPORTED_SCHEMA_VERSIONS,
    SchemaValidationError,
    classify_missing_reason,
    validate_dataset,
    validate_record,
)

__all__ = [
    "ChronisDataset",
    "FeatureMetadata",
    "FeatureRecord",
    "MeasurementStatus",
    "MissingnessSignals",
    "MissingReason",
    "PHYSIOLOGICAL_RANGES",
    "SUPPORTED_SCHEMA_VERSIONS",
    "SchemaValidationError",
    "classify_missing_reason",
    "validate_dataset",
    "validate_record",
]
