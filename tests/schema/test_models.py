from datetime import UTC, datetime

import pytest

from chronis_ml.schema.models import (
    ChronisDataset,
    FeatureMetadata,
    FeatureRecord,
    MeasurementStatus,
    MissingReason,
)
from chronis_ml.schema.validation import SchemaValidationError, validate_dataset


def observed_record() -> FeatureRecord:
    return FeatureRecord(
        user_id="user_001",
        timestamp=datetime(2026, 8, 16, 12, 0, tzinfo=UTC),
        feature_name="heart_rate",
        value=72.0,
        modality="ppg",
        status=MeasurementStatus.OBSERVED,
        unit="bpm",
        source="globem",
    )


def test_valid_observed_record() -> None:
    dataset = ChronisDataset.from_records(
        [observed_record()],
        [FeatureMetadata(name="heart_rate", modality="ppg", unit="bpm")],
    )
    validate_dataset(dataset)


def test_missing_value_requires_typed_reason() -> None:
    record = FeatureRecord(
        user_id="user_001",
        timestamp=datetime(2026, 8, 16, 12, 0, tzinfo=UTC),
        feature_name="heart_rate",
        value=None,
        modality="ppg",
        status=MeasurementStatus.MISSING,
    )

    with pytest.raises(SchemaValidationError, match="missing_reason"):
        validate_dataset(ChronisDataset.from_records([record]))


def test_missing_value_is_not_zero() -> None:
    record = FeatureRecord(
        user_id="user_001",
        timestamp=datetime(2026, 8, 16, 12, 0, tzinfo=UTC),
        feature_name="heart_rate",
        value=None,
        modality="ppg",
        status=MeasurementStatus.MISSING,
        missing_reason=MissingReason.SENSOR_FAILURE,
    )

    validate_dataset(ChronisDataset.from_records([record]))
    assert record.value is None
    assert record.missing_reason is MissingReason.SENSOR_FAILURE


def test_observed_value_cannot_have_missing_reason() -> None:
    base = observed_record()
    record = FeatureRecord(
        user_id=base.user_id,
        timestamp=base.timestamp,
        feature_name=base.feature_name,
        value=base.value,
        modality=base.modality,
        status=base.status,
        missing_reason=MissingReason.NOT_WORN,
        unit=base.unit,
        source=base.source,
    )

    with pytest.raises(SchemaValidationError):
        validate_dataset(ChronisDataset.from_records([record]))


def test_timestamp_must_be_timezone_aware() -> None:
    record = FeatureRecord(
        user_id="user_001",
        timestamp=datetime(2026, 8, 16, 12, 0),
        feature_name="heart_rate",
        value=72.0,
        modality="ppg",
        status=MeasurementStatus.OBSERVED,
    )

    with pytest.raises(SchemaValidationError, match="timezone-aware"):
        validate_dataset(ChronisDataset.from_records([record]))
