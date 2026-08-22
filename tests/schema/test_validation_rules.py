"""Tests for the S1.3 schema-validation gaps: duplicate-observation
detection, invalid missing-reason/modality combinations, NaN-vs-typed-NULL
policy, impossible physiological value ranges, feature-name allowlisting,
user isolation, and schema-version compatibility.
"""

import math
from datetime import UTC, datetime

import pytest

from chronis_ml.schema.models import (
    ChronisDataset,
    FeatureRecord,
    MeasurementStatus,
    MissingReason,
)
from chronis_ml.schema.validation import SchemaValidationError, validate_dataset


def make_observed(
    *,
    user_id: str = "user_001",
    timestamp: datetime | None = None,
    feature_name: str = "heart_rate",
    value: float = 72.0,
    modality: str = "ppg",
    schema_version: str = "1.0",
) -> FeatureRecord:
    return FeatureRecord(
        user_id=user_id,
        timestamp=timestamp or datetime(2026, 8, 16, 12, 0, tzinfo=UTC),
        feature_name=feature_name,
        value=value,
        modality=modality,
        status=MeasurementStatus.OBSERVED,
        schema_version=schema_version,
    )


def make_missing(
    *,
    user_id: str = "user_001",
    timestamp: datetime | None = None,
    feature_name: str = "heart_rate",
    modality: str = "ppg",
    reason: MissingReason = MissingReason.SENSOR_FAILURE,
) -> FeatureRecord:
    return FeatureRecord(
        user_id=user_id,
        timestamp=timestamp or datetime(2026, 8, 16, 12, 0, tzinfo=UTC),
        feature_name=feature_name,
        value=None,
        modality=modality,
        status=MeasurementStatus.MISSING,
        missing_reason=reason,
    )


# --- Duplicate-observation detection -----------------------------------


def test_duplicate_observation_is_rejected() -> None:
    timestamp = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)
    record_a = make_observed(timestamp=timestamp, value=70.0)
    record_b = make_observed(timestamp=timestamp, value=71.0)  # same key, diff value

    dataset = ChronisDataset.from_records([record_a, record_b])

    with pytest.raises(SchemaValidationError, match="duplicate observation"):
        validate_dataset(dataset)


def test_non_duplicate_records_pass() -> None:
    record_a = make_observed(timestamp=datetime(2026, 8, 16, 12, 0, tzinfo=UTC))
    record_b = make_observed(timestamp=datetime(2026, 8, 17, 12, 0, tzinfo=UTC))

    dataset = ChronisDataset.from_records([record_a, record_b])

    validate_dataset(dataset)  # should not raise


# --- Invalid missing-reason / modality combinations ---------------------


def test_audio_paused_on_non_audio_modality_is_rejected() -> None:
    record = make_missing(modality="imu", reason=MissingReason.AUDIO_PAUSED)

    with pytest.raises(SchemaValidationError, match="AUDIO_PAUSED"):
        validate_dataset(ChronisDataset.from_records([record]))


def test_audio_paused_on_audio_modality_is_accepted() -> None:
    record = make_missing(modality="audio", reason=MissingReason.AUDIO_PAUSED)

    validate_dataset(ChronisDataset.from_records([record]))  # should not raise


# --- NaN-vs-typed-NULL policy --------------------------------------------


def test_observed_nan_value_is_rejected() -> None:
    record = make_observed(value=math.nan)

    with pytest.raises(SchemaValidationError, match="finite"):
        validate_dataset(ChronisDataset.from_records([record]))


def test_observed_inf_value_is_rejected() -> None:
    record = make_observed(value=math.inf)

    with pytest.raises(SchemaValidationError, match="finite"):
        validate_dataset(ChronisDataset.from_records([record]))


# --- Impossible physiological value ranges -------------------------------


def test_negative_heart_rate_is_rejected() -> None:
    record = make_observed(feature_name="heart_rate", value=-10.0)

    with pytest.raises(SchemaValidationError, match="plausible physiological range"):
        validate_dataset(ChronisDataset.from_records([record]))


def test_implausibly_high_heart_rate_is_rejected() -> None:
    record = make_observed(feature_name="heart_rate", value=900.0)

    with pytest.raises(SchemaValidationError, match="plausible physiological range"):
        validate_dataset(ChronisDataset.from_records([record]))


def test_plausible_heart_rate_is_accepted() -> None:
    record = make_observed(feature_name="heart_rate", value=68.0)

    validate_dataset(ChronisDataset.from_records([record]))  # should not raise


def test_unknown_feature_name_skips_range_check() -> None:
    """A feature with no known physiological range must not be rejected
    just because it has no entry in the range table — this is a targeted
    corruption check, not a closed allowlist."""

    record = make_observed(feature_name="screen_time_minutes", value=999_999.0)

    validate_dataset(ChronisDataset.from_records([record]))  # should not raise


# --- Feature-name allowlisting --------------------------------------------


def test_feature_name_outside_allowlist_is_rejected() -> None:
    record = make_observed(feature_name="heart_rate")

    with pytest.raises(SchemaValidationError, match="not in the allowed feature list"):
        validate_dataset(
            ChronisDataset.from_records([record]),
            allowed_feature_names=frozenset({"steps"}),
        )


def test_feature_name_inside_allowlist_is_accepted() -> None:
    record = make_observed(feature_name="heart_rate")

    validate_dataset(
        ChronisDataset.from_records([record]),
        allowed_feature_names=frozenset({"heart_rate", "steps"}),
    )  # should not raise


def test_allowlist_not_enforced_when_omitted() -> None:
    record = make_observed(feature_name="heart_rate")

    validate_dataset(ChronisDataset.from_records([record]))  # no allowlist, no raise


# --- User isolation (structural property test) ----------------------------


def test_by_user_never_leaks_another_users_records() -> None:
    user_a_record = make_observed(user_id="user_a")
    user_b_record = make_observed(user_id="user_b")

    dataset = ChronisDataset.from_records([user_a_record, user_b_record])

    assert dataset.by_user("user_a") == (user_a_record,)
    assert dataset.by_user("user_b") == (user_b_record,)
    assert user_b_record not in dataset.by_user("user_a")
    assert user_a_record not in dataset.by_user("user_b")


def test_by_user_unknown_id_returns_empty() -> None:
    dataset = ChronisDataset.from_records([make_observed(user_id="user_a")])

    assert dataset.by_user("nonexistent_user") == ()


# --- Schema-version compatibility -----------------------------------------


def test_unsupported_schema_version_is_rejected() -> None:
    record = make_observed(schema_version="0.1")

    with pytest.raises(SchemaValidationError, match="unsupported schema_version"):
        validate_dataset(ChronisDataset.from_records([record]))


def test_supported_schema_version_is_accepted() -> None:
    record = make_observed(schema_version="1.0")

    validate_dataset(ChronisDataset.from_records([record]))  # should not raise
