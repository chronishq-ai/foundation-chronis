"""Validation rules for the canonical Chronis schema.

Includes:
  - The S1.2 typed-missingness decision table (`classify_missing_reason`),
    which turns raw `MissingnessSignals` into the correct `MissingReason`.
    Wiring this into a specific loader's data flow is Sprint 2 work; this
    module only specifies and tests the rule itself, per the Sprint 1
    remediation scope.
  - The S1.3 validators that were previously missing: duplicate-
    observation detection, missing-reason/modality consistency,
    NaN-vs-typed-NULL policy, physiological value ranges, feature-name
    allowlisting, and schema-version compatibility.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .models import (
    ChronisDataset,
    FeatureMetadata,
    FeatureRecord,
    MeasurementStatus,
    MissingnessSignals,
    MissingReason,
)

_FEATURE_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})
"""Schema versions this codebase knows how to validate. A record tagged
with anything outside this set is an explicit incompatibility, not a
silently-accepted unknown (S1.3 requirement)."""

AUDIO_MODALITY = "audio"
"""The only modality `MissingReason.AUDIO_PAUSED` may legitimately be
attached to. A paused microphone does not silence an accelerometer."""

# Known plausible physiological ranges, inclusive. Deliberately small and
# conservative (wide bounds) — the goal is to catch impossible/corrupt
# values (e.g. negative heart rate), not to do clinical range-checking.
# Extend as more feature names are confirmed against real datasets.
PHYSIOLOGICAL_RANGES: dict[str, tuple[float, float]] = {
    "heart_rate": (20.0, 250.0),
    "respiration_rate": (4.0, 60.0),
    "spo2": (0.0, 100.0),
    "steps": (0.0, 100_000.0),
    "skin_temperature": (20.0, 45.0),  # Celsius
}


class SchemaValidationError(ValueError):
    """Raised when canonical data violates the schema."""


def _valid_snake_case(value: str) -> bool:
    return bool(_FEATURE_NAME_RE.fullmatch(value))


def classify_missing_reason(
    *,
    modality: str,
    signals: MissingnessSignals,
) -> MissingReason:
    """S1.2 decision table: classify why a reading is missing.

    Rules, in priority order:

    1. A discrete mic-off event explains a missing AUDIO reading as
       AUDIO_PAUSED. It does NOT explain a missing reading in any other
       modality — a paused microphone says nothing about whether the
       accelerometer or PPG sensor were working.
    2. IMU stillness together with PPG dropout, at the same time,
       is classified as NOT_WORN: a device producing zero movement and
       zero pulse signal simultaneously is far more consistent with "not
       on the participant's body" than with two independent sensor
       failures occurring at once. Either signal alone is NOT sufficient
       (a participant can be genuinely still while worn; a PPG can drop
       out briefly from a worn device due to motion artifact).
    3. Anything else falls back to SENSOR_FAILURE, the default state for
       a missing reading with no more specific explanation available.

    This function does not touch `FeatureRecord` construction — callers
    (loaders) are expected to call this to choose the `MissingReason`
    they then pass to `build_missing_record`.
    """

    if signals.mic_off_event and modality == AUDIO_MODALITY:
        return MissingReason.AUDIO_PAUSED

    if signals.imu_stillness and signals.ppg_dropout:
        return MissingReason.NOT_WORN

    return MissingReason.SENSOR_FAILURE


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

    if record.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SchemaValidationError(
            f"unsupported schema_version {record.schema_version!r}; "
            f"expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    if record.status is MeasurementStatus.OBSERVED:
        if record.value is None:
            raise SchemaValidationError("observed records must contain a value")
        if record.missing_reason is not None:
            raise SchemaValidationError("observed records cannot contain missing_reason")
        if math.isnan(record.value) or math.isinf(record.value):
            raise SchemaValidationError(
                "observed value must be a finite number, not NaN/inf — "
                "use MeasurementStatus.MISSING with a typed missing_reason instead"
            )
        _validate_physiological_range(record)

    if record.status is MeasurementStatus.MISSING:
        if record.value is not None:
            raise SchemaValidationError("missing records must use value=None")
        if record.missing_reason is None:
            raise SchemaValidationError("missing records must contain a typed missing_reason")
        if (
            record.missing_reason is MissingReason.AUDIO_PAUSED
            and record.modality != AUDIO_MODALITY
        ):
            raise SchemaValidationError(
                f"missing_reason=AUDIO_PAUSED is only valid for modality "
                f"{AUDIO_MODALITY!r}, got modality={record.modality!r}"
            )


def _validate_physiological_range(record: FeatureRecord) -> None:
    """Reject observed values outside a known-plausible physiological
    range for that feature. Features with no known range are skipped —
    this is a targeted corruption check, not a closed allowlist."""

    bounds = PHYSIOLOGICAL_RANGES.get(record.feature_name)
    if bounds is None:
        return

    lower, upper = bounds
    assert record.value is not None  # guaranteed by caller (OBSERVED branch)

    if not (lower <= record.value <= upper):
        raise SchemaValidationError(
            f"{record.feature_name}={record.value!r} is outside the plausible "
            f"physiological range [{lower}, {upper}]"
        )


def validate_metadata(metadata: FeatureMetadata) -> None:
    """Validate feature metadata."""
    if not _valid_snake_case(metadata.name):
        raise SchemaValidationError("metadata.name must use lowercase snake_case")
    if not _valid_snake_case(metadata.modality):
        raise SchemaValidationError("metadata.modality must use lowercase snake_case")


def _validate_no_duplicate_observations(dataset: ChronisDataset) -> None:
    """Reject a dataset containing more than one record for the same
    (user_id, timestamp, feature_name, modality) key — an unversioned
    duplicate observation, which should never occur from a correctly
    behaving loader."""

    keys = [
        (record.user_id, record.timestamp, record.feature_name, record.modality)
        for record in dataset.records
    ]
    counts = Counter(keys)
    duplicates = [key for key, count in counts.items() if count > 1]

    if duplicates:
        user_id, timestamp, feature_name, modality = duplicates[0]
        raise SchemaValidationError(
            f"duplicate observation for user_id={user_id!r}, "
            f"timestamp={timestamp!r}, feature_name={feature_name!r}, "
            f"modality={modality!r} ({counts[duplicates[0]]} occurrences); "
            f"{len(duplicates)} duplicate key(s) total"
        )


def _validate_feature_name_allowlist(
    dataset: ChronisDataset,
    allowed_feature_names: frozenset[str],
) -> None:
    """Reject any record or metadata entry whose feature_name is not in
    the supplied allowlist. Optional — only enforced when the caller
    provides an allowlist to `validate_dataset`."""

    for record in dataset.records:
        if record.feature_name not in allowed_feature_names:
            raise SchemaValidationError(
                f"feature_name {record.feature_name!r} is not in the allowed feature list"
            )

    for metadata in dataset.features:
        if metadata.name not in allowed_feature_names:
            raise SchemaValidationError(
                f"metadata feature name {metadata.name!r} is not in the allowed feature list"
            )


def validate_dataset(
    dataset: ChronisDataset,
    allowed_feature_names: frozenset[str] | None = None,
) -> None:
    """Validate all records and feature metadata.

    `allowed_feature_names`, if provided, additionally restricts every
    record and metadata entry to a known-approved set of feature names
    (S1.3 feature-name allowlisting).
    """
    for record in dataset.records:
        validate_record(record)

    for metadata in dataset.features:
        validate_metadata(metadata)

    _validate_no_duplicate_observations(dataset)

    if allowed_feature_names is not None:
        _validate_feature_name_allowlist(dataset, allowed_feature_names)
