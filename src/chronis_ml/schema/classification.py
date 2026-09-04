"""T4A — Data source/object classification schema.

Makes the invariant "OMSignal is physiological/movement data, never
audio" STRUCTURAL rather than a naming convention — a config or record
attempting an invalid (source, object_type, representation) combination
raises immediately, at construction time.

This is a general-purpose classification model, not a TILES-specific
hack: any future data source in the system attaches one of these to
declare, unambiguously, what kind of object it is producing.

Design notes:
  - `DataSource` is deliberately open-ended (new sources get added as
    the system grows) but `ObjectType` and `Representation` encode a
    closed set of currently-known categories, with `_VALID_COMBINATIONS`
    as the single source of truth for what's structurally allowed.
  - Raw audio is NOT modeled as a `FeatureRecord` — a scalar time-series
    value has no meaningful way to represent "a recording." See
    `RawDataObject` below, which is the non-numeric counterpart,
    mirroring the point/interval/event/snippet separation already
    established in the Sprint 1B synthetic harness (never collapse
    incompatible shapes into one schema).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class DataSource(StrEnum):
    """Where data originated. Open-ended — extend as new sources are
    added to the system."""

    OMSIGNAL = "omsignal"
    AUDIO_DEVICE = "audio_device"
    FITBIT = "fitbit"
    GLOBEM = "globem"
    UNKNOWN = "unknown"
    """Explicitly distinct from omitting a source entirely — see
    `test_ambiguous_source_is_rejected`. UNKNOWN is still a declared
    value; a classification with NO source at all (None) is a
    different, also-rejected failure mode."""


class ObjectType(StrEnum):
    """What kind of object this data represents. Closed set — adding a
    new category requires also adding its valid (source, representation)
    combinations to `_VALID_COMBINATIONS` below, by design, so a new
    category can never silently bypass classification rules."""

    MOTION_FEATURES = "motion_features"
    PHYSIOLOGICAL_FEATURES = "physiological_features"
    RAW_AUDIO = "raw_audio"
    PROSODY_FEATURES = "prosody_features"
    BIOMETRIC_FEATURES = "biometric_features"
    TRANSCRIPT = "transcript"


class Representation(StrEnum):
    """Whether this object is the raw captured signal, or a feature
    already derived from it."""

    RAW = "raw"
    DERIVED_FEATURE = "derived_feature"


# The single source of truth for which (source, object_type, representation)
# triples are structurally valid. Anything not listed here is rejected.
_VALID_COMBINATIONS: frozenset[tuple[DataSource, ObjectType, Representation]] = frozenset(
    {
        # OMSignal: physiological/movement only. Never audio-family types.
        (DataSource.OMSIGNAL, ObjectType.MOTION_FEATURES, Representation.RAW),
        (DataSource.OMSIGNAL, ObjectType.MOTION_FEATURES, Representation.DERIVED_FEATURE),
        (DataSource.OMSIGNAL, ObjectType.PHYSIOLOGICAL_FEATURES, Representation.RAW),
        (DataSource.OMSIGNAL, ObjectType.PHYSIOLOGICAL_FEATURES, Representation.DERIVED_FEATURE),
        # Audio device: raw recording, or any of the 3 derived audio categories.
        (DataSource.AUDIO_DEVICE, ObjectType.RAW_AUDIO, Representation.RAW),
        (DataSource.AUDIO_DEVICE, ObjectType.PROSODY_FEATURES, Representation.DERIVED_FEATURE),
        (DataSource.AUDIO_DEVICE, ObjectType.BIOMETRIC_FEATURES, Representation.DERIVED_FEATURE),
        (DataSource.AUDIO_DEVICE, ObjectType.TRANSCRIPT, Representation.DERIVED_FEATURE),
        # Fitbit/GLOBEM: physiological/movement, same shape as OMSignal.
        (DataSource.FITBIT, ObjectType.MOTION_FEATURES, Representation.RAW),
        (DataSource.FITBIT, ObjectType.MOTION_FEATURES, Representation.DERIVED_FEATURE),
        (DataSource.FITBIT, ObjectType.PHYSIOLOGICAL_FEATURES, Representation.RAW),
        (DataSource.FITBIT, ObjectType.PHYSIOLOGICAL_FEATURES, Representation.DERIVED_FEATURE),
        (DataSource.GLOBEM, ObjectType.MOTION_FEATURES, Representation.DERIVED_FEATURE),
        (DataSource.GLOBEM, ObjectType.PHYSIOLOGICAL_FEATURES, Representation.DERIVED_FEATURE),
    }
)


class DataClassificationError(ValueError):
    """Raised when a (source, object_type, representation) combination
    is structurally invalid, or required provenance fields are missing.
    Always raised at construction time — never a silent acceptance
    followed by a downstream surprise."""


@dataclass(frozen=True, slots=True)
class DataClassification:
    """Attached to any canonical object (a `FeatureRecord`, or a
    `RawDataObject` below) to declare, unambiguously, what it is.

    Construction itself enforces every structural invariant — it is not
    possible to hold an invalid `DataClassification` instance in memory,
    since `__post_init__` validates on every construction path,
    including deserialization (see `from_dict`/`to_dict`).
    """

    source: DataSource
    object_type: ObjectType
    representation: Representation
    provenance_version: str
    """Version of the classification/policy ruleset this object was
    classified under — allows detecting stale classifications if the
    rules change later."""
    sensitivity: str = "standard"
    """Downstream sensitivity tag. Deliberately a free-text field here,
    not a boolean — the constitutional/policy layer (T4B) is the
    authority on what sensitivity values mean and how they gate access,
    not this schema module."""

    def __post_init__(self) -> None:
        combination = (self.source, self.object_type, self.representation)

        if combination not in _VALID_COMBINATIONS:
            raise DataClassificationError(
                f"invalid classification: source={self.source.value!r}, "
                f"object_type={self.object_type.value!r}, "
                f"representation={self.representation.value!r} — this combination "
                f"is not in the allowed set. If this is a genuinely new, "
                f"intentional combination, it must be added to "
                f"_VALID_COMBINATIONS explicitly, not bypassed."
            )

        if not self.provenance_version.strip():
            raise DataClassificationError(
                "provenance_version must not be empty — an unversioned "
                "classification cannot be safely reconciled if the "
                "classification ruleset changes later"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source.value,
            "object_type": self.object_type.value,
            "representation": self.representation.value,
            "provenance_version": self.provenance_version,
            "sensitivity": self.sensitivity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> DataClassification:
        """Round-trips through the SAME validation as direct
        construction — a corrupted or hand-edited stored record cannot
        be silently reloaded into an invalid state."""

        return cls(
            source=DataSource(data["source"]),
            object_type=ObjectType(data["object_type"]),
            representation=Representation(data["representation"]),
            provenance_version=data["provenance_version"],
            sensitivity=data.get("sensitivity", "standard"),
        )


@dataclass(frozen=True, slots=True)
class RawDataObject:
    """Non-numeric canonical object — a raw recording or other blob-like
    artifact that cannot be represented as a `FeatureRecord`'s single
    scalar value. `storage_reference` points to where the actual bytes
    live (this class never carries the bytes themselves)."""

    user_id: str
    timestamp: datetime
    classification: DataClassification
    storage_reference: str

    def __post_init__(self) -> None:
        if not self.user_id.strip():
            raise DataClassificationError("user_id must not be empty")
        if self.timestamp.tzinfo is None:
            raise DataClassificationError("timestamp must be timezone-aware")
        if not self.storage_reference.strip():
            raise DataClassificationError("storage_reference must not be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "classification": self.classification.to_dict(),
            "storage_reference": self.storage_reference,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RawDataObject:
        from datetime import datetime as _datetime

        return cls(
            user_id=data["user_id"],  # type: ignore[arg-type]
            timestamp=_datetime.fromisoformat(data["timestamp"]),  # type: ignore[arg-type]
            classification=DataClassification.from_dict(data["classification"]),  # type: ignore[arg-type]
            storage_reference=data["storage_reference"],  # type: ignore[arg-type]
        )
