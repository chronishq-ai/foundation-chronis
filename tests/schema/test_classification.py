"""T4A — Data classification schema tests.

Required negative tests (per the resolved S1.1/T4 direction), each
present below:
  - OMSignal classified as audio -> reject
  - audio-derived feature classified as raw recording -> reject
  - ambiguous source/provenance where a safe classification cannot be
    made -> reject
  - inappropriate object-type/source combinations -> reject

Plus the required persistence/round-trip regression proving
classification survives store-and-reload, and a FeatureRecord
integration test proving the schema-level fix (T4A) actually attaches
at the record level, not just in isolation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from chronis_ml.schema.classification import (
    DataClassification,
    DataClassificationError,
    DataSource,
    ObjectType,
    RawDataObject,
    Representation,
)
from chronis_ml.schema.models import FeatureRecord, MeasurementStatus


def make_timestamp() -> datetime:
    return datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


# --- Required negative test 1: OMSignal classified as audio ------------------


def test_omsignal_classified_as_raw_audio_is_rejected() -> None:
    with pytest.raises(DataClassificationError, match="invalid classification"):
        DataClassification(
            source=DataSource.OMSIGNAL,
            object_type=ObjectType.RAW_AUDIO,
            representation=Representation.RAW,
            provenance_version="1.0",
        )


def test_omsignal_classified_as_prosody_is_rejected() -> None:
    """OMSignal must not be classifiable as ANY audio-family type, not
    just raw audio specifically."""

    with pytest.raises(DataClassificationError, match="invalid classification"):
        DataClassification(
            source=DataSource.OMSIGNAL,
            object_type=ObjectType.PROSODY_FEATURES,
            representation=Representation.DERIVED_FEATURE,
            provenance_version="1.0",
        )


def test_omsignal_classified_as_transcript_is_rejected() -> None:
    with pytest.raises(DataClassificationError, match="invalid classification"):
        DataClassification(
            source=DataSource.OMSIGNAL,
            object_type=ObjectType.TRANSCRIPT,
            representation=Representation.DERIVED_FEATURE,
            provenance_version="1.0",
        )


def test_omsignal_physiological_and_motion_are_accepted() -> None:
    """Positive control — OMSignal's actually-valid classifications
    must still work; this proves the rejection above is targeted, not
    an overly-broad block on OMSignal entirely."""

    physio = DataClassification(
        source=DataSource.OMSIGNAL,
        object_type=ObjectType.PHYSIOLOGICAL_FEATURES,
        representation=Representation.DERIVED_FEATURE,
        provenance_version="1.0",
    )
    motion = DataClassification(
        source=DataSource.OMSIGNAL,
        object_type=ObjectType.MOTION_FEATURES,
        representation=Representation.RAW,
        provenance_version="1.0",
    )
    assert physio.source is DataSource.OMSIGNAL
    assert motion.source is DataSource.OMSIGNAL


# --- Required negative test 2: audio-derived feature as raw recording -------


def test_prosody_features_classified_as_raw_is_rejected() -> None:
    """A derived feature (prosody) can never be tagged representation=RAW
    — that would misrepresent a derived statistic as the actual
    recording."""

    with pytest.raises(DataClassificationError, match="invalid classification"):
        DataClassification(
            source=DataSource.AUDIO_DEVICE,
            object_type=ObjectType.PROSODY_FEATURES,
            representation=Representation.RAW,
            provenance_version="1.0",
        )


def test_biometric_features_classified_as_raw_is_rejected() -> None:
    with pytest.raises(DataClassificationError, match="invalid classification"):
        DataClassification(
            source=DataSource.AUDIO_DEVICE,
            object_type=ObjectType.BIOMETRIC_FEATURES,
            representation=Representation.RAW,
            provenance_version="1.0",
        )


def test_raw_audio_as_derived_feature_is_also_rejected() -> None:
    """The inverse must also be rejected: raw audio can never be
    labeled as a derived feature — that would understate what it
    actually is (the real recording)."""

    with pytest.raises(DataClassificationError, match="invalid classification"):
        DataClassification(
            source=DataSource.AUDIO_DEVICE,
            object_type=ObjectType.RAW_AUDIO,
            representation=Representation.DERIVED_FEATURE,
            provenance_version="1.0",
        )


# --- Required negative test 3: ambiguous source/provenance -------------------


def test_missing_provenance_version_is_rejected() -> None:
    with pytest.raises(DataClassificationError, match="provenance_version must not be empty"):
        DataClassification(
            source=DataSource.AUDIO_DEVICE,
            object_type=ObjectType.RAW_AUDIO,
            representation=Representation.RAW,
            provenance_version="",
        )


def test_whitespace_only_provenance_version_is_rejected() -> None:
    with pytest.raises(DataClassificationError, match="provenance_version must not be empty"):
        DataClassification(
            source=DataSource.AUDIO_DEVICE,
            object_type=ObjectType.RAW_AUDIO,
            representation=Representation.RAW,
            provenance_version="   ",
        )


def test_unknown_source_with_sensitive_object_type_is_rejected() -> None:
    """DataSource.UNKNOWN paired with any object type is not in
    _VALID_COMBINATIONS at all — an unresolved/ambiguous source can
    never be paired with a real object type; it must be resolved to a
    real source before classification can succeed."""

    with pytest.raises(DataClassificationError, match="invalid classification"):
        DataClassification(
            source=DataSource.UNKNOWN,
            object_type=ObjectType.RAW_AUDIO,
            representation=Representation.RAW,
            provenance_version="1.0",
        )


# --- Required negative test 4: inappropriate object-type/source combos ------


def test_fitbit_classified_as_biometric_voice_features_is_rejected() -> None:
    with pytest.raises(DataClassificationError, match="invalid classification"):
        DataClassification(
            source=DataSource.FITBIT,
            object_type=ObjectType.BIOMETRIC_FEATURES,
            representation=Representation.DERIVED_FEATURE,
            provenance_version="1.0",
        )


def test_globem_classified_as_raw_audio_is_rejected() -> None:
    with pytest.raises(DataClassificationError, match="invalid classification"):
        DataClassification(
            source=DataSource.GLOBEM,
            object_type=ObjectType.RAW_AUDIO,
            representation=Representation.RAW,
            provenance_version="1.0",
        )


# --- Positive controls: every valid audio-family classification works -------


def test_raw_audio_from_audio_device_is_accepted() -> None:
    classification = DataClassification(
        source=DataSource.AUDIO_DEVICE,
        object_type=ObjectType.RAW_AUDIO,
        representation=Representation.RAW,
        provenance_version="1.0",
    )
    assert classification.object_type is ObjectType.RAW_AUDIO


def test_prosody_from_audio_device_is_accepted() -> None:
    DataClassification(
        source=DataSource.AUDIO_DEVICE,
        object_type=ObjectType.PROSODY_FEATURES,
        representation=Representation.DERIVED_FEATURE,
        provenance_version="1.0",
    )


def test_biometric_voice_fingerprint_from_audio_device_is_accepted() -> None:
    DataClassification(
        source=DataSource.AUDIO_DEVICE,
        object_type=ObjectType.BIOMETRIC_FEATURES,
        representation=Representation.DERIVED_FEATURE,
        provenance_version="1.0",
    )


def test_transcript_from_audio_device_is_accepted() -> None:
    DataClassification(
        source=DataSource.AUDIO_DEVICE,
        object_type=ObjectType.TRANSCRIPT,
        representation=Representation.DERIVED_FEATURE,
        provenance_version="1.0",
    )


# --- RawDataObject: raw audio doesn't fit FeatureRecord's numeric shape -----


def test_raw_audio_object_construction() -> None:
    classification = DataClassification(
        source=DataSource.AUDIO_DEVICE,
        object_type=ObjectType.RAW_AUDIO,
        representation=Representation.RAW,
        provenance_version="1.0",
    )

    raw_object = RawDataObject(
        user_id="user_001",
        timestamp=make_timestamp(),
        classification=classification,
        storage_reference="s3://bucket/user_001/2026-08-16/audio_0001.wav",
    )

    assert raw_object.classification.object_type is ObjectType.RAW_AUDIO


def test_raw_data_object_requires_nonempty_storage_reference() -> None:
    classification = DataClassification(
        source=DataSource.AUDIO_DEVICE,
        object_type=ObjectType.RAW_AUDIO,
        representation=Representation.RAW,
        provenance_version="1.0",
    )

    with pytest.raises(DataClassificationError, match="storage_reference"):
        RawDataObject(
            user_id="user_001",
            timestamp=make_timestamp(),
            classification=classification,
            storage_reference="",
        )


# --- FeatureRecord integration: classification attaches at the record level -


def test_feature_record_carries_classification() -> None:
    classification = DataClassification(
        source=DataSource.OMSIGNAL,
        object_type=ObjectType.PHYSIOLOGICAL_FEATURES,
        representation=Representation.DERIVED_FEATURE,
        provenance_version="1.0",
    )

    record = FeatureRecord(
        user_id="user_001",
        timestamp=make_timestamp(),
        feature_name="heart_rate",
        value=72.0,
        modality="physiological",
        status=MeasurementStatus.OBSERVED,
        classification=classification,
    )

    assert record.classification is not None
    assert record.classification.source is DataSource.OMSIGNAL


def test_feature_record_classification_defaults_to_none() -> None:
    """Fully backward compatible: existing code that never sets
    classification keeps working unchanged."""

    record = FeatureRecord(
        user_id="user_001",
        timestamp=make_timestamp(),
        feature_name="heart_rate",
        value=72.0,
        modality="physiological",
        status=MeasurementStatus.OBSERVED,
    )

    assert record.classification is None


# --- REQUIRED: persistence/round-trip regression -----------------------------


def test_classification_survives_json_round_trip() -> None:
    """Per the T4A closure requirement: classification must survive
    store-and-reload, not just exist inside the loader. This proves it
    at the JSON-serialization level, which is the persistence mechanism
    actually available in this repo today (matching the pattern
    already used by `tiles_participant_index.Manifest.to_dict()`/
    `from_dict()`). A real feature-store round-trip (S2.4) is separate,
    not-yet-built infrastructure — this test proves the classification
    object ITSELF has no representation loss through serialization,
    which is the precondition any real store must also satisfy."""

    original = DataClassification(
        source=DataSource.AUDIO_DEVICE,
        object_type=ObjectType.PROSODY_FEATURES,
        representation=Representation.DERIVED_FEATURE,
        provenance_version="1.0",
        sensitivity="elevated",
    )

    serialized = json.dumps(original.to_dict())
    reloaded = DataClassification.from_dict(json.loads(serialized))

    assert reloaded == original
    assert reloaded.object_type is ObjectType.PROSODY_FEATURES  # not silently downgraded/lost


def test_raw_data_object_survives_json_round_trip() -> None:
    original = RawDataObject(
        user_id="user_001",
        timestamp=make_timestamp(),
        classification=DataClassification(
            source=DataSource.AUDIO_DEVICE,
            object_type=ObjectType.BIOMETRIC_FEATURES,
            representation=Representation.DERIVED_FEATURE,
            provenance_version="1.0",
        ),
        storage_reference="s3://bucket/user_001/voice_signature_0001.bin",
    )

    serialized = json.dumps(original.to_dict())
    reloaded = RawDataObject.from_dict(json.loads(serialized))

    assert reloaded == original
    assert reloaded.classification.object_type is ObjectType.BIOMETRIC_FEATURES


def test_from_dict_rejects_a_hand_corrupted_stored_record() -> None:
    """A stored record that was somehow hand-edited into an invalid
    state (e.g. someone manually changed source to omsignal in a JSON
    file) must be rejected on RELOAD too, not just on original
    construction — the invariant is enforced at every entry point, not
    only the happy path."""

    corrupted_dict = {
        "source": "omsignal",
        "object_type": "raw_audio",
        "representation": "raw",
        "provenance_version": "1.0",
        "sensitivity": "standard",
    }

    with pytest.raises(DataClassificationError, match="invalid classification"):
        DataClassification.from_dict(corrupted_dict)
