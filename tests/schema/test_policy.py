"""T4B — Founder policy tests. One test group per founder rule."""

from __future__ import annotations

from dataclasses import dataclass

from chronis_ml.schema.classification import (
    DataClassification,
    DataSource,
    ObjectType,
    Representation,
)
from chronis_ml.schema.policy import AccessDecision, evaluate_access


@dataclass(frozen=True, slots=True)
class FakeConsentContext:
    audio_processing_consent: bool = False
    raw_audio_retention_consent: bool = False
    is_first_party: bool = True
    third_party_grant: bool = False

    def user_permits_audio_processing(self) -> bool:
        return self.audio_processing_consent

    def user_permits_raw_audio_retention(self) -> bool:
        return self.raw_audio_retention_consent

    def requesting_party_is_first_party(self) -> bool:
        return self.is_first_party

    def third_party_has_explicit_grant(self) -> bool:
        return self.third_party_grant


def prosody_classification() -> DataClassification:
    return DataClassification(
        source=DataSource.AUDIO_DEVICE,
        object_type=ObjectType.PROSODY_FEATURES,
        representation=Representation.DERIVED_FEATURE,
        provenance_version="1.0",
    )


def raw_audio_classification() -> DataClassification:
    return DataClassification(
        source=DataSource.AUDIO_DEVICE,
        object_type=ObjectType.RAW_AUDIO,
        representation=Representation.RAW,
        provenance_version="1.0",
    )


def biometric_classification() -> DataClassification:
    return DataClassification(
        source=DataSource.AUDIO_DEVICE,
        object_type=ObjectType.BIOMETRIC_FEATURES,
        representation=Representation.DERIVED_FEATURE,
        provenance_version="1.0",
    )


def physiological_classification() -> DataClassification:
    return DataClassification(
        source=DataSource.OMSIGNAL,
        object_type=ObjectType.PHYSIOLOGICAL_FEATURES,
        representation=Representation.DERIVED_FEATURE,
        provenance_version="1.0",
    )


# --- Rule 1: derived voice/prosody features -----------------------------------


def test_rule1_prosody_allowed_with_consent() -> None:
    decision = evaluate_access(
        prosody_classification(), FakeConsentContext(audio_processing_consent=True)
    )
    assert decision.decision is AccessDecision.ALLOW
    assert decision.rule == "1"


def test_rule1_prosody_denied_without_consent() -> None:
    decision = evaluate_access(
        prosody_classification(), FakeConsentContext(audio_processing_consent=False)
    )
    assert decision.decision is AccessDecision.DENY
    assert decision.rule == "1"


# --- Rule 2: raw recorded audio ------------------------------------------------


def test_rule2_raw_audio_allowed_with_retention_consent() -> None:
    decision = evaluate_access(
        raw_audio_classification(), FakeConsentContext(raw_audio_retention_consent=True)
    )
    assert decision.decision is AccessDecision.ALLOW
    assert decision.rule == "2"


def test_rule2_raw_audio_denied_without_retention_consent() -> None:
    decision = evaluate_access(
        raw_audio_classification(), FakeConsentContext(raw_audio_retention_consent=False)
    )
    assert decision.decision is AccessDecision.DENY
    assert decision.rule == "2"


def test_rule2_is_independent_of_rule1_consent() -> None:
    """Raw-audio retention consent and general audio-processing consent
    are separate flags — granting one must not silently grant the
    other."""

    context = FakeConsentContext(audio_processing_consent=True, raw_audio_retention_consent=False)
    decision = evaluate_access(raw_audio_classification(), context)
    assert decision.decision is AccessDecision.DENY


# --- Rule 3: voice fingerprints — internal only --------------------------------


def test_rule3_biometric_allowed_for_first_party() -> None:
    decision = evaluate_access(biometric_classification(), FakeConsentContext(is_first_party=True))
    assert decision.decision is AccessDecision.ALLOW
    assert decision.rule == "3"


def test_rule3_biometric_denied_for_third_party_without_grant() -> None:
    """Rule 4 (deny-by-default) takes precedence for third parties —
    Rule 3's internal-use-only framing means a third party gets NO
    automatic access, only Rule 4's explicit-grant exception applies."""

    decision = evaluate_access(
        biometric_classification(),
        FakeConsentContext(is_first_party=False, third_party_grant=False),
    )
    assert decision.decision is AccessDecision.DENY
    assert decision.rule == "4"


# --- Rule 4: third-party access — deny by default -----------------------------


def test_rule4_third_party_denied_by_default() -> None:
    decision = evaluate_access(
        prosody_classification(), FakeConsentContext(is_first_party=False, third_party_grant=False)
    )
    assert decision.decision is AccessDecision.DENY
    assert decision.rule == "4"


def test_rule4_third_party_allowed_with_explicit_grant() -> None:
    decision = evaluate_access(
        prosody_classification(), FakeConsentContext(is_first_party=False, third_party_grant=True)
    )
    assert decision.decision is AccessDecision.ALLOW
    assert decision.rule == "4"


def test_rule4_applies_regardless_of_object_type() -> None:
    """Deny-by-default for third parties must hold even for raw audio,
    even for physiological data — Rule 4 gates everything, not just
    audio-family objects."""

    for classification in (
        raw_audio_classification(),
        biometric_classification(),
        physiological_classification(),
    ):
        decision = evaluate_access(
            classification, FakeConsentContext(is_first_party=False, third_party_grant=False)
        )
        assert decision.decision is AccessDecision.DENY
        assert decision.rule == "4"


# --- Out-of-scope classifications ----------------------------------------------


def test_physiological_data_outside_the_4_rules_scope() -> None:
    decision = evaluate_access(
        physiological_classification(), FakeConsentContext(is_first_party=True)
    )
    assert decision.decision is AccessDecision.ALLOW
    assert decision.rule == "n/a"
