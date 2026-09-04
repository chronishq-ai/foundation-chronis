"""T4B — Founder-approved data access policy.

IMPORTANT HONESTY NOTE, read before relying on this module:

This encodes the 4 founder-approved rules as a standalone,
independently-testable policy table. It is NOT wired into "the
constitutional/consent path" — I do not have visibility into that
system's actual code/interface in this repo or this conversation, so
claiming to have wired it in would be fabricating an integration I
cannot verify. What this module provides is the CORRECT, ready-to-plug
policy logic, encoded exactly per the founder's rules, with full test
coverage of the rules themselves — the actual wiring into the real
constitutional system is the remaining step, and needs someone with
access to that system's real interface to complete it (see the
`ConsentContext` Protocol below, which documents the exact shape of
input this module expects from that system once it's connected).

Per the resolved direction: "Senior review is still required where the
code connects these classifications to actual authorization/consent
behavior... that review now answers 'does this correctly enforce the
founder-approved policy', not 'should the policy be yes or no.'" This
module is built to make that review straightforward — the policy table
below is the single place all 4 rules live, and every rule has its own
test.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from chronis_ml.schema.classification import DataClassification, ObjectType


class AccessDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


class ConsentContext(Protocol):
    """The shape of input this policy module expects from the real
    constitutional/consent system, once wired in. Defined here as a
    Protocol so the real system's actual consent object just needs to
    satisfy this shape — no inheritance required, no coupling to
    whatever that system's real class hierarchy looks like.
    """

    def user_permits_audio_processing(self) -> bool:
        """True if the user's selected mode/consent permits the
        relevant derived-audio-feature processing (Founder Rule 1)."""
        ...

    def user_permits_raw_audio_retention(self) -> bool:
        """True if the user's selected mode/consent permits raw-audio
        capture/retention (Founder Rule 2)."""
        ...

    def requesting_party_is_first_party(self) -> bool:
        """True if the requester is Chronis itself (internal), False
        for any third party (partner/family/other) — Founder Rule 4 is
        deny-by-default for anyone who is NOT first-party."""
        ...

    def third_party_has_explicit_grant(self) -> bool:
        """True only if the user has explicitly granted this specific
        third party access via the constitutional permission system
        (Founder Rule 4's only exception to deny-by-default)."""
        ...


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    decision: AccessDecision
    reason: str
    rule: str
    """Which founder rule (1-4) this decision was made under, for
    audit/explainability."""


def evaluate_access(
    classification: DataClassification,
    context: ConsentContext,
) -> PolicyDecision:
    """Evaluate the 4 founder-approved rules against one classified
    object and one consent context. This is the single entry point the
    real constitutional system should call once wired in.
    """

    # Founder Rule 4 (checked first: deny-by-default for third parties,
    # regardless of what the object is): third-party access is denied
    # unless an explicit grant exists.
    if not context.requesting_party_is_first_party():
        if context.third_party_has_explicit_grant():
            return PolicyDecision(
                decision=AccessDecision.ALLOW,
                reason="third party has an explicit user-granted permission",
                rule="4",
            )
        return PolicyDecision(
            decision=AccessDecision.DENY,
            reason="third-party access is deny-by-default; no explicit grant present",
            rule="4",
        )

    # Founder Rule 3: voice fingerprints/biometric features are
    # internal-use only, and this function's ALLOW here only covers
    # first-party internal use (third-party access was already handled
    # by Rule 4 above, which would have denied it before reaching here
    # for a biometric object with no explicit grant).
    if classification.object_type is ObjectType.BIOMETRIC_FEATURES:
        return PolicyDecision(
            decision=AccessDecision.ALLOW,
            reason=(
                "voice fingerprint/biometric data may be used internally to "
                "recognize a recurring opaque speaker; no automatic real-world "
                "name or public/family-facing identity inference is authorized "
                "by this decision alone"
            ),
            rule="3",
        )

    # Founder Rule 2: raw audio storage/processing requires the user's
    # raw-audio-retention consent specifically.
    if classification.object_type is ObjectType.RAW_AUDIO:
        if context.user_permits_raw_audio_retention():
            return PolicyDecision(
                decision=AccessDecision.ALLOW,
                reason="user consent permits raw-audio capture/retention",
                rule="2",
            )
        return PolicyDecision(
            decision=AccessDecision.DENY,
            reason="raw-audio retention consent not granted",
            rule="2",
        )

    # Founder Rule 1: derived voice/prosody features require the
    # user's audio-processing consent.
    if classification.object_type in (ObjectType.PROSODY_FEATURES, ObjectType.TRANSCRIPT):
        if context.user_permits_audio_processing():
            return PolicyDecision(
                decision=AccessDecision.ALLOW,
                reason="user consent permits the relevant audio processing",
                rule="1",
            )
        return PolicyDecision(
            decision=AccessDecision.DENY,
            reason="audio-processing consent not granted",
            rule="1",
        )

    # Non-audio-family classifications (physiological/motion features
    # etc.) are outside these 4 founder rules' scope entirely — this
    # policy module makes no claim about them one way or the other.
    return PolicyDecision(
        decision=AccessDecision.ALLOW,
        reason="classification is outside the scope of the 4 founder audio/voice rules",
        rule="n/a",
    )
