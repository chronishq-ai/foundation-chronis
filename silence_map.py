"""
Sprint 11 - Silence Map.

Classifies silence using turn-taking and physiological co-signals.
This module is a composition layer; it does not train a new model.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional, Sequence


class SilenceMapError(ValueError):
    """Invalid Silence Map input."""


@dataclass(frozen=True)
class SilenceInput:
    user_id: str
    silence_seconds: float
    turn_expected: bool
    typing_active: bool
    physiological_delta: float


@dataclass(frozen=True)
class SilenceResult:
    classification: str
    confidence: float
    explanation: str


class SilenceMap:
    """
    Classifies three Sprint 11 silence categories:

        attentive
        avoidant
        conversational

    Evidence is deliberately kept interpretable.
    """

    def classify(self, sample: SilenceInput) -> SilenceResult:
        self._validate(sample)

        if sample.turn_expected and not sample.typing_active:
            if sample.physiological_delta > 0.5:
                classification = "avoidant"
                confidence = 0.85
                explanation = (
                    "Expected turn was not taken and physiological "
                    "co-signal indicates increased activation."
                )
            else:
                classification = "conversational"
                confidence = 0.75
                explanation = (
                    "An expected conversational turn was silent "
                    "without a strong physiological activation signal."
                )

        elif sample.typing_active and not sample.turn_expected:
            classification = "attentive"
            confidence = 0.85
            explanation = (
                "Typing activity continues while no conversational "
                "turn is expected."
            )

        elif sample.typing_active:
            classification = "attentive"
            confidence = 0.75
            explanation = (
                "Active typing provides evidence of continued "
                "engagement during the silent interval."
            )

        else:
            classification = "conversational"
            confidence = 0.60
            explanation = (
                "Silence is present without sufficient evidence "
                "for attentive or avoidant behavior."
            )

        return SilenceResult(
            classification=classification,
            confidence=max(0.0, min(1.0, confidence)),
            explanation=explanation,
        )

    @staticmethod
    def classify_many(
        samples: Sequence[SilenceInput],
    ) -> list[SilenceResult]:
        engine = SilenceMap()
        return [engine.classify(sample) for sample in samples]

    @staticmethod
    def _validate(sample: SilenceInput) -> None:
        if not sample.user_id:
            raise SilenceMapError("user_id must not be empty")

        if not isinstance(sample.silence_seconds, (int, float)):
            raise SilenceMapError(
                "silence_seconds must be numeric"
            )

        if not isfinite(sample.silence_seconds):
            raise SilenceMapError(
                "silence_seconds must be finite"
            )

        if sample.silence_seconds <= 0:
            raise SilenceMapError(
                "silence_seconds must be greater than zero"
            )

        if not isinstance(sample.turn_expected, bool):
            raise SilenceMapError(
                "turn_expected must be boolean"
            )

        if not isinstance(sample.typing_active, bool):
            raise SilenceMapError(
                "typing_active must be boolean"
            )

        if not isinstance(
            sample.physiological_delta,
            (int, float),
        ):
            raise SilenceMapError(
                "physiological_delta must be numeric"
            )

        if not isfinite(sample.physiological_delta):
            raise SilenceMapError(
                "physiological_delta must be finite"
            )