"""
divergence_engine/state.py

Sprint 8, Day 22 — Alignment score, the four type-score formulas, and the
DivergenceState object.

Per the directive: DivergenceState is a canonical-record-compliant, APPEND-ONLY
entity. Nothing here ever mutates a prior DivergenceState in place — a new
observation window produces a new DivergenceState, chained via `previous_state_id`.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import uuid


AMBIGUITY_THRESHOLD = 0.15  # MP-05: explicit, provisional hyperparameter — never finalized silently.


@dataclass(frozen=True)
class TypeScores:
    """The four divergence type scores. Each in [0, 1]."""
    ignorance: float
    aspiration: float
    self_protection: float
    active_transition: float

    def dominant(self) -> Optional[str]:
        """
        Returns the dominant type name, or None if the top two scores are within
        AMBIGUITY_THRESHOLD of each other (Sprint 8 Day 24: never force a
        classification in that case).
        """
        scored = sorted(
            [
                ("ignorance", self.ignorance),
                ("aspiration", self.aspiration),
                ("self_protection", self.self_protection),
                ("active_transition", self.active_transition),
            ],
            key=lambda kv: kv[1],
            reverse=True,
        )
        top_name, top_val = scored[0]
        second_name, second_val = scored[1]
        if (top_val - second_val) < AMBIGUITY_THRESHOLD:
            return None
        return top_name


@dataclass(frozen=True)
class Provenance:
    """Every number in a DivergenceState must be traceable to how it was computed."""
    fisher_p_value: Optional[float]
    fisher_bonferroni_alpha: Optional[float]
    granger_f_stat: Optional[float]
    granger_p_value: Optional[float]
    granger_bonferroni_alpha: Optional[float]
    lag_order: Optional[int]
    n_behavioral_sessions_in_regime: int
    n_narrative_sessions_in_regime: int
    power_gate_passed: bool  # MP-09: 20-session-per-regime gate
    computed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class DivergenceState:
    """
    Append-only. A new DivergenceState is created per (user, domain, observation
    window) — never mutated. `previous_state_id` chains history.
    """
    state_id: str
    user_id: str
    domain_id: str
    observation_window_start: datetime
    observation_window_end: datetime
    type_scores: TypeScores
    confidence: float  # sourced from Sprint 7's conformal-calibrated narrative uncertainty
    provenance: Provenance
    previous_state_id: Optional[str] = None

    @staticmethod
    def new(
        user_id: str,
        domain_id: str,
        window_start: datetime,
        window_end: datetime,
        type_scores: TypeScores,
        confidence: float,
        provenance: Provenance,
        previous_state_id: Optional[str] = None,
    ) -> "DivergenceState":
        return DivergenceState(
            state_id=str(uuid.uuid4()),
            user_id=user_id,
            domain_id=domain_id,
            observation_window_start=window_start,
            observation_window_end=window_end,
            type_scores=type_scores,
            confidence=confidence,
            provenance=provenance,
            previous_state_id=previous_state_id,
        )


def compute_type_scores(
    *,
    ignorance_evidence: float,
    aspiration_evidence: float,
    self_protection_evidence: float,
    active_transition_evidence: float,
    claim_influence_discount: float = 0.0,
) -> TypeScores:
    """
    Combines raw per-type evidence (already computed by the co-occupancy +
    Granger conditions in cooccupancy.py / granger.py) into normalized [0,1]
    type scores.

    `claim_influence_discount` in [0,1]: Sprint 15's Observer-Effect safeguard
    reduces evidence weight for changes flagged `potentially_claim_influenced`.
    Wired here as a pass-through hook so Sprint 15 can plug in without touching
    this function's signature.
    """
    raw = {
        "ignorance": max(0.0, ignorance_evidence * (1 - claim_influence_discount)),
        "aspiration": max(0.0, aspiration_evidence * (1 - claim_influence_discount)),
        "self_protection": max(0.0, self_protection_evidence * (1 - claim_influence_discount)),
        "active_transition": max(0.0, active_transition_evidence * (1 - claim_influence_discount)),
    }
    total = sum(raw.values())
    if total <= 0:
        return TypeScores(0.0, 0.0, 0.0, 0.0)
    return TypeScores(**{k: v / total for k, v in raw.items()})
