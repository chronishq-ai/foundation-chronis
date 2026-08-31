"""
claims_engine/claim_levels.py

Sprint 9, Day 25 — Level 0-3 claim types and their hard admissibility gates.

Hard rule (directive, restated in code): if a Level 3 claim fails even one of
its five conditions, NOTHING surfaces — not a softened version, nothing.
This is implemented as `evaluate_level3_gates` returning either a fully-passed
gate result or a definitive rejection; there is no partial/soft path anywhere
in this module.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Sequence
import uuid

from upstream_interfaces import AttractorRecord, Domain
from divergence_engine.state import DivergenceState


class ClaimLevel(Enum):
    LEVEL_0 = 0  # event fact — always surfaceable
    LEVEL_1 = 1  # behavioral pattern — all 3 attractor conditions
    LEVEL_2 = 2  # disposition — convergent Level-1 + shared latent driver + domain confidence
    LEVEL_3 = 3  # identity claim — all 5 hard gates simultaneously


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class GateEvaluation:
    level: ClaimLevel
    admissible: bool
    checks: Sequence[GateCheck]

    def failed_checks(self) -> List[GateCheck]:
        return [c for c in self.checks if not c.passed]


@dataclass(frozen=True)
class Claim:
    """
    A single, typed, provenance-carrying claim. Append-only — corrections
    (Sprint 17 "Teach Chronis") are counter-annotations, never overwrites (G2).
    """
    claim_id: str
    user_id: str
    domain_id: str
    level: ClaimLevel
    dominant_divergence_type: Optional[str]  # None for Level 0/1 claims
    gate_evaluation: GateEvaluation
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_dual_structured: bool = False  # genuinely split (phase-split/context-split) identity pattern
    dual_structure_components: Optional[Sequence["Claim"]] = None

    @staticmethod
    def new(
        user_id: str,
        domain_id: str,
        level: ClaimLevel,
        gate_evaluation: GateEvaluation,
        dominant_divergence_type: Optional[str] = None,
        is_dual_structured: bool = False,
        dual_structure_components: Optional[Sequence["Claim"]] = None,
    ) -> "Claim":
        return Claim(
            claim_id=str(uuid.uuid4()),
            user_id=user_id,
            domain_id=domain_id,
            level=level,
            dominant_divergence_type=dominant_divergence_type,
            gate_evaluation=gate_evaluation,
            is_dual_structured=is_dual_structured,
            dual_structure_components=dual_structure_components,
        )


# ---------------------------------------------------------------------------
# Level 0 — event fact. Always surfaceable; no gate to fail.
# ---------------------------------------------------------------------------

def evaluate_level0(event_exists: bool) -> GateEvaluation:
    check = GateCheck("event_recorded", event_exists, "Level 0 facts are always surfaceable once recorded.")
    return GateEvaluation(level=ClaimLevel.LEVEL_0, admissible=event_exists, checks=[check])


# ---------------------------------------------------------------------------
# Level 1 — behavioral pattern. All 3 attractor conditions (hard AND, from
# Sprint 4): revisit_count, mean_dwell_time, transition_stability.
# ---------------------------------------------------------------------------

def evaluate_level1(attractor: AttractorRecord) -> GateEvaluation:
    checks = [
        GateCheck("attractor_declared", attractor.declared,
                  "Sprint 4's hard AND over revisit_count/mean_dwell_time/transition_stability."),
    ]
    admissible = attractor.declared
    return GateEvaluation(level=ClaimLevel.LEVEL_1, admissible=admissible, checks=checks)


# ---------------------------------------------------------------------------
# Level 2 — disposition. Convergent Level-1 patterns + shared latent driver
# (Sprint 8's co-occupancy+Granger test, power-gated) + domain confidence.
# ---------------------------------------------------------------------------

DOMAIN_CONFIDENCE_FLOOR = 0.5  # person-calibrated in production; explicit provisional constant here.


def evaluate_level2(
    level1_evaluations: Sequence[GateEvaluation],
    divergence_state: DivergenceState,
    domain: Domain,
) -> GateEvaluation:
    convergent_level1 = all(e.admissible for e in level1_evaluations) and len(level1_evaluations) >= 1
    power_gate_passed = divergence_state.provenance.power_gate_passed
    shared_latent_driver = power_gate_passed and (divergence_state.type_scores.dominant() is not None)
    domain_confidence_ok = domain.confidence >= DOMAIN_CONFIDENCE_FLOOR

    checks = [
        GateCheck("convergent_level1_patterns", convergent_level1),
        GateCheck("power_gate_passed_mp09", power_gate_passed,
                  "MP-09: below 20 sessions/regime, Granger never ran -> no Level 2, no exceptions."),
        GateCheck("shared_latent_driver_detected", shared_latent_driver),
        GateCheck("domain_confidence_floor", domain_confidence_ok, f">= {DOMAIN_CONFIDENCE_FLOOR}"),
    ]
    admissible = all(c.passed for c in checks)
    return GateEvaluation(level=ClaimLevel.LEVEL_2, admissible=admissible, checks=checks)


# ---------------------------------------------------------------------------
# Level 3 — identity claim. ALL 5 hard gates simultaneously. Fail even one ->
# nothing surfaces, not a softened version.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Level3Inputs:
    level2_evaluation: GateEvaluation
    divergence_state: DivergenceState
    cross_phase_survival: bool           # domain survived >=1 detected phase transition
    n_self_reflection_sessions_last_30d: int  # for self-protection-type reflective-engagement gate
    no_contradiction_without_new_evidence: bool  # per Day 26 surfacing policy
    six_month_human_review_clear: bool   # standing mandatory human-review requirement for Level 3 text


def evaluate_level3_gates(inputs: Level3Inputs) -> GateEvaluation:
    dominant_type = inputs.divergence_state.type_scores.dominant()

    gate1 = GateCheck("level2_admissible", inputs.level2_evaluation.admissible)
    gate2 = GateCheck(
        "divergence_type_unambiguous",
        dominant_type is not None,
        "Two type scores within 0.15 of each other -> never promotes to Level 3.",
    )
    gate3 = GateCheck("cross_phase_survival", inputs.cross_phase_survival)

    # Reflective-engagement safety gate: self-protection claims require >=3
    # self-reflection-mode sessions in the past 30 days before surfacing at all.
    if dominant_type == "self_protection":
        reflective_gate_passed = inputs.n_self_reflection_sessions_last_30d >= 3
    else:
        reflective_gate_passed = True
    gate4 = GateCheck(
        "reflective_engagement_safety_gate",
        reflective_gate_passed,
        ">=3 self-reflection-mode sessions/30d required for self-protection claims.",
    )

    gate5 = GateCheck(
        "no_contradiction_without_new_evidence_and_review_clear",
        inputs.no_contradiction_without_new_evidence and inputs.six_month_human_review_clear,
    )

    checks = [gate1, gate2, gate3, gate4, gate5]
    admissible = all(c.passed for c in checks)  # hard AND across all five, no exceptions.

    return GateEvaluation(level=ClaimLevel.LEVEL_3, admissible=admissible, checks=checks)
