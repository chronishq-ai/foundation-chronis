"""
divergence_engine/engine.py

Sprint 8 — main entry point. Wires Day 22 (state/type-scores), Day 23
(co-occupancy + Granger conditions) into a single `compute_divergence_state`
call, and produces the append-only DivergenceState.

Evidence-to-type mapping (per Bible Part 5.5-5.7, restated here for
traceability — the formal weighting is a product decision your team should
confirm against the Bible's exact coefficients before shipping):

  Ignorance (I):        strong behavioral attractor + LOW/absent narrative
                         engagement in that domain (no significant co-occupancy,
                         narrative regime near-silent).
  Aspiration (Asp):     WEAKENING behavioral attractor + narrative regime shows
                         future-tense/agentic content not yet behaviorally
                         realized (narrative "leads", low behavioral co-occupancy,
                         but narrative fast-state trending toward the domain).
  Self-Protection:      STABLE behavioral attractor + avoidant/passive
                         narrative regime (co-occupancy present but narrative
                         Granger-causes behavior weakly / narrative dampens).
  Active Transition:    BOTH systems shifting together, with a recoverable
                         lag between them (bidirectional or lagged Granger
                         significance, both regimes non-stationary in window).
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple
import numpy as np

from .state import DivergenceState, Provenance, TypeScores, compute_type_scores
from .cooccupancy import fisher_cooccupancy_test, CooccupancyResult
from .granger import within_regime_granger_test, GrangerResult


@dataclass(frozen=True)
class DivergenceInputs:
    user_id: str
    domain_id: str
    window_start: datetime
    window_end: datetime
    p_t: np.ndarray            # behavioral regime labels, full window
    q_t: np.ndarray            # narrative regime labels, full window
    m_t: np.ndarray            # behavioral fast state, full window
    n_t: np.ndarray            # narrative fast state, full window
    behavioral_regime_id: int
    narrative_regime_id: int
    n_domain_pairs_tested: int  # for Bonferroni correction across all domain pairs
    behavioral_attractor_weakening: bool  # from Sprint 4/attractor trend
    narrative_conformal_confidence: float  # Sprint 7's calibrated confidence
    claim_influence_discount: float = 0.0  # Sprint 15 Observer-Effect hook


def _slice_to_regime(arr: np.ndarray, regime_labels: np.ndarray, regime_id: int) -> np.ndarray:
    mask = regime_labels == regime_id
    return arr[mask]


def _slice_to_joint_regime_window(
    m_t: np.ndarray, n_t: np.ndarray, p_t: np.ndarray, q_t: np.ndarray,
    behavioral_regime_id: int, narrative_regime_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sessions where BOTH systems are simultaneously in the regime pair under
    test, keeping m_t/n_t time-aligned (fixes the misalignment you'd get from
    independently masking each series and truncating to min length).
    """
    joint_mask = (p_t == behavioral_regime_id) & (q_t == narrative_regime_id)
    return m_t[joint_mask], n_t[joint_mask]


def compute_divergence_state(
    inputs: DivergenceInputs,
    previous_state_id: Optional[str] = None,
) -> DivergenceState:
    # --- Condition 1: regime co-occupancy ---
    cooc: CooccupancyResult = fisher_cooccupancy_test(
        inputs.p_t,
        inputs.q_t,
        inputs.behavioral_regime_id,
        inputs.narrative_regime_id,
        n_pairs_tested=inputs.n_domain_pairs_tested,
    )

    # --- Condition 2: within-regime Granger predictability (MP-09 gated) ---
    m_in_regime, n_in_regime = _slice_to_joint_regime_window(
        inputs.m_t, inputs.n_t, inputs.p_t, inputs.q_t,
        inputs.behavioral_regime_id, inputs.narrative_regime_id,
    )

    granger: GrangerResult = within_regime_granger_test(
        m_in_regime,
        n_in_regime,
        n_pairs_tested=inputs.n_domain_pairs_tested,
    )

    # --- Evidence -> type scores ---
    cooc_strength = 1.0 if cooc.significant else 0.0
    narrative_leads = granger.significant_n_causes_m if granger.ran else False
    behavioral_leads = granger.significant_m_causes_n if granger.ran else False
    bidirectional = narrative_leads and behavioral_leads

    ignorance_evidence = (1.0 - cooc_strength) * (1.0 if not granger.ran or not (narrative_leads or behavioral_leads) else 0.3)
    aspiration_evidence = (
        (1.0 if inputs.behavioral_attractor_weakening else 0.0)
        * (1.0 if narrative_leads and not behavioral_leads else 0.4)
    )
    self_protection_evidence = (
        cooc_strength
        * (1.0 if (not inputs.behavioral_attractor_weakening) and granger.ran and not bidirectional else 0.3)
    )
    active_transition_evidence = (
        (1.0 if inputs.behavioral_attractor_weakening else 0.5)
        * (1.0 if bidirectional else 0.2)
    )

    type_scores = compute_type_scores(
        ignorance_evidence=ignorance_evidence,
        aspiration_evidence=aspiration_evidence,
        self_protection_evidence=self_protection_evidence,
        active_transition_evidence=active_transition_evidence,
        claim_influence_discount=inputs.claim_influence_discount,
    )

    provenance = Provenance(
        fisher_p_value=cooc.p_value,
        fisher_bonferroni_alpha=cooc.bonferroni_alpha,
        granger_f_stat=granger.f_statistic_n_causes_m if granger.ran else None,
        granger_p_value=granger.p_value_n_causes_m if granger.ran else None,
        granger_bonferroni_alpha=granger.bonferroni_alpha,
        lag_order=granger.lag_order,
        n_behavioral_sessions_in_regime=len(m_in_regime),
        n_narrative_sessions_in_regime=len(n_in_regime),
        power_gate_passed=granger.power_gate_passed,
    )

    return DivergenceState.new(
        user_id=inputs.user_id,
        domain_id=inputs.domain_id,
        window_start=inputs.window_start,
        window_end=inputs.window_end,
        type_scores=type_scores,
        confidence=inputs.narrative_conformal_confidence,
        provenance=provenance,
        previous_state_id=previous_state_id,
    )


def level2_claim_permitted(state: DivergenceState) -> bool:
    """
    Hard rule (DoD): a domain pair below the 20-session power gate never
    produces a Level 2 claim. Code-enforced here, checked again independently
    in claims_engine (defense in depth, not redundancy-as-decoration).
    """
    return state.provenance.power_gate_passed
