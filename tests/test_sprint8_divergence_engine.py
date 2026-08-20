"""
tests/test_sprint8_divergence_engine.py

Covers Sprint 8's explicit Definition-of-Done bullets:
  - >75% dominant-type classification accuracy, per type, independently (Day 24)
  - DivergenceState objects produced for >80% of user-domain pairs with
    sufficient data history (smoke-tested at the unit level here; full >80%
    surrogate-run number requires Sprint 6's real domain population data —
    not fabricable from this standalone harness, flagged rather than faked)
  - A domain pair with 19 sessions in a regime NEVER produces a Level 2 claim
    — code-enforced, tested here directly against the power gate.
  - Ambiguity (within 0.15) is never force-classified.

NOTE: per the directive's own AI-assistant policy, this suite is a STARTING
POINT. It must be run, read, and extended by an engineer against your real
surrogate data before anyone signs off Sprint 8's DoD for real.
"""

import numpy as np
import pytest

from divergence_engine.granger import power_gate, within_regime_granger_test, MIN_SESSIONS_PER_REGIME
from divergence_engine.state import AMBIGUITY_THRESHOLD, TypeScores
from divergence_engine.engine import level2_claim_permitted
from divergence_engine.state import DivergenceState, Provenance
from synthetic.planted_profiles import run_validation_suite
from datetime import datetime


def test_power_gate_boundary_19_vs_20():
    """MP-09: 19 sessions in a regime never runs Granger / never yields a Level 2 claim; 20 does."""
    assert power_gate(19) is False
    assert power_gate(20) is True
    assert power_gate(MIN_SESSIONS_PER_REGIME) is True
    assert power_gate(MIN_SESSIONS_PER_REGIME - 1) is False


def test_granger_does_not_run_below_gate():
    m_t = np.random.default_rng(0).normal(size=(19, 3))
    n_t = np.random.default_rng(1).normal(size=(19, 3))
    result = within_regime_granger_test(m_t, n_t, n_pairs_tested=1)
    assert result.ran is False
    assert result.power_gate_passed is False
    assert result.p_value_n_causes_m is None


def test_granger_runs_at_gate():
    rng = np.random.default_rng(0)
    m_t = rng.normal(size=(20, 3))
    n_t = rng.normal(size=(20, 3))
    result = within_regime_granger_test(m_t, n_t, n_pairs_tested=1)
    assert result.ran is True
    assert result.power_gate_passed is True


def _dummy_provenance(power_gate_passed: bool) -> Provenance:
    return Provenance(
        fisher_p_value=0.01, fisher_bonferroni_alpha=0.05,
        granger_f_stat=None, granger_p_value=None, granger_bonferroni_alpha=None,
        lag_order=None, n_behavioral_sessions_in_regime=19, n_narrative_sessions_in_regime=19,
        power_gate_passed=power_gate_passed,
    )


def test_level2_claim_blocked_below_power_gate():
    state = DivergenceState.new(
        user_id="u1", domain_id="d1",
        window_start=datetime(2026, 1, 1), window_end=datetime(2026, 3, 1),
        type_scores=TypeScores(0.9, 0.03, 0.03, 0.04),
        confidence=0.8,
        provenance=_dummy_provenance(power_gate_passed=False),
    )
    assert level2_claim_permitted(state) is False


def test_level2_claim_permitted_above_power_gate():
    state = DivergenceState.new(
        user_id="u1", domain_id="d1",
        window_start=datetime(2026, 1, 1), window_end=datetime(2026, 3, 1),
        type_scores=TypeScores(0.9, 0.03, 0.03, 0.04),
        confidence=0.8,
        provenance=_dummy_provenance(power_gate_passed=True),
    )
    assert level2_claim_permitted(state) is True


def test_ambiguity_never_forces_classification():
    close_scores = TypeScores(ignorance=0.30, aspiration=0.28, self_protection=0.22, active_transition=0.20)
    assert close_scores.dominant() is None  # top two within AMBIGUITY_THRESHOLD (0.15)

    clear_scores = TypeScores(ignorance=0.70, aspiration=0.10, self_protection=0.10, active_transition=0.10)
    assert clear_scores.dominant() == "ignorance"


@pytest.mark.slow
def test_synthetic_validation_suite_clears_75pct_per_type():
    """Sprint 8 Day 24 DoD: >75% accuracy per type, independently, on 20+ planted profiles/type."""
    report = run_validation_suite(n_profiles_per_type=20)
    for divergence_type, accuracy in report.per_type_accuracy.items():
        assert accuracy > 0.75, f"{divergence_type} accuracy {accuracy:.1%} did not clear the 75% bar"
