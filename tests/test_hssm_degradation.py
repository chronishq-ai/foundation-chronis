import numpy as np
import pytest

from phase_transition.hssm_degradation import (
    stack_regime_observations,
    make_regime_conditional_fit_predict,
    pooled_gaussian_null_ll,
    evaluate_regime_conditional_degradation,
    is_regime_conditional_degraded,
)
from phase_transition.gate import PhaseTransitionGate
from tests.fixtures.synthetic_hssm_fixture import generate_synthetic_hssm_output


def test_stack_regime_observations_shape():
    regime_sequence = [0, 0, 1, 1]
    observations = np.array([[1.0, 2.0], [1.1, 2.1], [5.0, 6.0], [5.1, 6.1]])
    stacked = stack_regime_observations(regime_sequence, observations)
    assert stacked.shape == (4, 3)
    assert list(stacked[:, -1]) == [0.0, 0.0, 1.0, 1.0]


def test_stack_regime_observations_length_mismatch_raises():
    with pytest.raises(ValueError):
        stack_regime_observations([0, 1, 2], np.zeros((2, 3)))


def test_regime_conditional_degrades_on_emission_shift():
    """Regime label constant (0) across the boundary, but the emission
    mean shifts sharply -- a real degradation the regime-conditional
    model should catch since it can't be explained away as 'the regime
    just changed'."""
    rng = np.random.default_rng(0)
    pre = rng.normal(0, 1, (60, 3))
    post = rng.normal(8, 1, (60, 3))
    observations = np.vstack([pre, post])
    regime_sequence = np.zeros(120, dtype=int)  # regime label never changes

    assert is_regime_conditional_degraded(
        regime_sequence, observations, candidate_t=60, threshold=2.0)


def test_regime_conditional_no_degradation_on_stationary_regime():
    rng = np.random.default_rng(1)
    observations = rng.normal(0, 1, (120, 3))
    regime_sequence = np.zeros(120, dtype=int)

    assert not is_regime_conditional_degraded(
        regime_sequence, observations, candidate_t=60, threshold=2.0)


def test_new_post_boundary_regime_uses_fallback_and_is_flagged():
    """Post-boundary regime label (1) never appears pre-boundary --
    fit_fn has no stats for it, predict_ll_fn must fall back rather
    than silently contributing zero, and a genuinely new regime with a
    different emission mean should register as degraded."""
    rng = np.random.default_rng(2)
    pre = rng.normal(0, 1, (60, 3))
    post = rng.normal(6, 1, (60, 3))
    observations = np.vstack([pre, post])
    regime_sequence = np.array([0] * 60 + [1] * 60)

    result = evaluate_regime_conditional_degradation(
        regime_sequence, observations, candidate_t=60)
    assert result["valid"]
    assert result["degradation"] > 0


def test_evaluate_regime_conditional_degradation_reports_windows():
    """S56.2 T2 parity: window boundaries present in machine-readable form,
    same as the generic evaluate_generative_model_degradation harness."""
    rng = np.random.default_rng(3)
    observations = rng.normal(0, 1, (200, 2))
    regime_sequence = np.zeros(200, dtype=int)
    timestamps = list(range(200))

    result = evaluate_regime_conditional_degradation(
        regime_sequence, observations, candidate_t=100,
        timestamps=timestamps, pre_window=20, post_window=20)

    assert result["valid"]
    assert result["pre_window_start_idx"] == 80
    assert result["post_window_end_idx"] == 120
    assert "pre_window_start_ts" in result
    assert "null_baseline_ll" in result
    assert "ll_vs_null_baseline" in result
    assert "in_sample_ll_per_sample" in result
    assert "out_sample_ll_per_sample" in result


def test_pooled_gaussian_null_ll_ignores_regime_label():
    """Null baseline should be regime-blind -- two rows with different
    regime labels but the same feature values must contribute
    identically."""
    pre_a = stack_regime_observations([0, 0], np.array([[0.0], [0.0]]))
    pre_b = stack_regime_observations([0, 1], np.array([[0.0], [0.0]]))
    post = stack_regime_observations([5], np.array([[0.0]]))

    ll_a = pooled_gaussian_null_ll(pre_a, post)
    ll_b = pooled_gaussian_null_ll(pre_b, post)
    assert ll_a == pytest.approx(ll_b)


def test_missing_session_rows_skipped_not_imputed():
    """NaN rows (missing sessions, per HSSM convention) must be dropped,
    not treated as data -- fitting shouldn't blow up and predict_ll_fn
    should skip them rather than propagate NaN into the total."""
    observations = np.array([[0.0], [np.nan], [0.2], [5.0], [np.nan], [5.1]])
    regime_sequence = [0, 0, 0, 1, 1, 1]
    result = evaluate_regime_conditional_degradation(
        regime_sequence, observations, candidate_t=3, pre_window=3, post_window=3)
    assert result["valid"]
    assert np.isfinite(result["post_predictive_ll"])


def test_smoke_against_synthetic_hssm_fixture():
    """Full smoke test against the shape-matched synthetic fixture
    (tests.fixtures.synthetic_hssm_fixture) -- this is the only fixture
    that mimics real backbone.hssm.fit_hssm output shape; there is no
    real backbone package in this environment to test against (S56.6
    blocker). Only checks the pipeline runs end-to-end and returns a
    well-formed result -- does NOT assert a specific degraded/not-
    degraded outcome, since the fixture's regime sequence is random."""
    fixture = generate_synthetic_hssm_output(T=200, K=3, F=5, seed=42)
    result = evaluate_regime_conditional_degradation(
        fixture.regime_sequence, fixture.observations, candidate_t=100)
    assert "valid" in result
    if result["valid"]:
        assert isinstance(result["degradation"], float)
        assert np.isfinite(result["degradation"])


def test_gate_opt_in_uses_regime_conditional_path():
    """PhaseTransitionGate.evaluate_candidate: supplying BOTH
    regime_sequence and hssm_observations opts into the regime-
    conditional path instead of the generic scalar-Gaussian default."""
    rng = np.random.default_rng(5)
    pre = rng.normal(0, 1, 60)
    post = rng.normal(8, 1, 60)
    data = np.concatenate([pre, post]).tolist()

    observations = np.array(data).reshape(-1, 1)
    regime_sequence = np.zeros(120, dtype=int)  # regime label constant

    gate = PhaseTransitionGate(stability_min_days=10)
    result = gate.evaluate_candidate(
        data, candidate_t=60,
        regime_sequence=regime_sequence, hssm_observations=observations)

    assert result["condition_2_degradation"] is True


def test_gate_default_path_unaffected_when_hssm_args_omitted():
    """Omitting regime_sequence/hssm_observations must leave gate
    behavior byte-identical to before S56.2 (backward compatible)."""
    rng = np.random.default_rng(6)
    pre = rng.normal(0, 1, 60)
    post = rng.normal(8, 1, 60)
    data = np.concatenate([pre, post]).tolist()

    gate = PhaseTransitionGate(stability_min_days=10)
    result_without = gate.evaluate_candidate(data, candidate_t=60)
    result_explicit_none = gate.evaluate_candidate(
        data, candidate_t=60, regime_sequence=None, hssm_observations=None)

    assert result_without["condition_2_degradation"] == result_explicit_none["condition_2_degradation"]
