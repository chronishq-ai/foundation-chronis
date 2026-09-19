"""Tests for Ticket VF-16: L-BFGS-B duration optimizer fallback tracking, warning emission,
and HSSMResult visibility contract.
"""

from __future__ import annotations
import warnings
import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import backbone.hssm.model
from backbone.hssm.model import GaussianHSMM, KimHSSMModel, FitState
from backbone.hssm.fitting import fit_hssm, fit_with_random_restarts, HSSMResult
from backbone.hssm.label_switching import canonicalize_labels


def _make_structured_data(seed: int = 42) -> np.ndarray:
    """Generates structured 2-regime observations with distinct means and dwell times."""
    rng = np.random.default_rng(seed)
    regime_0 = rng.normal(loc=0.0, scale=0.5, size=(50, 2))
    regime_1 = rng.normal(loc=5.0, scale=0.5, size=(50, 2))
    return np.vstack([regime_0, regime_1, regime_0, regime_1])


def test_t1_forced_optimizer_failure_sets_fallback_flags_and_emits_warning():
    """T1 (VF-16): Construct a case where the L-BFGS-B optimizer fails.
    Confirm:
      - Warning is emitted identifying the regime and failure reason.
      - model.duration_fallback_used_ is True.
      - model.duration_optimizer_success_[k] is False for failing regime.
      - Fallback is recorded in HSSMResult.convergence_metadata, duration_info, and properties.
    """
    X = _make_structured_data(seed=42)

    orig_minimize = backbone.hssm.model.minimize

    # Rig minimize to fail on regime 1
    call_count = {"n": 0}

    def failing_minimize(fun, x0, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] % 2 == 0:
            # Simulate optimizer non-convergence / line search failure
            return OptimizeResult(
                x=np.array([x0[0] + 0.1, x0[1] + 0.1]),
                success=False,
                status=1,
                message="ABNORMAL_TERMINATION_IN_LNSRCH: Iteration limit or line search failure",
                fun=1e5,
            )
        return orig_minimize(fun, x0, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(backbone.hssm.model, "minimize", failing_minimize)
        with pytest.warns(UserWarning, match=r"\[GaussianHSMM\] Duration optimization fallback used for regime") as record:
            res = fit_hssm(X, candidate_ks=(2,), n_initializations=10, max_iter=20, random_seed=42)

    # 1. Warning validation
    assert len(record) > 0, "A UserWarning must be emitted when duration optimizer falls back"
    warning_text = str(record[0].message)
    assert "ABNORMAL_TERMINATION_IN_LNSRCH" in warning_text or "did not converge" in warning_text

    # 2. GaussianHSMM model attributes
    model = res.model
    assert model.duration_fallback_used_ is True
    assert isinstance(model.duration_optimizer_success_, dict)
    assert False in model.duration_optimizer_success_.values()

    # 3. HSSMResult contract and properties
    assert isinstance(res, HSSMResult)
    assert res.duration_fallback_used is True
    assert res.duration_optimizer_success is False
    assert isinstance(res.per_regime_optimizer_status, dict)
    assert False in res.per_regime_optimizer_status.values()

    # 4. convergence_metadata contract
    assert res.convergence_metadata["duration_fallback_used"] is True
    assert res.convergence_metadata["duration_optimizer_success"] is False
    assert res.convergence_metadata["per_regime_optimizer_status"] == model.duration_optimizer_success_

    # 5. duration_info contract
    assert res.duration_info["duration_fallback_used"] is True
    assert res.duration_info["duration_optimizer_success"] == model.duration_optimizer_success_

    # 6. run_log contract in convergence_metadata
    run_log = res.convergence_metadata.get("run_log", [])
    assert len(run_log) >= 1
    assert any(r.get("duration_fallback_used") is True for r in run_log)


def test_t1_direct_m_step_fallback_verification():
    """T1 sub-case: Direct verification of GaussianHSMM._m_step fallback mechanics."""
    model = GaussianHSMM(n_regimes=2, n_features=2, max_duration=10, seed=0)
    X = np.ones((10, 2))
    regime_post = np.full((10, 2), 0.5)
    entry_post = np.full((10, 2, 10), 0.05)
    xi_counts = np.ones((2, 2))

    model.pi = np.full(2, 0.5)
    model.mu = np.zeros((2, 2))
    model.var = np.ones((2, 2))
    model.dur_mu = np.array([1.0, 1.0])
    model.dur_sigma = np.array([0.5, 0.5])

    def force_fail_minimize(fun, x0, *args, **kwargs):
        return OptimizeResult(
            x=np.array([99.0, 99.0]),
            success=False,
            status=1,
            message="FORCED_TEST_FAILURE",
            fun=1e5,
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(backbone.hssm.model, "minimize", force_fail_minimize)
        with pytest.warns(UserWarning, match="FORCED_TEST_FAILURE"):
            model._m_step(X, regime_post, entry_post, xi_counts)

    assert model.duration_fallback_used_ is True
    assert model.duration_optimizer_success_[0] is False
    assert model.duration_optimizer_success_[1] is False
    assert "FORCED_TEST_FAILURE" in model.duration_optimizer_messages_[0]
    # Verify fallback to init_mu and init_sigma (not 99.0)
    assert model.dur_mu[0] != 99.0
    assert model.dur_sigma[0] != 99.0


def test_t2_normal_fit_records_optimizer_success_with_no_fallback():
    """T2 (VF-16): Construct a normal case where the optimizer succeeds.
    Confirm:
      - No fallback warnings are emitted.
      - model.duration_fallback_used_ is False.
      - model.duration_optimizer_success_ is True for all regimes.
      - HSSMResult flags correctly show success and no fallback.
    """
    X = _make_structured_data(seed=123)

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        res = fit_hssm(X, candidate_ks=(2,), n_initializations=10, max_iter=30, random_seed=123)

    # Confirm no duration fallback warnings were emitted
    fallback_warnings = [
        w for w in recorded_warnings
        if "Duration optimization fallback used" in str(w.message)
    ]
    assert len(fallback_warnings) == 0

    model = res.model
    assert model.duration_fallback_used_ is False
    assert all(model.duration_optimizer_success_.values())
    assert len(model.duration_optimizer_success_) == 2

    # Verify HSSMResult properties and metadata
    assert res.duration_fallback_used is False
    assert res.duration_optimizer_success is True
    assert res.per_regime_optimizer_status == {0: True, 1: True}
    assert res.convergence_metadata["duration_fallback_used"] is False
    assert res.convergence_metadata["duration_optimizer_success"] is True
    assert res.duration_info["duration_fallback_used"] is False
    assert res.duration_info["duration_optimizer_success"] == {0: True, 1: True}


def test_t3_label_canonicalization_permutes_duration_optimizer_status():
    """T3 (VF-16): Confirm canonicalize_labels permutes duration optimizer status
    dictionaries aligned to the new regime ordering.
    """
    model = GaussianHSMM(n_regimes=3, n_features=2, seed=0)
    # Set artificial parameters where regime 2 has lowest activity norm, regime 0 has highest
    model.mu = np.array([[10.0, 10.0], [5.0, 5.0], [1.0, 1.0]])  # Norms: ~14.1, ~7.07, ~1.41
    model.pi = np.array([0.2, 0.3, 0.5])
    model.A = np.eye(3)
    model.var = np.ones((3, 2))
    model.dur_mu = np.array([1.0, 2.0, 3.0])
    model.dur_sigma = np.array([0.5, 0.5, 0.5])
    model._is_fitted = True

    # Regime 0 (highest norm) had fallback, others succeeded
    model.duration_optimizer_success_ = {0: False, 1: True, 2: True}
    model.duration_optimizer_messages_ = {0: "Failed", 1: "Converged", 2: "Converged"}

    canonical_model = canonicalize_labels(model)

    # Ascending order of norms is [2, 1, 0]
    assert canonical_model._label_order_applied == [2, 1, 0]
    # New regime 0 was old 2 (True), new 1 was old 1 (True), new 2 was old 0 (False)
    assert canonical_model.duration_optimizer_success_[0] is True
    assert canonical_model.duration_optimizer_success_[1] is True
    assert canonical_model.duration_optimizer_success_[2] is False
    assert canonical_model.duration_optimizer_messages_[2] == "Failed"


def test_t3_kim_model_initializes_duration_optimizer_tracking():
    """T3 (VF-16): Confirm KimHSSMModel initializes tracking attributes correctly."""
    kim = KimHSSMModel(
        n_regimes=2,
        n_features=2,
        transition_matrix=np.array([[0.8, 0.2], [0.3, 0.7]]),
        emission_means=np.array([[0.0, 0.0], [2.0, 2.0]]),
        emission_covariances=[np.eye(2), np.eye(2)],
        duration_mu=np.array([1.0, 1.2]),
        duration_sigma=np.array([0.4, 0.5]),
    )
    assert kim.duration_fallback_used_ is False
    assert kim.duration_optimizer_success_ == {0: True, 1: True}
    assert 0 in kim.duration_optimizer_messages_
