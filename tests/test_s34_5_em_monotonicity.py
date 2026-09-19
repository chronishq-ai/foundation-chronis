import pytest
import numpy as np
from backbone.hssm.model import GaussianHSMM, FitState
from backbone.hssm.fitting import fit_with_random_restarts, FittingConvergenceError


def test_s34_5_t1_monotonicity_violation_triggers_rejected_update_and_preserves_history():
    """T1 (VF-15): Construct a scenario that triggers a genuine monotonicity violation.
    Confirm:
      - fit_state_ == FitState.REJECTED_UPDATE (not CONVERGED)
      - converged_ is False
      - log_likelihood_history_ retains BOTH the rejected value and the restored value as
        separate entries (not overwritten in place)
      - model.log_likelihood_ holds the restored value
    """
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3))
    # Seed 0 with 3 regimes on this data triggers a monotonicity decrease during EM
    model = GaussianHSMM(n_regimes=3, n_features=3, seed=0)
    model.fit(X, n_iter=50)

    assert model.fit_state_ == FitState.REJECTED_UPDATE
    assert model.converged_ is False
    assert len(model.log_likelihood_history_) >= 3

    rejected_cand = model.log_likelihood_history_[-2]
    restored_val = model.log_likelihood_history_[-1]
    prev_val = model.log_likelihood_history_[-3]

    # Candidate was a decrease
    assert rejected_cand < prev_val - 1e-4
    # Restored value matches the state prior to the candidate
    assert np.isclose(restored_val, prev_val)
    assert np.isclose(model.log_likelihood_, restored_val)
    # The history truthfully records the non-monotonic candidate dip and restoration
    assert not model.is_log_likelihood_monotonic(tol=1e-4)


def test_s34_5_t2_genuine_convergence_sets_converged_fit_state():
    """T2 (VF-15): Confirm a genuinely converged fit (no violations) still correctly
    gets fit_state_ == FitState.CONVERGED and converged_ == True."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3))
    # Seed 2 converges without monotonicity rejections
    model = GaussianHSMM(n_regimes=3, n_features=3, seed=2)
    model.fit(X, n_iter=50, tol=1e-3)

    assert model.fit_state_ == FitState.CONVERGED
    assert model.converged_ is True
    assert model.is_log_likelihood_monotonic(tol=1e-4)


def test_s34_5_t3_random_restarts_never_selects_rejected_update_or_failed():
    """T3 (VF-15): Confirm fit_with_random_restarts() never selects a REJECTED_UPDATE
    or FAILED run as the 'best' model, even if its logged likelihood looks numerically high."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(50, 3))

    real_fit = GaussianHSMM.fit
    call_count = {"n": 0}

    def rigged_fit(self, X, n_iter=100, tol=1e-4, verbose=False, timestamps=None):
        call_count["n"] += 1
        result = real_fit(self, X, n_iter=n_iter, tol=tol, verbose=verbose, timestamps=timestamps)
        if call_count["n"] == 2:
            # Rig this run to have REJECTED_UPDATE with an artificially high LL
            self.fit_state_ = FitState.REJECTED_UPDATE
            self.log_likelihood_ = 1e9
        elif call_count["n"] == 3:
            # Rig this run to have FAILED with an artificially high LL
            self.fit_state_ = FitState.FAILED
            self.log_likelihood_ = 1e8
        return result

    GaussianHSMM.fit = rigged_fit
    try:
        best_model, run_log = fit_with_random_restarts(
            X, n_regimes=3, n_features=3, n_init=10, base_seed=0, n_iter=50,
        )
    finally:
        GaussianHSMM.fit = real_fit

    assert best_model.fit_state_ == FitState.CONVERGED
    assert best_model.converged_ is True
    assert best_model.log_likelihood_ < 1e7

    # Check run log entries
    assert run_log[1]["fit_state"] == FitState.REJECTED_UPDATE
    assert run_log[1]["converged"] is False
    assert run_log[2]["fit_state"] == FitState.FAILED
    assert run_log[2]["converged"] is False


def test_s34_5_t4_monotonicity_ci_check():
    """T4: Standing CI check verifying is_log_likelihood_monotonic method exists and operates."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    model = GaussianHSMM(n_regimes=2, n_features=2, seed=0)
    model.fit(X, n_iter=20)
    assert hasattr(model, "is_log_likelihood_monotonic"), "Model must have is_log_likelihood_monotonic method wired into CI"
    assert model.is_log_likelihood_monotonic(tol=1e-4)
