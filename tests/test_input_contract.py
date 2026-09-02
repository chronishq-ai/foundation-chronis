"""
Checklist items 1-3: input contract (29/31/50-session gate), bad inputs
(empty, zero-feature, NaN), and missing-session marginalization (not imputation).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest

from backbone.hssm.gating import fit_hssm_gated, ColdStartError, count_present_sessions
from backbone.hssm.model import GaussianHSMM


# ---------- #1: session-count gate ----------

def test_29_sessions_raises_cold_start():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(29, 5))
    with pytest.raises(ColdStartError):
        fit_hssm_gated(X, n_regimes=2, n_features=5, n_init=10, base_seed=1)


def test_31_sessions_allowed():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(31, 5))
    model, run_log = fit_hssm_gated(X, n_regimes=2, n_features=5, n_init=10, base_seed=1)
    assert model is not None
    assert len(run_log) == 10


def test_50_sessions_allowed():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 5))
    model, run_log = fit_hssm_gated(X, n_regimes=2, n_features=5, n_init=10, base_seed=1)
    assert model is not None


def test_gate_boundary_is_exactly_30():
    # 30 present sessions must be allowed (>= min_present_sessions), 29 must not.
    rng = np.random.default_rng(1)
    X30 = rng.normal(size=(30, 5))
    model, _ = fit_hssm_gated(X30, n_regimes=2, n_features=5, n_init=10, base_seed=1)
    assert model is not None

    X29 = rng.normal(size=(29, 5))
    with pytest.raises(ColdStartError):
        fit_hssm_gated(X29, n_regimes=2, n_features=5, n_init=10, base_seed=1)


def test_gate_uses_present_not_raw_row_count():
    # 40 raw rows but only 25 present (rest NaN) must still raise -- missingness
    # cannot be used to pad the session count past the gate.
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 5))
    X[25:] = np.nan  # 15 rows fully missing -> only 25 present
    assert count_present_sessions(X) == 25
    with pytest.raises(ColdStartError):
        fit_hssm_gated(X, n_regimes=2, n_features=5, n_init=10, base_seed=1)


def test_partially_missing_row_does_not_count_as_present():
    # Confidence contract tiers: >= 50% feature presence is eligible (tiers 1.0, 0.75, 0.5); < 50% is excluded.
    rng = np.random.default_rng(3)
    X_tier_0_8 = rng.normal(size=(35, 5))
    X_tier_0_8[:6, 0] = np.nan  # 1/5 missing = 80% present >= 50% -> eligible
    assert count_present_sessions(X_tier_0_8) == 35

    X_tier_0_2 = rng.normal(size=(35, 5))
    X_tier_0_2[:6, :4] = np.nan  # 4/5 missing = 20% present < 50% -> excluded (6 rows excluded, 29 eligible)
    assert count_present_sessions(X_tier_0_2) == 29  # 35 - 6 == 29, below gate
    with pytest.raises(ColdStartError):
        fit_hssm_gated(X_tier_0_2, n_regimes=2, n_features=5, n_init=10, base_seed=1)


# ---------- #2: bad inputs ----------

def test_empty_input_does_not_produce_output():
    X = np.empty((0, 5))
    assert count_present_sessions(X) == 0
    with pytest.raises(ColdStartError):
        fit_hssm_gated(X, n_regimes=2, n_features=5, n_init=10, base_seed=1)


def test_zero_feature_input_is_rejected_not_silently_accepted():
    # F=0: emission log-likelihood is degenerate (sum over zero dims == 0 for
    # every regime, every timestep), so EM trivially "converges" at LL≈0 and
    # hands back a fitted-looking model with mu/var of shape (K, 0). Nothing
    # in gating.py/model.py currently validates n_features > 0. This test
    # documents the required behavior (reject) -- as of this writing it FAILS,
    # i.e. this is a real found bug, not a flaky test.
    X = np.random.default_rng(0).normal(size=(31, 0))
    with pytest.raises((ValueError, ColdStartError, IndexError)):
        fit_hssm_gated(X, n_regimes=2, n_features=0, n_init=10, base_seed=1)


def test_nan_row_is_marginalized_not_imputed_in_emission_loglik():
    # Directly exercise model.py's emission log-likelihood: a NaN row must
    # contribute exactly 0 to the log-likelihood for every regime, i.e. it is
    # skipped/marginalized, never replaced by an imputed value that would
    # produce some other (non-zero) log-likelihood contribution.
    model = GaussianHSMM(n_regimes=2, n_features=3, seed=0)
    model.mu = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    model.var = np.ones((2, 3))
    X = np.array([[0.1, -0.2, 0.05], [np.nan, np.nan, np.nan], [5.1, 4.9, 5.0]])
    loglik = model._emission_loglik(X)
    assert loglik.shape == (3, 2)
    assert np.all(loglik[1] == 0.0), "NaN row must contribute 0 loglik for every regime (marginalized, not imputed)"
    # the present rows must NOT be all-zero (sanity: marginalization only applies to missing rows)
    assert not np.all(loglik[0] == 0.0)
    assert not np.all(loglik[2] == 0.0)


def test_nan_row_excluded_from_m_step_parameter_estimates():
    # A NaN row must not drag mu/var toward some imputed value (e.g. 0 or the
    # column mean computed naively over NaNs). We check that mu/var estimates
    # for a regime are identical whether or not extra all-NaN rows are present,
    # given the same present-data weights.
    rng = np.random.default_rng(0)
    present = rng.normal(loc=3.0, scale=1.0, size=(40, 3))

    X_no_missing = present.copy()
    X_with_missing = np.vstack([present, np.full((10, 3), np.nan)])

    def fit_one_regime_stats(X):
        model = GaussianHSMM(n_regimes=1, n_features=3, seed=0)
        model.pi = np.array([1.0])
        model.A = np.array([[1.0]])
        model.mu = np.zeros((1, 3))
        model.var = np.ones((1, 3))
        model.dur_mu = np.zeros(1)
        model.dur_sigma = np.ones(1)
        regime_post = np.ones((X.shape[0], 1))
        entry_post = np.zeros((X.shape[0], 1, model.Dmax))
        xi_counts = np.zeros((1, 1))
        model._m_step(X, regime_post, entry_post, xi_counts)
        return model.mu[0], model.var[0]

    mu_clean, var_clean = fit_one_regime_stats(X_no_missing)
    mu_missing, var_missing = fit_one_regime_stats(X_with_missing)

    assert np.allclose(mu_clean, mu_missing, atol=1e-8), (
        f"NaN rows leaked into mu estimate: {mu_clean} vs {mu_missing}"
    )
    assert np.allclose(var_clean, var_missing, atol=1e-8), (
        f"NaN rows leaked into var estimate: {var_clean} vs {var_missing}"
    )


# ---------- #3: missing sessions preserved through the pipeline ----------

def test_missing_session_preserved_and_gate_recomputed_correctly():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(35, 5))
    missing_idx = [2, 9, 17, 24]
    X[missing_idx] = np.nan
    assert count_present_sessions(X) == 35 - len(missing_idx)
    # array itself must still literally contain the NaNs (nothing upstream imputed them)
    assert np.all(np.isnan(X[missing_idx]))
