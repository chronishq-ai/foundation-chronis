import pytest
import numpy as np
from backbone.hssm.model import GaussianHSMM, NotFittedError
from backbone.hssm.fitting import select_k_by_bic, fit_hssm
from backbone.hssm.gating import count_present_sessions, fit_hssm_gated, ColdStartError
from backbone.shared.feature_reduction import reduce_dimensionality


class PreFixGaussianHSMM(GaussianHSMM):
    """Simulates the pre-fix model which discards the entire row on any single NaN."""
    def _emission_loglik(self, X: np.ndarray) -> np.ndarray:
        if self.var is None or self.mu is None:
            raise NotFittedError("Emission parameters var and mu are not initialized")
        T, K = X.shape[0], self.K
        loglik = np.zeros((T, K))
        for k in range(K):
            var = self.var[k]
            diff = X - self.mu[k]
            ll = -0.5 * np.sum(diff**2 / var + np.log(2 * np.pi * var), axis=1)
            loglik[:, k] = ll
        missing = np.isnan(X).any(axis=1)
        loglik[missing, :] = 0.0
        return loglik


def test_s34_1_t1_partial_missingness_legacy_bug_document():
    """Historical test documenting pre-fix behavior under PreFixGaussianHSMM."""
    K, F = 2, 2
    mu = np.array([[1.0, 2.0], [10.0, 20.0]])
    var = np.array([[0.5, 0.8], [1.5, 2.5]])
    
    prefix_model = PreFixGaussianHSMM(n_regimes=K, n_features=F, seed=42)
    prefix_model.mu, prefix_model.var = mu.copy(), var.copy()
    
    X = np.array([[np.nan, 1.9]])
    loglik = prefix_model._emission_loglik(X)
    assert loglik[0, 0] == 0.0, "Pre-fix code discards row with partial missingness"


def test_s34_1_t1_fixed_analytical_gaussian_likelihood():
    """Step 2: Hand-computable Gaussian test verifying the fixed model likelihood.
    
    Modality A (feature 0) is missing (NaN), Modality B (feature 1) is present (1.9).
    Regime 0: mu = 2.0, var = 0.8, val = 1.9.
    Analytical derivation:
      diff = 1.9 - 2.0 = -0.1
      diff^2 / var = 0.01 / 0.8 = 0.0125
      ln(2 * pi * var) = ln(2 * pi * 0.8) = ln(5.026548245743669) = 1.614748641178652
      expected_loglik = -0.5 * (0.0125 + 1.614748641178652) = -0.813624320589326
    """
    K, F = 2, 2
    mu = np.array([[1.0, 2.0], [10.0, 20.0]])
    var = np.array([[0.5, 0.8], [1.5, 2.5]])
    
    model = GaussianHSMM(n_regimes=K, n_features=F, seed=42)
    model.mu, model.var = mu.copy(), var.copy()
    
    X = np.array([[np.nan, 1.9]])
    loglik = model._emission_loglik(X)
    
    expected_loglik_f1 = -0.5 * (((1.9 - 2.0)**2 / 0.8) + np.log(2.0 * np.pi * 0.8))
    assert abs(expected_loglik_f1 - (-0.8136167575475678)) < 1e-6
    assert np.isclose(loglik[0, 0], expected_loglik_f1, atol=1e-5), (
        f"Fixed model emission loglik ({loglik[0, 0]}) must match analytical loglik ({expected_loglik_f1})"
    )


def test_s34_1_t2_fully_missing_row_contributes_zero():
    """T2: Row missing ALL modalities -> zero contribution in fixed model."""
    K, F = 2, 2
    mu = np.array([[1.0, 2.0], [10.0, 20.0]])
    var = np.array([[0.5, 0.8], [1.5, 2.5]])
    
    model = GaussianHSMM(n_regimes=K, n_features=F, seed=42)
    model.mu, model.var = mu.copy(), var.copy()
    
    X = np.array([[np.nan, np.nan]])
    loglik = model._emission_loglik(X)
    assert loglik[0, 0] == 0.0, "Fully missing row must contribute 0.0 loglik"
    assert loglik[0, 1] == 0.0, "Fully missing row must contribute 0.0 loglik"


def test_s34_1_t3_bic_selection_stability_fixed():
    """T3: Dataset with 20% of rows having one missing feature.
    Fixed code matches control selection (selects true K=2)."""
    rng = np.random.default_rng(8)
    T = 200
    
    regimes = np.array([0]*100 + [1]*100)
    X_control = np.zeros((T, 2))
    X_control[regimes == 0] = rng.normal(0.0, 1.5, size=(100, 2))
    X_control[regimes == 1] = rng.normal(4.0, 1.5, size=(100, 2))
    
    X_missing = X_control.copy()
    nan_rows = rng.choice(T, size=40, replace=False)
    for idx, r in enumerate(nan_rows):
        X_missing[r, idx % 2] = np.nan
        
    _, report_ctrl = select_k_by_bic(
        X_control, n_features=2, k_candidates=(2, 3), n_init=10, n_iter=150, base_seed=42, bypass_init_gate=True
    )
    assert report_ctrl["selected_k"] == 2
    
    _, report_missing = select_k_by_bic(
        X_missing, n_features=2, k_candidates=(2, 3), n_init=10, n_iter=150, base_seed=42, bypass_init_gate=True
    )
    assert report_missing["selected_k"] == 2, f"Fixed model must select true K=2, got {report_missing['selected_k']}"


def test_s34_1_step1_feature_reduction_partial_missingness_fixed():
    """Step 1(c): Test that feature reduction/PCA preserves partially-missing rows."""
    rng = np.random.default_rng(42)
    T, F = 50, 10
    Z = rng.normal(0, 1, size=(T, F))
    feature_names = [f"feat_{i}" for i in range(F)]

    # Row 0 has 1 missing feature (feature 0 is NaN, features 1..9 present)
    Z[0, 0] = np.nan

    Z_reduced, _, _ = reduce_dimensionality(
        Z, feature_names=feature_names, target_dims=8, min_dims=8, max_dims=10
    )

    # Fixed behavior: Z_reduced[0] is NOT entirely NaN; it contains valid PCA components
    assert not np.isnan(Z_reduced[0]).all(), "Fixed PCA must preserve partially observed row 0"
    assert not np.isnan(Z_reduced[0]).any(), "Z_reduced[0] scores should be fully computed from available features"


def test_s34_1_step2_full_pipeline_fit_hssm_partial_missingness_fixed():
    """Step 1(e): Test the full end-to-end fit_hssm() path with partially-missing observations."""
    rng = np.random.default_rng(123)
    T, F = 40, 4
    X = rng.normal(0, 1, size=(T, F))

    for r in range(10):
        X[r, 0] = np.nan

    result = fit_hssm(
        matrix=X,
        candidate_ks=(2,),
        n_initializations=5,
        max_iter=100,
        allow_fast_test_fit=True,
        random_seed=42,
    )

    assert result.model._is_fitted, "fit_hssm should return a fitted HSSM model"
    emis_loglik = result.model._emission_loglik(X[:10])
    assert not np.all(emis_loglik == 0.0), "Fixed fit_hssm must compute non-zero emission loglik for partially missing rows"


def test_s34_1_step3_cold_start_confidence_contract_tiers_fixed():
    """Step 1(d): Test cold-start/session-eligibility logic with confidence contract tiers
    (1.0, 0.75, 0.5 = passed, <0.5 = excluded)."""
    T = 35  # >= 30 threshold
    X_tier_1_0 = np.ones((T, 4))
    assert count_present_sessions(X_tier_1_0) == 35

    # Tier 0.75: 1 feature missing (3/4 present = 0.75 confidence) -> eligible (35)
    X_tier_0_75 = np.ones((T, 4))
    X_tier_0_75[:, 0] = np.nan
    assert count_present_sessions(X_tier_0_75) == 35, "Tier 0.75 must be counted as present (35)"

    # Tier 0.5: 2 features missing (2/4 present = 0.5 confidence) -> eligible (35)
    X_tier_0_5 = np.ones((T, 4))
    X_tier_0_5[:, :2] = np.nan
    assert count_present_sessions(X_tier_0_5) == 35, "Tier 0.5 must be counted as present (35)"

    # Tier < 0.5: 3 features missing (1/4 present = 0.25 confidence) -> excluded (0)
    X_tier_0_25 = np.ones((T, 4))
    X_tier_0_25[:, :3] = np.nan
    assert count_present_sessions(X_tier_0_25) == 0, "Tier < 0.5 must be excluded (0)"

    # Verify fit_hssm_gated fits Tier 0.75 without raising ColdStartError
    best_model, _ = fit_hssm_gated(X_tier_0_75, n_regimes=2, n_features=4, n_init=10)
    assert best_model._is_fitted

    # Verify fit_hssm_gated raises ColdStartError for Tier < 0.5
    with pytest.raises(ColdStartError):
        fit_hssm_gated(X_tier_0_25, n_regimes=2, n_features=4, n_init=10)
