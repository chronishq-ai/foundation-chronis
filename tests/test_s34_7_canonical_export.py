import pytest
import numpy as np
from backbone.hssm import fit_hssm, HSSMResult, GaussianHSMM
from backbone.attractors.validation import fit_user_pipeline


def test_t1_fresh_import_and_hssm_result_schema():
    """T1 — Fresh import & backend-neutral schema verification.
    
    Verifies that fit_hssm returns an HSSMResult object exposing all required fields:
      - m_t: None (unsupported continuous state for GaussianHSMM baseline, S34.3 gap)
      - m_t_uncertainty: None
      - p_t: numpy array of shape (T,)
      - regime_posterior: numpy array of shape (T, K)
      - duration_info: dict containing duration_mu, duration_sigma, max_duration
      - duration_unit: "sessions" or "calendar_days"
      - selected_k: integer
      - model_class: "GaussianHSMM"
      - convergence_metadata: dict containing bic_by_k, loglik_by_k, etc.
    """
    X = np.random.default_rng(0).normal(size=(35, 2))
    res = fit_hssm(X, candidate_ks=(2, 3), n_initializations=2, allow_fast_test_fit=True)

    assert isinstance(res, HSSMResult)
    assert isinstance(res.model, GaussianHSMM)
    
    # Unsupported continuous state fields return None (no fake values)
    assert res.m_t is None
    assert res.m_t_uncertainty is None
    
    # Supported fields
    assert isinstance(res.p_t, np.ndarray)
    assert res.p_t.shape == (35,)
    assert isinstance(res.regime_posterior, np.ndarray)
    assert res.regime_posterior.shape == (35, res.selected_k)
    
    assert isinstance(res.duration_info, dict)
    assert "duration_mu" in res.duration_info
    assert "duration_sigma" in res.duration_info
    assert "max_duration" in res.duration_info
    assert res.duration_unit == "sessions"
    
    assert isinstance(res.selected_k, int)
    assert res.selected_k in (2, 3)
    assert res.model_class == "GaussianHSMM"
    
    assert isinstance(res.convergence_metadata, dict)
    assert "bic_by_k" in res.convergence_metadata
    assert "loglik_by_k" in res.convergence_metadata
    assert "convergence_rate_by_k" in res.convergence_metadata
    
    # Backward compatibility properties
    assert res.k_selected == res.selected_k
    assert res.bic_by_k == res.convergence_metadata["bic_by_k"]
    assert res.loglik_by_k == res.convergence_metadata["loglik_by_k"]


def test_t1_export_with_timestamps():
    """T1 sub-case: verify duration_unit changes to 'calendar_days' when timestamps are provided."""
    X = np.random.default_rng(1).normal(size=(35, 2))
    timestamps = np.arange(35, dtype=float)

    res = fit_hssm(X, candidate_ks=(2,), n_initializations=2, allow_fast_test_fit=True, timestamps=timestamps)
    assert isinstance(res, HSSMResult)
    assert res.duration_unit == "calendar_days"


def test_t2_cross_package_pipeline_smoke_test():
    """T2 — Cross-package smoke test.
    
    Verifies that real pipeline downstream code runs end-to-end calling the real
    fit_hssm_gated / fit_with_random_restarts entry points without monkeypatching.
    """
    model, run_log, user = fit_user_pipeline(n_sessions=35, n_features_raw=12, n_regimes=2, target_dims=8, seed=42)
    assert model._is_fitted
    assert len(run_log) == 10
    assert user["X"].shape[0] == 35
