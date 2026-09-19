"""
Tests for canonical API and HSSMResult output contract.
Verifies directive Section A, Section Q, and Tests 1-4:
  - Single fit_hssm entrypoint
  - All ~40 required fields present and typed
  - Fast state is strictly None for baseline model (no faking)
  - Backward-compatibility aliases work seamlessly
"""

import numpy as np
import pytest
from backbone.hssm.fitting import fit_hssm, HSSMResult
from backbone.hssm.model import FitState
from backbone.hssm.gating import ColdStartError


def test_fit_hssm_is_canonical_entrypoint():
    """Test 1: fit_hssm is the single entry point returning an HSSMResult."""
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(0, 0.5, (30, 2)),
        np.random.normal(3, 0.5, (30, 2)),
    ])
    
    result = fit_hssm(
        matrix=X,
        candidate_ks=(2,),
        n_initializations=10,
        max_iter=50,
        random_seed=42,
        allow_fast_test_fit=True,
    )
    
    assert isinstance(result, HSSMResult)


def test_hssm_result_contains_all_directive_fields():
    """Test 2: All directive-required fields are present on HSSMResult."""
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(0, 0.5, (30, 2)),
        np.random.normal(3, 0.5, (30, 2)),
    ])
    timestamps = np.linspace(100.0, 160.0, 60)
    
    result = fit_hssm(
        matrix=X,
        candidate_ks=(2,),
        n_initializations=10,
        max_iter=50,
        random_seed=42,
        allow_fast_test_fit=True,
        timestamps=timestamps,
        user_id="test_user_42",
        dataset_hash="test_hash_abc",
    )
    
    # Section Q fields:
    # 1. Identity
    assert result.user_id == "test_user_42"
    assert result.model_identity == "ExplicitDurationSwitchingBaselineV1"
    assert result.model_family == "GaussianHSMM"
    assert isinstance(result.model_capability, dict)
    assert result.is_baseline_model is True
    assert result.fast_state_supported is False
    
    # 2. Slow state
    assert result.p_t_estimate is not None
    assert len(result.p_t_estimate) == 60
    assert result.p_t_posterior is not None
    assert result.p_t_posterior.shape == (60, 2)
    
    # 3. Fast state (MUST be None for baseline)
    assert result.m_t_posterior_mean is None
    assert result.m_t_uncertainty is None
    
    # 4. Duration
    assert isinstance(result.duration_parameters, dict)
    assert result.duration_model_identity == "explicit_truncated_gaussian"
    assert result.Dmax > 0
    assert result.duration_unit_source == "calendar_days"
    
    # 5. Temporal
    assert result.calendar_start == 100.0
    assert result.calendar_end == 160.0
    assert result.observation_density > 0.0
    
    # 6. Versioning
    assert result.model_version == "1.0.0"
    assert result.feature_schema_version == "1.0.0"
    assert result.alignment_version == "1.0.0"
    
    # 7. Eligibility
    assert result.eligible_session_count >= 30
    assert result.min_present_sessions == 30
    
    # 8. Data hash
    assert result.fit_dataset_hash == "test_hash_abc"
    
    # 9. Model selection
    assert result.K == 2
    assert isinstance(result.BIC, float)
    assert isinstance(result.AIC, float)
    
    # 10. Fit state
    assert isinstance(result.fit_state, FitState)
    assert isinstance(result.optimizer_success, bool)
    assert isinstance(result.fallback_used, bool)
    assert isinstance(result.run_log, list)
    assert len(result.run_log) >= 10
    
    # 11. Neural residual
    assert result.neural_residual_enabled is False
    assert result.residual_mode == "DISABLED"


def test_baseline_never_fakes_fast_state():
    """Test 3: Fast state m_t is NEVER populated with dummy numbers for the baseline."""
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(0, 0.5, (25, 2)),
        np.random.normal(3, 0.5, (25, 2)),
    ])
    
    result = fit_hssm(
        matrix=X,
        candidate_ks=(2,),
        n_initializations=10,
        max_iter=50,
        random_seed=42,
        allow_fast_test_fit=True,
    )
    
    assert result.fast_state_supported is False
    assert result.m_t_posterior_mean is None
    assert result.m_t_uncertainty is None
    assert result.m_t is None  # legacy alias also returns None


def test_legacy_backward_compatibility_aliases():
    """Test 4: Legacy callers can access .p_t, .selected_k, .duration_info, .convergence_metadata."""
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(0, 0.5, (25, 2)),
        np.random.normal(3, 0.5, (25, 2)),
    ])
    
    result = fit_hssm(
        matrix=X,
        candidate_ks=(2,),
        n_initializations=10,
        max_iter=50,
        random_seed=42,
        allow_fast_test_fit=True,
    )
    
    # Legacy aliases
    assert result.p_t is not None
    assert len(result.p_t) == 50
    assert result.regime_posterior is not None
    assert result.k_selected == 2
    assert "duration_mu" in result.duration_info
    assert "bic_by_k" in result.convergence_metadata
    assert result.model is not None
