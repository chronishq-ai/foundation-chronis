"""
Tests for baseline model identity and capability contract.
Verifies directive Section C & Section Q:
  - Model identity: ExplicitDurationSwitchingBaselineV1
  - Model family: GaussianHSMM
  - Capability flags: is_baseline_model=True, fast_state_supported=False
  - Duration identity: explicit_truncated_gaussian
  - Duration unit source tracking
  - FitState enum and AIC support
"""

import numpy as np
import pytest
from backbone.hssm.model import GaussianHSMM, FitState
from backbone.hssm.fitting import fit_hssm, HSSMResult


def test_baseline_model_identity_properties():
    """Test 26, 27, 28, 29, 30: Model identity, family, and capability contract."""
    model = GaussianHSMM(K=3, n_features=2, Dmax=30)
    
    # Model identity
    assert model.model_identity == "ExplicitDurationSwitchingBaselineV1"
    assert model.model_family == "GaussianHSMM"
    assert model.is_baseline_model is True
    assert model.fast_state_supported is False
    assert model.duration_model_identity == "explicit_truncated_gaussian"
    assert model.duration_unit_source == "session_index"
    
    # Capability dict
    caps = model.model_capability
    assert caps["supports_fast_state_m_t"] is False
    assert caps["supports_slow_state_p_t"] is True
    assert caps["supports_explicit_duration"] is True
    assert caps["supports_neural_residual"] is False
    assert caps["is_production_grade_full_hssm"] is False


def test_baseline_duration_unit_source_with_timestamps():
    """Test 31: duration_unit_source switches to calendar_days when timestamps provided."""
    model = GaussianHSMM(K=2, n_features=2, Dmax=20, duration_unit="calendar_days")
    assert model.duration_unit_source == "calendar_days"


def test_fit_state_enum_contains_model_selection_contested():
    """Test: FitState has MODEL_SELECTION_CONTESTED as 7th state."""
    assert hasattr(FitState, "MODEL_SELECTION_CONTESTED")
    assert FitState.MODEL_SELECTION_CONTESTED.value == "MODEL_SELECTION_CONTESTED"
    
    all_states = [s.value for s in FitState]
    expected = [
        "RUNNING", "CONVERGED", "REJECTED_UPDATE", "NUMERICAL_FAILURE",
        "MAX_ITER_REACHED", "FAILED", "MODEL_SELECTION_CONTESTED"
    ]
    for exp in expected:
        assert exp in all_states


def test_aic_property_and_computation():
    """Test: AIC computation on GaussianHSMM."""
    np.random.seed(42)
    X = np.random.randn(60, 2)
    model = GaussianHSMM(K=2, n_features=2, Dmax=15)
    model.fit(X, n_iter=50)
    
    aic_val = model.aic()
    bic_val = model.bic()
    
    assert isinstance(aic_val, float)
    assert not np.isnan(aic_val)
    assert not np.isinf(aic_val)
    assert isinstance(bic_val, float)


def test_hssm_result_baseline_identity_fields():
    """Test 32: fit_hssm returns result with correct baseline identity fields."""
    np.random.seed(42)
    # Generate 60 steps of clear 2-regime data
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
    assert result.model_identity == "ExplicitDurationSwitchingBaselineV1"
    assert result.model_family == "GaussianHSMM"
    assert result.is_baseline_model is True
    assert result.fast_state_supported is False
    assert result.m_t_posterior_mean is None
    assert result.m_t_uncertainty is None
    assert result.duration_model_identity == "explicit_truncated_gaussian"
    assert result.neural_residual_enabled is False
    assert result.residual_mode == "DISABLED"
    assert result.fit_state in (FitState.CONVERGED, FitState.MAX_ITER_REACHED, FitState.MODEL_SELECTION_CONTESTED)
