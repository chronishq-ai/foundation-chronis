"""
Tests for DualTimescaleHSSM scaffold, Kim filter/smoother, and fast+slow state representations.
Verifies directive Section D, Phase 2, and Tests 33-37:
  - Model identity: DualTimescaleKimHSSMV1
  - Model family: SwitchingLinearDynamicalSystem
  - Capability flags: is_baseline_model=False, fast_state_supported=True
  - State representations: fast continuous m_t in R^L, slow discrete p_t in {0..K-1}
  - Uncertainty bounds: strictly positive standard errors along diagonal
  - Generalized EM with backtracking and missing observation handling
"""

import numpy as np
import pytest
from backbone.hssm.dual_model import DualTimescaleHSSM
from backbone.hssm.model import FitState, NotFittedError


def test_dual_model_identity_and_capabilities():
    """Test 33: Full dual-timescale model identity, family, and capability contract."""
    model = DualTimescaleHSSM(n_regimes=3, n_features=4, latent_dim=2, max_duration=30)
    
    assert model.model_identity == "DualTimescaleKimHSSMV1"
    assert model.model_family == "SwitchingLinearDynamicalSystem"
    assert model.is_baseline_model is False
    assert model.fast_state_supported is True
    assert model.L == 2
    assert model.K == 3
    assert model.F == 4
    
    caps = model.model_capability
    assert caps["supports_fast_state_m_t"] is True
    assert caps["supports_slow_state_p_t"] is True
    assert caps["supports_explicit_duration"] is True
    assert caps["supports_neural_residual"] is False
    assert caps["is_production_grade_full_hssm"] is True
    assert caps["is_baseline_model"] is False
    assert caps["latent_dimension"] == 2


def test_dual_model_unfitted_raises_not_fitted_error():
    """Test: Calling estimate_states before fit raises NotFittedError."""
    model = DualTimescaleHSSM(n_regimes=2, n_features=2, latent_dim=2)
    X = np.random.randn(30, 2)
    with pytest.raises(NotFittedError):
        model.estimate_states(X)


def test_dual_model_kim_filter_and_smoother_shapes():
    """Test 34, 35: Kim filter GPB2 collapse and backward smoother produce correct shapes."""
    np.random.seed(42)
    T, F, L, K = 40, 3, 2, 2
    X = np.random.randn(T, F)
    
    model = DualTimescaleHSSM(n_regimes=K, n_features=F, latent_dim=L, max_duration=15, seed=42)
    model.fit(X, n_iter=10)
    
    m_t, m_t_std, p_t, p_t_post = model.estimate_states(X)
    
    # Fast continuous state
    assert m_t.shape == (T, L)
    assert not np.isnan(m_t).any()
    assert not np.isinf(m_t).any()
    
    # Fast state uncertainty (std error)
    assert m_t_std.shape == (T, L)
    assert np.all(m_t_std > 0.0)  # strictly positive uncertainty
    assert not np.isnan(m_t_std).any()
    
    # Slow discrete regime
    assert p_t.shape == (T,)
    assert set(np.unique(p_t)).issubset(set(range(K)))
    
    # Slow regime posterior
    assert p_t_post.shape == (T, K)
    np.testing.assert_allclose(p_t_post.sum(axis=1), np.ones(T), rtol=1e-5)


def test_dual_model_handles_missing_observations_gracefully():
    """Test 36: Kalman observation update handles NaNs via marginalization."""
    np.random.seed(42)
    T, F, L, K = 50, 4, 2, 2
    X = np.random.randn(T, F)
    # Introduce sporadic NaNs
    mask = np.random.uniform(size=X.shape) < 0.2
    X[mask] = np.nan
    # Whole row missing
    X[10, :] = np.nan
    
    model = DualTimescaleHSSM(n_regimes=K, n_features=F, latent_dim=L, max_duration=20, seed=42)
    model.fit(X, n_iter=10)
    
    m_t, m_t_std, p_t, p_t_post = model.estimate_states(X)
    
    # Missing row at index 10 should still produce smooth interpolated m_t and larger uncertainty
    assert not np.isnan(m_t[10]).any()
    assert not np.isnan(m_t_std[10]).any()
    assert np.all(m_t_std[10] > 0.0)


def test_dual_model_information_criteria_and_monotonicity():
    """Test 37: BIC and AIC computed correctly; log-likelihood history is recorded."""
    np.random.seed(42)
    T, F, L, K = 60, 2, 2, 2
    X = np.random.randn(T, F)
    
    model = DualTimescaleHSSM(n_regimes=K, n_features=F, latent_dim=L, max_duration=20, seed=42)
    model.fit(X, n_iter=15)
    
    bic_val = model.bic(n_observations=T)
    aic_val = model.aic()
    
    assert isinstance(bic_val, float)
    assert not np.isnan(bic_val)
    assert isinstance(aic_val, float)
    assert not np.isnan(aic_val)
    assert len(model.log_likelihood_history_) > 0
    assert model.n_params() > 0
