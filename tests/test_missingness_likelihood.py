"""
Tests for missingness handling and marginalization in likelihood computations.
Verifies directive Section B & Section Q:
  - Emission log-likelihood marginalization under missing data (NaN)
  - Partially observed rows contribute only observed dimensions
  - Fully missing rows have zero log-likelihood contribution across all regimes (uniform marginal)
  - Fitting succeeds without NaN propagation on partially missing sequences
  - Heldout log-likelihood computes safely in the presence of NaNs
"""

import numpy as np
import pytest
from backbone.hssm.model import GaussianHSMM
from backbone.hssm.fitting import fit_hssm, _compute_heldout_predictive_loglik


def test_emission_loglik_marginalization_partial_missing():
    """Test 20: Missing dimension is marginalized (contributes 0 in log-space)."""
    model = GaussianHSMM(K=2, n_features=2, Dmax=10)
    model.mu = np.array([[0.0, 0.0], [3.0, 3.0]])
    model.var = np.array([[1.0, 1.0], [1.0, 1.0]])
    
    # 3 rows: row 0 full, row 1 dim 1 missing, row 2 dim 0 missing
    X = np.array([
        [0.0, 0.0],
        [0.0, np.nan],
        [np.nan, 0.0],
    ])
    
    ll = model._emission_loglik(X)
    assert not np.isnan(ll).any()
    assert not np.isinf(ll).any()
    
    # Row 1 and Row 2 have exactly half the log-density terms of Row 0 (for regime 0)
    # 1D standard normal at 0 is log(1/sqrt(2pi)) ≈ -0.9189
    # 2D standard normal at (0,0) is 2 * -0.9189 ≈ -1.8379
    np.testing.assert_allclose(ll[0, 0], 2 * -0.5 * np.log(2 * np.pi), rtol=1e-5)
    np.testing.assert_allclose(ll[1, 0], -0.5 * np.log(2 * np.pi), rtol=1e-5)
    np.testing.assert_allclose(ll[2, 0], -0.5 * np.log(2 * np.pi), rtol=1e-5)


def test_emission_loglik_all_missing_row():
    """Test 21: Completely missing row assigns 0.0 log-likelihood across all regimes."""
    model = GaussianHSMM(K=2, n_features=2, Dmax=10)
    model.mu = np.array([[0.0, 0.0], [3.0, 3.0]])
    model.var = np.array([[1.0, 1.0], [1.0, 1.0]])
    
    X = np.array([
        [np.nan, np.nan],
    ])
    ll = model._emission_loglik(X)
    assert ll.shape == (1, 2)
    assert ll[0, 0] == 0.0
    assert ll[0, 1] == 0.0


def test_fitting_converges_with_sporadic_missingness():
    """Test 23: GaussianHSMM fitting succeeds without NaN errors when ~15% of values are missing."""
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(0, 0.5, (30, 2)),
        np.random.normal(3, 0.5, (30, 2)),
    ])
    # Introduce sporadic NaNs
    mask = np.random.uniform(size=X.shape) < 0.15
    X[mask] = np.nan
    
    result = fit_hssm(
        matrix=X,
        candidate_ks=(2,),
        n_initializations=10,
        max_iter=15,
        random_seed=42,
        allow_fast_test_fit=True,
    )
    
    assert result.p_t_estimate is not None
    assert len(result.p_t_estimate) == 60
    assert not np.isnan(result.p_t_posterior).any()


def test_heldout_loglik_handles_missing_data():
    """Test 24: _compute_heldout_predictive_loglik handles data containing NaNs."""
    np.random.seed(42)
    X = np.random.randn(60, 2)
    X[50:, 0] = np.nan  # some missing in heldout region
    
    model = GaussianHSMM(K=2, n_features=2, Dmax=10)
    model.fit(X, n_iter=5)
    
    heldout_ll = _compute_heldout_predictive_loglik(X, model, holdout_fraction=0.2)
    assert heldout_ll is not None
    assert isinstance(heldout_ll, float)
    assert not np.isnan(heldout_ll)
    assert not np.isinf(heldout_ll)
