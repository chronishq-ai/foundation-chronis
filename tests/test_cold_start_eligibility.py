"""
Tests for cold start and eligibility gating.
Verifies directive Section B and Tests 14-19:
  - ColdStartError raised when eligible sessions < min_present_sessions
  - No silent output or low-confidence output produced on cold start
  - Authoritative compute_eligibility_mask logic
  - Missingness and confidence thresholding impact on eligibility
"""

import numpy as np
import pytest
from backbone.hssm.dataset import make_dataset, compute_eligibility_mask
from backbone.hssm.fitting import fit_hssm
from backbone.hssm.gating import ColdStartError, fit_hssm_gated


def test_cold_start_error_raised_below_threshold():
    """Test 14: fit_hssm raises ColdStartError when present sessions < min_present_sessions."""
    np.random.seed(42)
    # Only 20 sessions when min is 30
    X = np.random.randn(20, 2)
    
    with pytest.raises(ColdStartError) as exc_info:
        fit_hssm(matrix=X, candidate_ks=(2,))
    
    assert "present sessions < cold-start minimum" in str(exc_info.value)


def test_cold_start_produces_no_silent_fallback():
    """Test 15: No low-confidence result is returned silently — an exception is always raised."""
    X = np.random.randn(15, 3)
    
    with pytest.raises(ColdStartError):
        fit_hssm_gated(X, min_present_sessions=30)


def test_eligibility_mask_accounts_for_missing_features():
    """Test 16: Rows with >= 50% missing features are marked ineligible."""
    feature_matrix = np.array([
        [1.0, 2.0, 3.0, 4.0],  # 0 missing -> eligible
        [1.0, np.nan, 3.0, 4.0],  # 1 missing (25%) -> eligible
        [1.0, np.nan, np.nan, 4.0],  # 2 missing (50%) -> eligible (50% threshold)
        [1.0, np.nan, np.nan, np.nan],  # 3 missing (75%) -> ineligible
        [np.nan, np.nan, np.nan, np.nan],  # 4 missing (100%) -> ineligible
    ])
    missingness = np.isnan(feature_matrix)
    confidence = np.where(missingness, 0.0, 1.0)
    
    elig = compute_eligibility_mask(feature_matrix, missingness, confidence, min_feature_presence=0.5)
    assert elig[0] == True
    assert elig[1] == True
    assert elig[2] == True
    assert elig[3] == False
    assert elig[4] == False


def test_eligibility_mask_accounts_for_low_confidence():
    """Test 17: Rows with low confidence features (< 0.5) are marked ineligible if fraction too high."""
    feature_matrix = np.ones((3, 4))
    missingness = np.zeros((3, 4), dtype=bool)
    confidence = np.array([
        [1.0, 1.0, 1.0, 1.0],  # all high confidence -> eligible
        [1.0, 0.8, 0.2, 0.1],  # 2 low confidence (<0.5) -> 50% high -> eligible
        [1.0, 0.1, 0.2, 0.1],  # 3 low confidence -> 25% high -> ineligible
    ])
    
    elig = compute_eligibility_mask(feature_matrix, missingness, confidence, min_confidence_threshold=0.5)
    assert elig[0] == True
    assert elig[1] == True
    assert elig[2] == False


def test_dataset_convenience_constructor_integrates_gating():
    """Test 18: make_dataset correctly identifies cold-start datasets."""
    cal = np.linspace(1.0, 50.0, 50)
    X = np.random.randn(50, 2)
    # Make 35 rows all-NaN so only 15 remain eligible
    X[:35, :] = np.nan
    
    ds = make_dataset(user_id="user_test", feature_matrix=X, calendar_index=cal, min_present_sessions=30)
    assert ds.eligible_session_count == 15
    with pytest.raises(ColdStartError):
        ds.check_eligibility_gate()


def test_fit_hssm_rejects_invalid_shapes():
    """Test 19: fit_hssm rejects 0 features or 1D arrays."""
    with pytest.raises(ValueError):
        fit_hssm(matrix=np.empty((50, 0)))
    with pytest.raises(ValueError):
        fit_hssm(matrix=np.array([1.0, 2.0, 3.0]))
