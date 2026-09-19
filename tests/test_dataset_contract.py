"""
Tests for HSSMFitDataset input contract.
Verifies directive Section B:
  - User ID presence and validation
  - Calendar-time index (rejects integer session indices without explicit opt-in)
  - Feature matrix dimension and schema versioning
  - Missingness, confidence, and eligibility masks
  - Deterministic dataset hash
  - Anti-leakage validation
"""

import numpy as np
import pytest
from backbone.hssm.dataset import HSSMFitDataset, make_dataset, compute_eligibility_mask
from backbone.hssm.gating import ColdStartError


def test_dataset_requires_valid_user_id():
    """Test 6: Enforce non-empty string user_id."""
    T, F = 50, 3
    cal = np.linspace(100.0, 150.0, T)
    X = np.random.randn(T, F)
    
    with pytest.raises(ValueError, match="user_id"):
        make_dataset(user_id="", feature_matrix=X, calendar_index=cal)


def test_dataset_rejects_sequential_integers_as_calendar_time():
    """Test 5: Reject integer session counts (0, 1, ..., T-1) masquerading as calendar time."""
    T, F = 40, 2
    sequential_idx = np.arange(T, dtype=float)
    X = np.random.randn(T, F)
    
    # Should raise ValueError unless time_grid_unit='session_index'
    with pytest.raises(ValueError, match="sequential session count"):
        make_dataset(
            user_id="user_123",
            feature_matrix=X,
            calendar_index=sequential_idx,
            time_grid_unit="calendar_days",
        )
    
    # Explicitly acknowledging session_index is allowed
    ds = make_dataset(
        user_id="user_123",
        feature_matrix=X,
        calendar_index=sequential_idx,
        time_grid_unit="session_index",
    )
    assert ds.time_grid_unit == "session_index"


def test_dataset_enforces_dimension_alignment():
    """Test 7: feature_matrix rows must match calendar_index length."""
    cal = np.array([1.0, 2.5, 4.0, 5.2])  # 4 rows
    X = np.random.randn(5, 2)  # 5 rows — mismatch!
    
    with pytest.raises(ValueError, match="must match"):
        make_dataset(user_id="user_123", feature_matrix=X, calendar_index=cal)


def test_dataset_enforces_feature_names_match_feature_count():
    """Test 8: feature_names length must match feature matrix column count."""
    cal = np.array([1.0, 2.5, 4.0])
    X = np.random.randn(3, 2)
    
    with pytest.raises(ValueError, match="feature_names"):
        make_dataset(
            user_id="user_123",
            feature_matrix=X,
            calendar_index=cal,
            feature_names=["f1"],  # 1 name for 2 features — mismatch!
        )


def test_dataset_masks_and_eligibility_computation():
    """Test 9, 10, 11: Missingness, confidence, and eligibility masks."""
    T, F = 10, 3
    cal = np.linspace(10.0, 20.0, T)
    X = np.ones((T, F))
    # Row 0: all observed -> eligible
    # Row 1: 2 of 3 features missing -> ineligible (< 50% observed)
    X[1, 0] = np.nan
    X[1, 1] = np.nan
    
    ds = make_dataset(user_id="user_test", feature_matrix=X, calendar_index=cal)
    
    assert ds.missingness_mask[0, :].sum() == 0
    assert ds.missingness_mask[1, 0] is True or ds.missingness_mask[1, 0] == 1
    assert ds.missingness_mask[1, 1] is True or ds.missingness_mask[1, 1] == 1
    assert ds.eligibility_mask[0] == True
    assert ds.eligibility_mask[1] == False


def test_dataset_deterministic_hash():
    """Test 12: Deterministic hash — same data produces identical hash; any change changes hash."""
    T, F = 35, 2
    cal = np.linspace(1.0, 35.0, T)
    X1 = np.ones((T, F))
    X2 = np.ones((T, F))
    X3 = np.ones((T, F))
    X3[0, 0] = 99.0  # slight change
    
    ds1 = make_dataset(user_id="u1", feature_matrix=X1, calendar_index=cal)
    ds2 = make_dataset(user_id="u1", feature_matrix=X2, calendar_index=cal)
    ds3 = make_dataset(user_id="u1", feature_matrix=X3, calendar_index=cal)
    
    assert ds1.dataset_hash == ds2.dataset_hash
    assert ds1.dataset_hash != ds3.dataset_hash
    assert len(ds1.dataset_hash) == 64  # SHA-256


def test_dataset_anti_leakage_validation():
    """Test 13: Future data leakage validation."""
    T, F = 35, 2
    cal = np.linspace(1.0, 35.0, T)
    X = np.ones((T, F))
    
    # Valid normalization metadata
    ds_valid = make_dataset(
        user_id="u1",
        feature_matrix=X,
        calendar_index=cal,
        normalization_metadata={"normalization_end_index": 20, "uses_future_data": False},
    )
    ds_valid.validate_no_future_leakage()  # Should pass
    
    # Leaking normalization metadata (uses future data)
    ds_leaking = make_dataset(
        user_id="u1",
        feature_matrix=X,
        calendar_index=cal,
        normalization_metadata={"uses_future_data": True},
    )
    with pytest.raises(ValueError, match="future leakage"):
        ds_leaking.validate_no_future_leakage()
    
    # Leaking normalization index beyond dataset length
    ds_idx_leak = make_dataset(
        user_id="u1",
        feature_matrix=X,
        calendar_index=cal,
        normalization_metadata={"normalization_end_index": 100},
    )
    with pytest.raises(ValueError, match="future data leakage"):
        ds_idx_leak.validate_no_future_leakage()


def test_dataset_check_eligibility_gate():
    """Test: check_eligibility_gate raises ColdStartError when eligible count < min."""
    T, F = 20, 2  # 20 < 30 default min
    cal = np.linspace(1.0, 20.0, T)
    X = np.ones((T, F))
    
    ds = make_dataset(user_id="u1", feature_matrix=X, calendar_index=cal, min_present_sessions=30)
    with pytest.raises(ColdStartError, match="eligible sessions < minimum"):
        ds.check_eligibility_gate()
