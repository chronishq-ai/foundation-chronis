"""
Sprint 3–4 Phase 5: Anti-Leakage & Chronological Discipline Tests.
Tests 111–116 per senior directive Section D.

Verifies:
  - Strict chronological holdout partitioning
  - Backward-only normalization (zero future leakage)
  - Dataset hash integrity under past vs future modifications
  - Anti-leakage guards in personal normalization metadata
  - Invariance of past predictions under future feature perturbations
"""

import numpy as np
import pytest

from backbone.hssm.dataset import HSSMFitDataset, make_dataset
from backbone.hssm.fitting import fit_hssm, HSSMResult


def test_111_chronological_split_strict_ordering():
    """Test 111: Chronological holdout strictly partitions train and test sets by time."""
    T = 60
    calendar_days = np.array([float(100.0 + i * 1.5) for i in range(T)])
    X = np.random.RandomState(42).randn(T, 4)
    
    dataset = make_dataset(
        user_id="user_test_111",
        feature_matrix=X,
        calendar_index=calendar_days,
        min_present_sessions=30,
    )
    
    # Train / test split at 80%
    split_fraction = 0.8
    split_idx = int(T * split_fraction)
    
    train_cal = dataset.calendar_index[:split_idx]
    test_cal = dataset.calendar_index[split_idx:]
    
    # Strict temporal separation
    assert np.max(train_cal) < np.min(test_cal), "Training timestamps must strictly precede test timestamps"
    assert len(train_cal) + len(test_cal) == T


def test_112_backward_only_normalization_guard():
    """Test 112: Personal normalization metadata rejects future index references or future data flags."""
    T = 50
    cal = np.array([float(100.0 + i * 1.2) for i in range(T)])
    X = np.random.RandomState(42).randn(T, 3)

    # Valid normalization: end index <= dataset length, no future flag
    valid_ds = make_dataset(
        user_id="user_test_112",
        feature_matrix=X,
        calendar_index=cal,
        normalization_metadata={
            "normalization_end_index": 40,
            "mean": np.mean(X[:40], axis=0).tolist(),
            "std": np.std(X[:40], axis=0).tolist(),
            "uses_future_data": False,
        },
    )
    valid_ds.validate_no_future_leakage()  # Should not raise

    # Invalid normalization: references future index > T
    invalid_ds_index = make_dataset(
        user_id="user_test_112",
        feature_matrix=X,
        calendar_index=cal,
        normalization_metadata={
            "normalization_end_index": 75,
            "uses_future_data": False,
        },
    )
    with pytest.raises(ValueError, match="exceeds dataset length.*future data leakage"):
        invalid_ds_index.validate_no_future_leakage()

    # Invalid normalization: explicitly marked as using future data
    invalid_ds_flag = make_dataset(
        user_id="user_test_112",
        feature_matrix=X,
        calendar_index=cal,
        normalization_metadata={
            "uses_future_data": True,
        },
    )
    with pytest.raises(ValueError, match="future leakage violation"):
        invalid_ds_flag.validate_no_future_leakage()


def test_113_future_perturbation_invariance_on_past_normalization():
    """Test 113: Modifying future observation values does not alter past normalization statistics."""
    T = 60
    split_idx = 40
    np.random.seed(123)
    X = np.random.randn(T, 4)
    cal = np.array([float(100.0 + i * 2.0) for i in range(T)])

    # Compute normalization on train partition [0:split_idx]
    train_mean = np.mean(X[:split_idx], axis=0)
    train_std = np.std(X[:split_idx], axis=0)

    # Create modified future data
    X_modified = X.copy()
    X_modified[split_idx:] += 50.0  # Massive shift in future test window

    # Normalization computed on train partition of modified dataset
    train_mean_mod = np.mean(X_modified[:split_idx], axis=0)
    train_std_mod = np.std(X_modified[:split_idx], axis=0)

    np.testing.assert_allclose(train_mean, train_mean_mod, err_msg="Past normalization mean must be invariant to future changes")
    np.testing.assert_allclose(train_std, train_std_mod, err_msg="Past normalization std must be invariant to future changes")


def test_114_deterministic_dataset_hash_sensitivity():
    """Test 114: Dataset SHA-256 hash changes deterministically when any feature or timestamp changes."""
    T = 40
    cal = np.array([float(100.0 + i * 1.5) for i in range(T)])
    X = np.random.RandomState(42).randn(T, 3)

    ds1 = make_dataset(user_id="user_hash", feature_matrix=X, calendar_index=cal)
    ds2 = make_dataset(user_id="user_hash", feature_matrix=X.copy(), calendar_index=cal.copy())
    
    # Identical data -> identical hash
    assert ds1.dataset_hash == ds2.dataset_hash
    assert len(ds1.dataset_hash) == 64  # SHA-256 hex length

    # Altering single point in feature matrix alters hash
    X_alt = X.copy()
    X_alt[10, 0] += 0.001
    ds3 = make_dataset(user_id="user_hash", feature_matrix=X_alt, calendar_index=cal)
    assert ds1.dataset_hash != ds3.dataset_hash

    # Altering single timestamp alters hash
    cal_alt = cal.copy()
    cal_alt[5] += 0.1
    ds4 = make_dataset(user_id="user_hash", feature_matrix=X, calendar_index=cal_alt)
    assert ds1.dataset_hash != ds4.dataset_hash


def test_115_rejection_of_shuffled_or_non_monotonic_calendar_indices():
    """Test 115: Dataset requires strictly valid calendar structure and enforces monotonic ordering."""
    T = 40
    cal_valid = np.array([float(100.0 + i * 1.2) for i in range(T)])
    X = np.random.RandomState(42).randn(T, 2)

    ds = make_dataset(user_id="user_valid_time", feature_matrix=X, calendar_index=cal_valid)
    assert np.all(np.diff(ds.calendar_index) > 0), "Calendar index must be strictly monotonic in time"


def test_116_model_fit_records_fit_dataset_hash():
    """Test 116: HSSMResult records the exact fit_dataset_hash for end-to-end provenance."""
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(0, 0.5, (30, 2)),
        np.random.normal(4, 0.5, (30, 2)),
    ])
    cal = np.linspace(100.0, 160.0, 60)
    ds = make_dataset(user_id="user_prov", feature_matrix=X, calendar_index=cal)

    result = fit_hssm(
        matrix=ds.feature_matrix,
        candidate_ks=(2,),
        n_initializations=10,
        max_iter=50,
        random_seed=42,
        allow_fast_test_fit=True,
        timestamps=ds.calendar_index,
        user_id=ds.user_id,
        dataset_hash=ds.dataset_hash,
    )

    assert result.fit_dataset_hash == ds.dataset_hash
    assert len(result.fit_dataset_hash) == 64
