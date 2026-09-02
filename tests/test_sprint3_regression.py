import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pytest

from backbone.hssm.model import KimHSSMModel, NotFittedError
from backbone.hssm.fitting import fit_hssm_model


def test_generate_regime_sequence_respects_duration_prior():
    """Verify that generate_regime_sequence holds regimes for durations drawn from the lognormal prior."""
    # Regime 0 has very short expected duration (mu=0.2), Regime 1 has long expected duration (mu=3.0)
    model = KimHSSMModel(
        n_regimes=2,
        n_features=1,
        transition_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
        emission_means=np.array([[0.0], [2.0]]),
        emission_covariances=[np.eye(1), np.eye(1)],
        duration_mu=np.array([0.2, 3.0]),
        duration_sigma=np.array([0.05, 0.05]),
        seed=42,
    )
    
    sequence = model.generate_regime_sequence(length=2000, initial_regime=0)
    
    # Calculate empirical mean run lengths for both regimes
    runs = []
    current_val = sequence[0]
    current_len = 1
    for val in sequence[1:]:
        if val == current_val:
            current_len += 1
        else:
            runs.append((current_val, current_len))
            current_val = val
            current_len = 1
    runs.append((current_val, current_len))
    
    run_lengths = {0: [], 1: []}
    for val, length in runs:
        run_lengths[val].append(length)
        
    mean_0 = np.mean(run_lengths[0]) if run_lengths[0] else 0.0
    mean_1 = np.mean(run_lengths[1]) if run_lengths[1] else 0.0
    
    # Verify that Regime 1's average run length is much longer than Regime 0's (at least 3x)
    assert mean_0 > 0
    assert mean_1 > 0
    assert mean_1 >= 3 * mean_0, f"Expected long regime dwell to be at least 3x short regime, got {mean_1:.2f} vs {mean_0:.2f}"
    
    # Assert they are close to theoretical expectations:
    # exp(0.2 + 0.05^2/2) = 1.22
    # exp(3.0 + 0.05^2/2) = 20.11
    assert abs(mean_0 - 1.22) < 0.5
    assert abs(mean_1 - 20.11) < 4.0


def test_fit_hssm_model_enforces_hard_min_init_by_default():
    """Verify fit_hssm_model enforces n_initializations >= 10 by default, and allows bypass if specified."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(35, 2))
    
    # Without allow_fast_test_fit (default), n_initializations < 10 must raise ValueError or AssertionError
    with pytest.raises((ValueError, AssertionError)):
        fit_hssm_model(X, candidate_ks=(2,), n_initializations=2, allow_fast_test_fit=False)
        
    # With allow_fast_test_fit=True, it should proceed normally
    model, report = fit_hssm_model(X, candidate_ks=(2,), n_initializations=2, allow_fast_test_fit=True)
    assert model is not None
    assert report["n_initializations"] == 2


def test_unfitted_model_raises_not_fitted_error():
    """Verify that calling properties or methods on an unfitted model raises a NotFittedError."""
    from backbone.hssm.model import GaussianHSMM
    model = GaussianHSMM(n_regimes=2, n_features=3)
    
    with pytest.raises(NotFittedError, match="Model has not been fitted yet"):
        _ = model.transition_matrix
        
    with pytest.raises(NotFittedError, match="Model has not been fitted yet"):
        _ = model.duration_mu
        
    with pytest.raises(NotFittedError, match="Model has not been fitted yet"):
        _ = model.duration_sigma
        
    with pytest.raises(NotFittedError, match="Model has not been fitted yet"):
        _ = model.emission_means
        
    with pytest.raises(NotFittedError, match="Model has not been fitted yet"):
        _ = model.emission_covariances
        
    with pytest.raises(NotFittedError, match="Model has not been fitted yet"):
        model.duration_probability([1, 2], 0)
        
    with pytest.raises(NotFittedError, match="Model has not been fitted yet"):
        model.generate_regime_sequence(10)

    with pytest.raises(NotFittedError, match="Model has not been fitted yet"):
        model.bic(10)

    with pytest.raises(NotFittedError, match="Model has not been fitted yet"):
        model.n_params()
