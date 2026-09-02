import pytest
import numpy as np
from backbone.hssm.model import GaussianHSMM


def test_s34_5_t1_monotonicity_guard_prevents_loglik_decreases():
    """T1: On EM fit fixtures, fit across multiple seeds and verify that the monotonicity guard
    rejects any non-monotonic log-likelihood decrease, ensuring strict monotonicity."""
    rng = np.random.default_rng(42)
    for seed in range(5):
        X = rng.normal(size=(50, 3))
        model = GaussianHSMM(n_regimes=3, n_features=3, seed=seed)
        model.fit(X, n_iter=50)
        
        is_mono = model.is_log_likelihood_monotonic(tol=1e-4)
        assert is_mono, f"EM log-likelihood must be strictly monotonic under the guard for seed {seed}"


def test_s34_5_t2_monotonicity_ci_check():
    """T2: Wire this into CI as a standing regression test (runs every push).
    This verifies that the monotonicity measurement capability is present on the model class."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    model = GaussianHSMM(n_regimes=2, n_features=2, seed=0)
    model.fit(X, n_iter=20)
    assert hasattr(model, "is_log_likelihood_monotonic"), "Model must have is_log_likelihood_monotonic method wired into CI"
    assert model.is_log_likelihood_monotonic(tol=1e-4)
