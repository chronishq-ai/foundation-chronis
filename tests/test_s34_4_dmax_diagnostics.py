import pytest
import numpy as np
from backbone.hssm.model import GaussianHSMM
from backbone.hssm.fitting import compute_dmax_tail_diagnostic, sweep_dmax_sensitivity


def test_compute_dmax_tail_diagnostic():
    model = GaussianHSMM(n_regimes=2, n_features=2, max_duration=40, seed=0)
    X = np.random.default_rng(0).normal(size=(40, 2))
    model.fit(X)

    tail_mass = compute_dmax_tail_diagnostic(model)
    assert len(tail_mass) == 2
    for k in range(2):
        assert 0.0 <= tail_mass[k] <= 1.0
        # dur_logpmf[-1] mass should match exp(logpmf[-1])
        expected_mass = np.exp(model._duration_logpmf()[k, -1])
        np.testing.assert_allclose(tail_mass[k], expected_mass, rtol=1e-5)


def test_sweep_dmax_sensitivity():
    X = np.random.default_rng(1).normal(size=(35, 2))
    report = sweep_dmax_sensitivity(
        X,
        n_regimes=2,
        n_features=2,
        dmax_candidates=(20, 30),
        n_init=10,
        base_seed=0,
    )
    assert 20 in report
    assert 30 in report
    for dmax in (20, 30):
        assert "log_likelihood" in report[dmax]
        assert "bic" in report[dmax]
        assert "tail_mass_by_regime" in report[dmax]
