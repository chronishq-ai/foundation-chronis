"""Tests for the Kim-style HSSM model scaffold."""

import numpy as np

from backbone.hssm.model import KimHSSMModel


def test_model_enforces_lognormal_duration_prior() -> None:
    """The model must explicitly use a log-normal duration prior, not geometric defaults."""
    model = KimHSSMModel(
        n_regimes=2,
        n_features=3,
        transition_matrix=np.array([[0.9, 0.1], [0.2, 0.8]]),
        emission_means=np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
        emission_covariances=[np.eye(3), np.eye(3)],
        duration_mu=np.array([1.5, 2.0]),
        duration_sigma=np.array([0.5, 0.7]),
        duration_prior="lognormal",
    )

    assert model.duration_prior == "lognormal"
    probs = model.duration_probability(np.array([1, 2, 3, 5]), regime_index=0)
    assert probs.shape == (4,)
    assert np.all(np.isfinite(probs))
    assert np.all(probs > 0)


def test_model_rejects_invalid_duration_prior() -> None:
    """Geometric or other unsupported priors should be rejected explicitly."""
    try:
        KimHSSMModel(
            n_regimes=2,
            n_features=2,
            transition_matrix=np.array([[0.8, 0.2], [0.3, 0.7]]),
            emission_means=np.array([[0.0, 0.0], [2.0, 2.0]]),
            emission_covariances=[np.eye(2), np.eye(2)],
            duration_mu=np.array([1.0, 1.2]),
            duration_sigma=np.array([0.4, 0.5]),
            duration_prior="geometric",
        )
    except ValueError:
        return

    raise AssertionError("Unsupported duration priors should raise a ValueError.")
def test_emission_loglik_matches_known_gaussian_value() -> None:
    """emission_loglik must match a manually verified multivariate normal log-pdf."""
    model = KimHSSMModel(
        n_regimes=2,
        n_features=3,
        transition_matrix=np.array([[0.9, 0.1], [0.2, 0.8]]),
        emission_means=np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
        emission_covariances=[np.eye(3), np.eye(3) * 2],
        duration_mu=np.array([1.5, 2.0]),
        duration_sigma=np.array([0.5, 0.7]),
    )
    obs = np.array([0.5, -0.3, 1.2])
    # Reference value computed independently via scipy.stats.multivariate_normal.logpdf
    # for mean=[0,0,0], cov=identity(3)
    expected_loglik = -3.646815599614018
    result = model.emission_loglik(obs, regime_index=0)
    assert np.isclose(result, expected_loglik, atol=1e-10)


def test_emission_loglik_rejects_non_positive_definite_covariance() -> None:
    """A non-positive-definite covariance must raise, not silently produce garbage."""
    bad_covariance = np.array([[1.0, 2.0], [2.0, 1.0]])  # not PD: eigenvalues -1, 3
    model = KimHSSMModel(
        n_regimes=1,
        n_features=2,
        transition_matrix=np.array([[1.0]]),
        emission_means=np.array([[0.0, 0.0]]),
        emission_covariances=[bad_covariance],
        duration_mu=np.array([1.0]),
        duration_sigma=np.array([0.5]),
    )
    try:
        model.emission_loglik(np.array([0.1, 0.1]), regime_index=0)
    except ValueError:
        return
    raise AssertionError("Non-positive-definite covariance should raise ValueError.")


def test_generate_regime_sequence_respects_transition_matrix() -> None:
    """Over many runs, empirical self-transition rate should roughly match the dynamically calculated HMM transition matrix."""
    np.random.seed(42)
    model = KimHSSMModel(
        n_regimes=2,
        n_features=1,
        transition_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
        emission_means=np.array([[0.0], [2.0]]),
        emission_covariances=[np.eye(1), np.eye(1)],
        duration_mu=np.array([1.0, 1.0]),
        duration_sigma=np.array([0.5, 0.5]),
    )
    sequence = model.generate_regime_sequence(length=5000, initial_regime=0)
    # expected duration = exp(1.0 + 0.5**2 / 2) = 3.0802
    # expected self-transition = 1.0 - 1.0 / 3.0802 = 0.6753
    in_regime_0 = sequence[:-1] == 0
    stayed_in_0 = (sequence[:-1] == 0) & (sequence[1:] == 0)
    empirical_self_transition = stayed_in_0.sum() / in_regime_0.sum()
    assert abs(empirical_self_transition - 0.675) < 0.04