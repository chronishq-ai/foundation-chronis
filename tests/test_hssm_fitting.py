"""Tests for the HSSM EM fitting harness."""

from __future__ import annotations

import numpy as np

from backbone.hssm.fitting import fit_hssm_model
from backbone.hssm.model import KimHSSMModel


def _make_true_model() -> KimHSSMModel:
    return KimHSSMModel(
        n_regimes=2,
        n_features=2,
        transition_matrix=np.array([[0.90, 0.10], [0.08, 0.92]], dtype=float),
        emission_means=np.array([
            [0.0, 0.0],
            [4.0, 4.0],
        ], dtype=float),
        emission_covariances=[
            np.eye(2, dtype=float),
            np.eye(2, dtype=float),
        ],
        duration_mu=np.array([1.2, 1.8], dtype=float),
        duration_sigma=np.array([0.35, 0.45], dtype=float),
        duration_prior="lognormal",
    )


def _generate_markov_observations(true_model: KimHSSMModel, length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    regime_sequence = true_model.generate_regime_sequence(length=length, initial_regime=0)
    observations = np.empty((length, 2), dtype=float)
    for t, regime in enumerate(regime_sequence):
        observations[t] = rng.multivariate_normal(
            mean=true_model.emission_means[regime],
            cov=true_model.emission_covariances[regime],
        )
    return observations


def _generate_duration_dominant_observations(true_mu: np.ndarray, true_sigma: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    path: list[int] = []
    while len(path) < 260:
        for regime_idx, (mu, sigma) in enumerate(zip(true_mu, true_sigma)):
            duration = max(2, int(round(np.random.lognormal(mean=mu, sigma=sigma, size=None))))
            path.extend([regime_idx] * duration)
            if len(path) >= 260:
                break
    path = path[:260]
    observations = np.empty((len(path), 2), dtype=float)
    means = np.array([[0.0, 0.0], [4.0, 4.0]], dtype=float)
    for t, regime_idx in enumerate(path):
        observations[t] = rng.multivariate_normal(mean=means[regime_idx], cov=np.eye(2))
    return observations


def test_fit_hssm_recovers_true_k_via_bic() -> None:
    true_model = _make_true_model()
    observations = _generate_markov_observations(true_model, 300, seed=7)

    model, report = fit_hssm_model(
        observations,
        candidate_ks=(2, 3, 4),
        n_initializations=10,
        max_iter=80,
        tol=1e-4,
        random_seed=13,
    )

    assert isinstance(model, KimHSSMModel)
    assert model.duration_prior == "lognormal"
    assert report["k_selected"] == 2
    assert report["bic_by_k"][2] < report["bic_by_k"][3]
    assert report["convergence_rate_by_k"][2] >= 0.0
    assert report["convergence_rate_by_k"][2] <= 1.0


def test_fit_hssm_transition_matrix_recovers_true_dynamics() -> None:
    true_model = _make_true_model()
    observations = _generate_markov_observations(true_model, 500, seed=11)

    fitted_model, _ = fit_hssm_model(
        observations,
        candidate_ks=(2,),
        n_initializations=10,
        max_iter=80,
        tol=1e-4,
        random_seed=13,
    )

    assert np.allclose(fitted_model.transition_matrix.sum(axis=1), 1.0, atol=1e-8)
    assert np.max(np.abs(fitted_model.transition_matrix - true_model.transition_matrix)) < 0.10


def test_fit_hssm_duration_parameters_learn_from_data() -> None:
    true_mu = np.array([1.2, 1.8], dtype=float)
    true_sigma = np.array([0.35, 0.45], dtype=float)
    init_mu = np.linspace(0.5, 2.0, 2)
    init_sigma = np.array([0.5, 0.5], dtype=float)

    observations = _generate_duration_dominant_observations(true_mu, true_sigma, seed=13)
    fitted_model, _ = fit_hssm_model(
        observations,
        candidate_ks=(2,),
        n_initializations=10,
        max_iter=80,
        tol=1e-4,
        random_seed=13,
    )

    fitted_mean_distance = np.mean(np.abs(fitted_model.duration_mu - true_mu))
    init_mean_distance = np.mean(np.abs(init_mu - true_mu))
    fitted_sigma_distance = np.mean(np.abs(fitted_model.duration_sigma - true_sigma))
    init_sigma_distance = np.mean(np.abs(init_sigma - true_sigma))

    assert fitted_mean_distance < init_mean_distance
    assert fitted_sigma_distance < init_sigma_distance


def test_fit_report_reports_convergence_rate() -> None:
    true_model = _make_true_model()
    observations = _generate_markov_observations(true_model, 220, seed=17)

    _, report = fit_hssm_model(
        observations,
        candidate_ks=(2, 3),
        n_initializations=10,
        max_iter=60,
        tol=1e-4,
        random_seed=13,
    )

    for k in (2, 3):
        conv_rate = report["convergence_rate_by_k"][k]
        assert 0.0 <= conv_rate <= 1.0
        assert k in report["loglik_by_k"]
        assert k in report["bic_by_k"]


def test_more_random_inits_does_not_change_selection_logic() -> None:
    true_model = _make_true_model()
    observations = _generate_markov_observations(true_model, 260, seed=19)

    _, report_10 = fit_hssm_model(
        observations,
        candidate_ks=(2, 3),
        n_initializations=10,
        max_iter=60,
        tol=1e-4,
        random_seed=13,
    )
    _, report_3 = fit_hssm_model(
        observations,
        candidate_ks=(2, 3),
        n_initializations=3,
        max_iter=60,
        tol=1e-4,
        random_seed=13,
    )

    assert report_10["k_selected"] == report_3["k_selected"]
    assert report_10["selected_loglik"] >= report_3["selected_loglik"] - 1e-9


def test_fit_hssm_is_reproducible_for_same_seed() -> None:
    true_model = _make_true_model()
    observations = _generate_markov_observations(true_model, 320, seed=23)

    fitted_a, report_a = fit_hssm_model(
        observations,
        candidate_ks=(2,),
        n_initializations=10,
        max_iter=80,
        tol=1e-4,
        random_seed=77,
    )
    fitted_b, report_b = fit_hssm_model(
        observations,
        candidate_ks=(2,),
        n_initializations=10,
        max_iter=80,
        tol=1e-4,
        random_seed=77,
    )

    assert report_a["k_selected"] == report_b["k_selected"]
    assert np.allclose(fitted_a.transition_matrix, fitted_b.transition_matrix)
    assert np.allclose(fitted_a.duration_mu, fitted_b.duration_mu)
    assert np.allclose(fitted_a.duration_sigma, fitted_b.duration_sigma)
