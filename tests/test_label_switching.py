"""Tests for the canonical regime-order fix."""

from __future__ import annotations

import numpy as np

from backbone.hssm.label_switching import canonicalize_regime_order
from backbone.hssm.model import KimHSSMModel


def test_regimes_are_sorted_by_activity_level() -> None:
    model = KimHSSMModel(
        n_regimes=3,
        n_features=2,
        transition_matrix=np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ], dtype=float),
        emission_means=np.array([
            [5.0, 5.0],
            [0.0, 0.0],
            [2.0, 2.0],
        ], dtype=float),
        emission_covariances=[
            np.eye(2),
            np.eye(2),
            np.eye(2),
        ],
        duration_mu=np.array([2.2, 1.0, 1.5], dtype=float),
        duration_sigma=np.array([0.4, 0.3, 0.5], dtype=float),
        duration_prior="lognormal",
    )

    fixed = canonicalize_regime_order(model)
    activity_scores = [float(np.mean(np.abs(row))) for row in fixed.emission_means]
    assert activity_scores == sorted(activity_scores)
    assert np.allclose(fixed.emission_means[0], np.array([0.0, 0.0]))
    assert np.allclose(fixed.emission_means[-1], np.array([5.0, 5.0]))


def test_transition_and_duration_vectors_follow_ordering() -> None:
    model = KimHSSMModel(
        n_regimes=2,
        n_features=1,
        transition_matrix=np.array([
            [0.2, 0.8],
            [0.7, 0.3],
        ], dtype=float),
        emission_means=np.array([
            [3.0],
            [1.0],
        ], dtype=float),
        emission_covariances=[
            np.array([[1.0]]),
            np.array([[1.0]]),
        ],
        duration_mu=np.array([2.0, 1.0], dtype=float),
        duration_sigma=np.array([0.5, 0.2], dtype=float),
        duration_prior="lognormal",
    )

    fixed = canonicalize_regime_order(model)
    assert np.allclose(fixed.duration_mu, np.array([1.0, 2.0]))
    assert np.allclose(fixed.duration_sigma, np.array([0.2, 0.5]))
    assert fixed.transition_matrix.shape == (2, 2)


def test_canonicalization_preserves_fitted_semantics_and_permutations() -> None:
    model = KimHSSMModel(
        n_regimes=3,
        n_features=2,
        transition_matrix=np.array([
            [0.0, 0.2, 0.8],
            [0.6, 0.0, 0.4],
            [0.3, 0.7, 0.0],
        ], dtype=float),
        emission_means=np.array([
            [3.0, 4.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ], dtype=float),
        emission_covariances=[np.eye(2), np.eye(2), np.eye(2)],
        duration_mu=np.array([2.0, 1.0, 1.5], dtype=float),
        duration_sigma=np.array([0.5, 0.2, 0.3], dtype=float),
        duration_prior="lognormal",
    )
    model.log_likelihood_ = -12.5
    original_pi = model.pi.copy()
    original_A = model.A.copy()
    original_mu = model.mu.copy()
    original_var = model.var.copy()
    original_dur_mu = model.dur_mu.copy()
    original_dur_sigma = model.dur_sigma.copy()
    original_fit_state = model.fit_state_

    fixed = canonicalize_regime_order(model)
    order = np.argsort(np.linalg.norm(original_mu, axis=1))

    assert fixed is model
    assert fixed.fit_state_ == original_fit_state
    assert fixed.log_likelihood_ == -12.5
    assert fixed._label_order_applied == order.tolist()
    assert np.allclose(fixed.pi, original_pi[order])
    assert np.allclose(fixed.A, original_A[np.ix_(order, order)])
    assert np.allclose(fixed.mu, original_mu[order])
    assert np.allclose(fixed.var, original_var[order])
    assert np.allclose(fixed.dur_mu, original_dur_mu[order])
    assert np.allclose(fixed.dur_sigma, original_dur_sigma[order])
    assert np.allclose(fixed.A.sum(axis=1), 1.0)
