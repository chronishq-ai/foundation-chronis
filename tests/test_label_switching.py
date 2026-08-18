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
