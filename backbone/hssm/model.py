"""Kim-style HSSM model scaffold for BACKBONE.

This module defines a minimal, testable Markov-switching state-space model for
capturing a fast latent state m_t and a slow discrete regime p_t, consistent
with the sprint specification for Kim (1994) modelling.

The critical modelling requirement is that the duration prior must be log-normal,
not the default geometric prior used by many off-the-shelf implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class KimHSSMModel:
    """A compact HSSM specification with explicit regime durations.

    Parameters
    ----------
    n_regimes:
        Number of discrete latent regimes K.
    n_features:
        Dimension of observed features for the fast latent state m_t.
    transition_matrix:
        K x K Markov transition matrix.
    emission_means:
        K x n_features matrix of regime-specific emission means.
    emission_covariances:
        Sequence of K covariance matrices for each regime.
    duration_mu:
        Mean parameter for log-normal regime-duration prior.
    duration_sigma:
        Standard deviation parameter for log-normal regime-duration prior.
    duration_prior:
        String describing the duration prior. The implementation requires
        ``"lognormal"`` and rejects geometric or unrecognized options.
    """

    n_regimes: int
    n_features: int
    transition_matrix: np.ndarray
    emission_means: np.ndarray
    emission_covariances: Sequence[np.ndarray]
    duration_mu: np.ndarray
    duration_sigma: np.ndarray
    duration_prior: str = "lognormal"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate model dimensions and model-specific assumptions."""
        if self.n_regimes <= 0:
            raise ValueError("n_regimes must be positive.")
        if self.n_features <= 0:
            raise ValueError("n_features must be positive.")
        if self.duration_prior != "lognormal":
            raise ValueError(
                "This HSSM implementation requires a log-normal duration prior, "
                "not a geometric/default prior."
            )

        self.transition_matrix = np.asarray(self.transition_matrix, dtype=float)
        if self.transition_matrix.shape != (self.n_regimes, self.n_regimes):
            raise ValueError("transition_matrix must be of shape (K, K).")
        if not np.allclose(self.transition_matrix.sum(axis=1), 1.0):
            raise ValueError("Each row of the transition matrix must sum to 1.")
        if np.any(self.transition_matrix < 0.0):
            raise ValueError("Transition probabilities must be non-negative.")

        self.emission_means = np.asarray(self.emission_means, dtype=float)
        if self.emission_means.shape != (self.n_regimes, self.n_features):
            raise ValueError("emission_means must be of shape (K, n_features).")

        if len(self.emission_covariances) != self.n_regimes:
            raise ValueError("emission_covariances must contain one covariance per regime.")

        self.duration_mu = np.asarray(self.duration_mu, dtype=float)
        self.duration_sigma = np.asarray(self.duration_sigma, dtype=float)
        if self.duration_mu.shape != (self.n_regimes,):
            raise ValueError("duration_mu must be a vector of length K.")
        if self.duration_sigma.shape != (self.n_regimes,):
            raise ValueError("duration_sigma must be a vector of length K.")
        if np.any(self.duration_sigma <= 0.0):
            raise ValueError("duration_sigma must be strictly positive.")

        self.metadata.setdefault("model_type", "Kim1994_HSSM")
        self.metadata.setdefault("duration_prior", self.duration_prior)

    def duration_probability(self, durations: Sequence[int] | np.ndarray, regime_index: int) -> np.ndarray:
        """Compute the log-normal duration probability for a given regime.

        The Kim (1994) HSSM uses a duration distribution that is not geometric.
        We encode the prior explicitly as a log-normal distribution for the regime
        duration in observations.

        Parameters
        ----------
        durations:
            Vector of regime durations in integer time steps.
        regime_index:
            Regime whose duration prior should be evaluated.

        Returns
        -------
        np.ndarray
            Probability density for each duration value under the selected regime's
            log-normal prior.
        """
        if not 0 <= regime_index < self.n_regimes:
            raise IndexError("regime_index out of range.")

        durations_arr = np.asarray(durations, dtype=float)
        if np.any(durations_arr <= 0):
            raise ValueError("Durations must be strictly positive integers or floats.")

        mu = float(self.duration_mu[regime_index])
        sigma = float(self.duration_sigma[regime_index])
        pdf = np.exp(-((np.log(durations_arr) - mu) ** 2) / (2.0 * sigma ** 2))
        pdf /= (durations_arr * sigma * np.sqrt(2.0 * np.pi))
        return pdf

    def generate_regime_sequence(self, length: int, initial_regime: int = 0) -> np.ndarray:
        """Generate a simple Markov regime sequence for validation and simulation."""
        if length <= 0:
            raise ValueError("length must be positive.")
        if not 0 <= initial_regime < self.n_regimes:
            raise IndexError("initial_regime out of range.")

        sequence = np.empty(length, dtype=int)
        current_regime = initial_regime
        sequence[0] = current_regime

        for idx in range(1, length):
            probabilities = self.transition_matrix[current_regime, :]
            current_regime = int(np.random.choice(self.n_regimes, p=probabilities))
            sequence[idx] = current_regime

        return sequence

    def emission_loglik(self, observation: np.ndarray, regime_index: int) -> float:
        """Compute Gaussian emission log-likelihood under a regime."""
        if not 0 <= regime_index < self.n_regimes:
            raise IndexError("regime_index out of range.")

        obs = np.asarray(observation, dtype=float)
        if obs.shape != (self.n_features,):
            raise ValueError(f"Observation must have shape ({self.n_features},).")

        covariance = np.asarray(self.emission_covariances[regime_index], dtype=float)
        diff = obs - self.emission_means[regime_index]
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("Emission covariance must be positive definite for the selected regime.")

        precision = np.linalg.inv(covariance)
        quadratic = float(diff.T @ precision @ diff)
        return -0.5 * (self.n_features * np.log(2.0 * np.pi) + logdet + quadratic)
