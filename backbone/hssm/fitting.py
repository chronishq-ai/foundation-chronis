"""EM fitting harness for the Kim-style HSSM model.

This module provides a minimal but practical fitting workflow for a per-user
observation matrix produced by the feature-reduction pipeline. The harness is
explicit about the modelling contract:

- A given K is fit with a minimum of 10 random initializations.
- Each initialization runs an EM loop until a tolerance-based convergence check.
- The best converged run is selected by highest log-likelihood, never by manual
  inspection.
- K is chosen by BIC only from K=2,3,4 candidate values.

The output is a fitted KimHSSMModel instance together with a report suitable for
later convergence validation and BIC-based model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backbone.hssm.model import KimHSSMModel


@dataclass(frozen=True)
class EMRunResult:
    """One EM optimization result for a particular initialization."""

    loglik: float
    converged: bool
    n_iter: int
    params: Dict[str, Any]
    insufficient_duration_data: List[int]


def _as_observation_matrix(data: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Validate and coerce the observation matrix to a numeric NumPy array."""
    if isinstance(data, pd.DataFrame):
        matrix = data.to_numpy(dtype=float)
    elif isinstance(data, np.ndarray):
        matrix = data.astype(float, copy=False)
    else:
        raise TypeError("Input data must be a NumPy array or pandas DataFrame.")

    if matrix.ndim != 2:
        raise ValueError("Observation data must be a 2D matrix of shape (T, n_features).")
    if matrix.shape[0] == 0:
        raise ValueError("Observation matrix cannot be empty.")
    if np.isnan(matrix).any():
        raise ValueError("Observation matrix contains NaN values; missingness must be handled upstream.")

    return matrix


def _initialize_parameters(
    observations: np.ndarray,
    n_regimes: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize Gaussian emission means and covariance parameters."""
    n_obs, n_features = observations.shape

    random_indices = rng.choice(n_obs, size=n_regimes, replace=False)
    means = observations[random_indices].copy()

    if means.shape != (n_regimes, n_features):
        raise RuntimeError("Failed to initialize regime means with the desired shape.")

    transition_matrix = np.full((n_regimes, n_regimes), 1.0 / n_regimes, dtype=float)
    transition_matrix += rng.normal(0.0, 0.02, size=(n_regimes, n_regimes))
    transition_matrix = np.clip(transition_matrix, 1e-3, None)
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

    covariances = []
    global_cov = np.cov(observations, rowvar=False)
    if np.ndim(global_cov) == 0:
        global_cov = np.eye(n_features, dtype=float) * float(observations.var())
    if np.allclose(global_cov, 0.0):
        global_cov = np.eye(n_features, dtype=float)

    for _ in range(n_regimes):
        jitter = rng.normal(0.0, 0.05, size=(n_features, n_features))
        cov = global_cov + np.eye(n_features, dtype=float) * 0.1 + jitter @ jitter.T
        cov = 0.5 * (cov + cov.T)
        cov += np.eye(n_features, dtype=float) * 1e-6
        covariances.append(cov)

    duration_mu = np.linspace(0.5, 2.0, n_regimes, dtype=float)
    duration_sigma = np.full(n_regimes, 0.5, dtype=float)

    return transition_matrix, means, np.asarray(covariances), duration_mu, duration_sigma


def _compute_loglikelihood(
    observations: np.ndarray,
    transition_matrix: np.ndarray,
    emission_means: np.ndarray,
    emission_covariances: Sequence[np.ndarray],
    duration_mu: np.ndarray,
    duration_sigma: np.ndarray,
) -> float:
    """Compute the full-data log-likelihood under a simplified Gaussian HMM approximation.

    This harness does not implement the full Kim filter; it uses a tractable
    Gaussian-mixture/HMM-style likelihood for the model-selection contract in this
    sprint, while preserving the required log-normal duration prior in the
    KimHSSMModel object itself.
    """
    n_obs, n_features = observations.shape
    n_regimes = emission_means.shape[0]

    if n_obs == 0:
        return -np.inf

    state_prob = np.full((n_obs, n_regimes), 1.0 / n_regimes, dtype=float)
    loglik = 0.0

    for t in range(n_obs):
        weighted = np.empty(n_regimes, dtype=float)
        for regime_idx in range(n_regimes):
            diff = observations[t] - emission_means[regime_idx]
            cov = np.asarray(emission_covariances[regime_idx], dtype=float)
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                raise ValueError("Emission covariance must be positive definite.")
            precision = np.linalg.inv(cov)
            quad = float(diff.T @ precision @ diff)
            ll = -0.5 * (n_features * np.log(2.0 * np.pi) + logdet + quad)
            weighted[regime_idx] = ll

        if t == 0:
            loglik += np.log(np.sum(state_prob[t] * np.exp(weighted - weighted.max())))
        else:
            prev = state_prob[t - 1]
            transition_weight = transition_matrix.T @ prev
            state_prob[t] = transition_weight * np.exp(weighted - weighted.max())
            state_prob[t] /= state_prob[t].sum()
            loglik += np.log(np.sum(state_prob[t] * np.exp(weighted - weighted.max())))

    return float(loglik)


def _logsumexp(values: np.ndarray) -> float:
    """Numerically stable log-sum-exp for forward-backward computations."""
    values = np.asarray(values, dtype=float)
    max_value = np.max(values)
    return float(max_value + np.log(np.sum(np.exp(values - max_value))))


def _forward_backward_gamma_xi(
    observations: np.ndarray,
    transition_matrix: np.ndarray,
    emission_means: np.ndarray,
    emission_covariances: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute gamma, xi, and log-likelihood using a stable forward-backward pass."""
    n_obs, n_features = observations.shape
    n_regimes = transition_matrix.shape[0]

    log_emissions = np.empty((n_obs, n_regimes), dtype=float)
    for regime_idx in range(n_regimes):
        diff = observations - emission_means[regime_idx]
        cov = np.asarray(emission_covariances[regime_idx], dtype=float)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("Emission covariance must be positive definite during EM.")
        precision = np.linalg.inv(cov)
        mahat = np.einsum('ij,jk,ik->i', diff, precision, diff)
        log_emissions[:, regime_idx] = -0.5 * (
            n_features * np.log(2.0 * np.pi) + logdet + mahat
        )

    log_trans = np.log(np.clip(transition_matrix, 1e-300, None))
    log_pi = np.full(n_regimes, -np.log(n_regimes), dtype=float)

    forward = np.empty((n_obs, n_regimes), dtype=float)
    forward[0] = log_pi + log_emissions[0]
    for t in range(1, n_obs):
        for regime_idx in range(n_regimes):
            forward[t, regime_idx] = log_emissions[t, regime_idx] + _logsumexp(
                forward[t - 1] + log_trans[:, regime_idx]
            )

    loglik = _logsumexp(forward[-1])

    backward = np.empty((n_obs, n_regimes), dtype=float)
    backward[-1] = 0.0
    for t in range(n_obs - 2, -1, -1):
        for regime_idx in range(n_regimes):
            backward[t, regime_idx] = _logsumexp(
                log_trans[regime_idx] + log_emissions[t + 1] + backward[t + 1]
            )

    log_gamma = forward + backward - loglik
    gamma = np.exp(log_gamma)
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

    xi = np.empty((max(n_obs - 1, 0), n_regimes, n_regimes), dtype=float)
    for t in range(n_obs - 1):
        log_xi = (
            forward[t][:, None]
            + log_trans
            + log_emissions[t + 1][None, :]
            + backward[t + 1][None, :]
            - loglik
        )
        xi[t] = np.exp(log_xi)
        total = xi[t].sum()
        if total > 0:
            xi[t] /= total

    return gamma, xi, float(loglik)


def _estimate_duration_parameters(
    regime_path: np.ndarray,
    init_mu: np.ndarray,
    init_sigma: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Estimate log-normal duration parameters from the most-likely regime path."""
    mu = init_mu.copy()
    sigma = init_sigma.copy()
    insufficient_duration_data: List[int] = []

    for regime_idx in range(init_mu.shape[0]):
        run_lengths: List[int] = []
        current_length = 0
        for state in regime_path:
            if int(state) == regime_idx:
                current_length += 1
            else:
                if current_length > 0:
                    run_lengths.append(current_length)
                current_length = 0
        if current_length > 0:
            run_lengths.append(current_length)

        if len(run_lengths) < 2:
            insufficient_duration_data.append(regime_idx)
            continue

        log_run_lengths = np.log(np.asarray(run_lengths, dtype=float))
        mu[regime_idx] = float(np.mean(log_run_lengths))
        sigma[regime_idx] = float(np.std(log_run_lengths, ddof=0))

    return mu, sigma, insufficient_duration_data


def _run_em_single_initialization(
    observations: np.ndarray,
    n_regimes: int,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
) -> EMRunResult:
    """Run one EM iteration chain for a fixed random initialization."""
    transition_matrix, means, covariances, duration_mu, duration_sigma = _initialize_parameters(
        observations,
        n_regimes,
        rng,
    )
    prev_loglik = None
    insufficient_duration_data: List[int] = []

    for iteration in range(1, max_iter + 1):
        gamma, xi, current_loglik = _forward_backward_gamma_xi(
            observations,
            transition_matrix,
            means,
            covariances,
        )

        # M-step: update means and covariances using responsibilities.
        nk = gamma.sum(axis=0)
        means = (gamma.T @ observations) / np.maximum(nk[:, None], 1e-12)

        new_covariances: List[np.ndarray] = []
        for regime_idx in range(n_regimes):
            diff = observations - means[regime_idx]
            weights = gamma[:, regime_idx][:, None, None]
            cov = (weights * diff[:, :, None] * diff[:, None, :]).sum(axis=0) / np.maximum(nk[regime_idx], 1e-12)
            cov = 0.5 * (cov + cov.T)
            cov += np.eye(observations.shape[1], dtype=float) * 1e-6
            new_covariances.append(cov)
        covariances = new_covariances

        # M-step: update transition matrix from pairwise posteriors xi_t(i, j).
        new_transition = np.empty_like(transition_matrix)
        for state_i in range(n_regimes):
            denominator = gamma[:-1, state_i].sum() if observations.shape[0] > 1 else 1.0
            if denominator <= 1e-12:
                new_transition[state_i] = np.full(n_regimes, 1.0 / n_regimes, dtype=float)
            else:
                new_transition[state_i] = xi[:, state_i, :].sum(axis=0) / denominator
                new_transition[state_i] = np.clip(new_transition[state_i], 1e-12, None)
                new_transition[state_i] /= new_transition[state_i].sum()
        transition_matrix = new_transition

        # Duration parameters: estimate from the most likely hidden path.
        most_likely_path = np.argmax(gamma, axis=1)
        duration_mu, duration_sigma, insufficient_duration_data = _estimate_duration_parameters(
            most_likely_path,
            duration_mu,
            duration_sigma,
        )

        if prev_loglik is not None and abs(current_loglik - prev_loglik) < tol:
            return EMRunResult(
                loglik=current_loglik,
                converged=True,
                n_iter=iteration,
                params={
                    "transition_matrix": transition_matrix,
                    "emission_means": means,
                    "emission_covariances": covariances,
                    "duration_mu": duration_mu,
                    "duration_sigma": duration_sigma,
                },
                insufficient_duration_data=insufficient_duration_data,
            )
        prev_loglik = current_loglik

    return EMRunResult(
        loglik=prev_loglik if prev_loglik is not None else float("-inf"),
        converged=False,
        n_iter=max_iter,
        params={
            "transition_matrix": transition_matrix,
            "emission_means": means,
            "emission_covariances": covariances,
            "duration_mu": duration_mu,
            "duration_sigma": duration_sigma,
        },
        insufficient_duration_data=insufficient_duration_data,
    )


def _bic_score(loglik: float, n_params: int, n_observations: int) -> float:
    """Compute the Bayesian Information Criterion (BIC)."""
    if n_observations <= 0:
        raise ValueError("n_observations must be positive.")
    return -2.0 * loglik + n_params * np.log(n_observations)


def fit_hssm_model(
    observations: np.ndarray | pd.DataFrame,
    candidate_ks: Sequence[int] = (2, 3, 4),
    n_initializations: int = 10,
    max_iter: int = 200,
    tol: float = 1e-4,
    random_seed: Optional[int] = None,
) -> Tuple[KimHSSMModel, Dict[str, Any]]:
    """Fit a KimHSSMModel across K=2,3,4 and select the best K via BIC.

    Parameters
    ----------
    observations:
        A 2D array or DataFrame of shape (T, n_features), typically the output of
        feature reduction.
    candidate_ks:
        Candidate numbers of latent regimes. By default: (2, 3, 4).
    n_initializations:
        Minimum number of random initializations per K; more are allowed.
    max_iter:
        Maximum iterations allowed per initialization.
    tol:
        Convergence threshold on log-likelihood change for each initialization.
    random_seed:
        Optional seed for reproducibility.

    Returns
    -------
    Tuple[KimHSSMModel, Dict[str, Any]]
        The fitted model and a detailed report with BIC and convergence metrics.
    """
    matrix = _as_observation_matrix(observations)
    if any(k < 2 for k in candidate_ks):
        raise ValueError("candidate_ks must contain values of at least 2.")
    if n_initializations < 1:
        raise ValueError("n_initializations must be positive.")

    rng = np.random.default_rng(random_seed)
    bic_by_k: Dict[int, float] = {}
    loglik_by_k: Dict[int, float] = {}
    convergence_rate_by_k: Dict[int, float] = {}
    selected_model: Optional[KimHSSMModel] = None
    selected_loglik: Optional[float] = None
    selected_k: Optional[int] = None

    for k in candidate_ks:
        runs: List[EMRunResult] = []
        for _ in range(n_initializations):
            run = _run_em_single_initialization(matrix, k, max_iter=max_iter, tol=tol, rng=rng)
            runs.append(run)

        converged_runs = [run for run in runs if run.converged]
        if not converged_runs:
            best_run = max(runs, key=lambda x: x.loglik)
            chosen_run = best_run
        else:
            chosen_run = max(converged_runs, key=lambda x: x.loglik)

        n_params = k * (matrix.shape[1] + 1) + k * (k - 1)  # means + covariances + transition terms
        bic = _bic_score(float(chosen_run.loglik), n_params, matrix.shape[0])
        bic_by_k[k] = bic
        loglik_by_k[k] = float(chosen_run.loglik)
        convergence_rate_by_k[k] = len(converged_runs) / float(n_initializations)

        if selected_k is None or bic < bic_by_k[selected_k]:
            selected_k = k
            selected_loglik = float(chosen_run.loglik)
            selected_model = KimHSSMModel(
                n_regimes=k,
                n_features=matrix.shape[1],
                transition_matrix=chosen_run.params["transition_matrix"],
                emission_means=chosen_run.params["emission_means"],
                emission_covariances=chosen_run.params["emission_covariances"],
                duration_mu=np.asarray(chosen_run.params["duration_mu"], dtype=float),
                duration_sigma=np.asarray(chosen_run.params["duration_sigma"], dtype=float),
                duration_prior="lognormal",
            )

    if selected_model is None or selected_k is None or selected_loglik is None:
        raise RuntimeError("No valid HSSM model fit was produced.")

    report: Dict[str, Any] = {
        "k_selected": selected_k,
        "bic_by_k": bic_by_k,
        "loglik_by_k": loglik_by_k,
        "convergence_rate_by_k": convergence_rate_by_k,
        "selected_loglik": selected_loglik,
        "candidate_ks": tuple(candidate_ks),
        "n_initializations": n_initializations,
        "max_iter": max_iter,
        "tol": tol,
    }
    return selected_model, report
