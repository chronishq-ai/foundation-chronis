"""
Day 16 — Synthetic HSSM-shaped data generator.

Standalone, no dependency on backbone.hssm. Matches the REAL HSSM output
interface/shape (per backbone.hssm.model.GaussianHSMM) so context-signature
extraction + HDBSCAN clustering can be built and tested now, and swapped to
real Team 2 output later by changing only the loader, not downstream logic.

Shape contract (matches real GaussianHSMM):
  - regime_sequence: int array, len T, values in [0, K-1]
  - transition_matrix (A): K x K, zero diagonal, rows sum to 1
  - emission_means (mu): K x F
  - emission_covariances: list of K diagonal F x F cov matrices
  - duration_mu, duration_sigma: K-length, log-normal params
  - session-index units, not calendar days
  - missing sessions -> NaN rows in observations (not imputed)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class SyntheticHSSMOutput:
    regime_sequence: np.ndarray       # (T,) int
    observations: np.ndarray          # (T, F) float, NaN = missing session
    transition_matrix: np.ndarray     # (K, K)
    emission_means: np.ndarray        # (K, F)
    emission_covariances: list        # list of (F, F) diag matrices
    duration_mu: np.ndarray           # (K,)
    duration_sigma: np.ndarray        # (K,)
    n_regimes: int
    n_features: int


def _random_transition_matrix(K: int, rng: np.random.Generator) -> np.ndarray:
    """Zero-diagonal, row-stochastic, matches real HSSM convention
    (dwell handled by duration dist, not self-loops)."""
    A = np.zeros((K, K))
    for k in range(K):
        others = [j for j in range(K) if j != k]
        if K > 1:
            raw = rng.dirichlet(np.ones(K - 1) * 2.0)
            for idx, j in enumerate(others):
                A[k, j] = raw[idx]
    return A


def generate_synthetic_hssm_output(
    T: int = 200,
    K: int = 3,
    F: int = 5,
    mean_dwell: float = 20.0,
    missing_rate: float = 0.05,
    seed: int | None = None,
) -> SyntheticHSSMOutput:
    """Generate one synthetic user's HSSM-shaped output: a regime sequence
    respecting a log-normal duration prior + transition matrix, plus
    Gaussian emissions per regime, plus randomly NaN'd missing sessions."""
    rng = np.random.default_rng(seed)

    A = _random_transition_matrix(K, rng)

    # duration params: log-normal, centered near mean_dwell (session-index units)
    mu_ln = np.log(mean_dwell) - 0.5 * 0.5**2
    duration_mu = np.full(K, mu_ln) + rng.normal(scale=0.3, size=K)
    duration_sigma = np.full(K, 0.5) + rng.uniform(0.0, 0.2, size=K)

    # emission params: K regime centers spread in F-dim space, ascending
    # activity level (L2 norm) to match canonicalize_labels convention
    base = rng.normal(scale=1.0, size=(K, F))
    activity_scale = np.linspace(1.0, 3.0, K)  # ascending
    emission_means = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-9) * activity_scale[:, None]
    emission_covariances = [np.diag(rng.uniform(0.3, 1.0, size=F)) for _ in range(K)]

    # generate regime sequence via duration prior + transitions
    d_vals = np.arange(1, int(mean_dwell * 4))
    regime_sequence = np.empty(T, dtype=int)
    t = 0
    current = rng.integers(0, K)
    while t < T:
        sigma = duration_sigma[current]
        mu = duration_mu[current]
        pdf = np.exp(-((np.log(d_vals) - mu) ** 2) / (2 * sigma**2)) / (d_vals * sigma * np.sqrt(2 * np.pi))
        pdf = pdf / pdf.sum()
        dur = rng.choice(d_vals, p=pdf)
        end = min(t + dur, T)
        regime_sequence[t:end] = current
        t = end
        if t >= T:
            break
        if A[current].sum() > 0:
            current = rng.choice(K, p=A[current] / A[current].sum())

    # generate observations from regime sequence
    observations = np.empty((T, F))
    for i in range(T):
        k = regime_sequence[i]
        cov_diag = np.diag(emission_covariances[k])
        observations[i] = rng.normal(loc=emission_means[k], scale=np.sqrt(cov_diag))

    # inject missingness (NaN rows) — session dropout, not imputed downstream
    missing_mask = rng.random(T) < missing_rate
    observations[missing_mask] = np.nan

    return SyntheticHSSMOutput(
        regime_sequence=regime_sequence,
        observations=observations,
        transition_matrix=A,
        emission_means=emission_means,
        emission_covariances=emission_covariances,
        duration_mu=duration_mu,
        duration_sigma=duration_sigma,
        n_regimes=K,
        n_features=F,
    )