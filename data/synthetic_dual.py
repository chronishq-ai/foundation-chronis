"""
Synthetic Dual-Timescale Data Generator.

Generates ground-truth behavioral sequences with:
  - Slow discrete regimes p_t in {0, ..., K-1} with explicit durations
  - Fast continuous latent state m_t in R^L with regime-dependent AR(1) dynamics
  - Observed multi-feature emissions y_t = C_k m_t + d_k + v_t
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import lognorm


@dataclass
class SyntheticDualData:
    """Container for ground truth synthetic dual-timescale data."""
    X: np.ndarray              # shape (T, F) observed feature matrix
    true_p_t: np.ndarray       # shape (T,) true discrete regime sequence
    true_m_t: np.ndarray       # shape (T, L) true fast continuous latent states
    timestamps: np.ndarray     # shape (T,) calendar timestamps
    durations: list[int]       # segment lengths


def generate_synthetic_dual_timescale_data(
    n_regimes: int = 3,
    n_features: int = 4,
    latent_dim: int = 2,
    n_segments: int = 6,
    mean_dwell: float = 15.0,
    seed: int = 42,
) -> SyntheticDualData:
    """Generate synthetic ground-truth data with fast continuous state and slow regimes."""
    rng = np.random.default_rng(seed)
    K, F, L = n_regimes, n_features, latent_dim

    # 1. Define regime parameters
    # Regimes have distinct observation intercepts d_k and dynamics A_k
    A_fast = np.zeros((K, L, L))
    b_fast = np.zeros((K, L))
    Q_fast = np.zeros((K, L, L))
    C_obs = np.zeros((K, F, L))
    d_obs = np.zeros((K, F))
    R_obs = np.zeros((K, F))

    for k in range(K):
        # Distinct continuous dynamics per regime
        A_fast[k] = 0.75 * np.eye(L)
        b_fast[k] = np.zeros(L)
        Q_fast[k] = 0.3 * np.eye(L)

        # Loading matrix: maps latent dim 0 to features 0,2; latent dim 1 to features 1,3
        C_obs[k] = np.zeros((F, L))
        for f in range(F):
            C_obs[k, f, f % L] = 1.5
        # Distinct macro-phase baseline for each regime
        d_obs[k] = np.full(F, (k - (K - 1) / 2.0) * 4.0)
        R_obs[k] = np.full(F, 0.05)

    # 2. Sample regime sequence with explicit durations
    regime_sequence: list[int] = []
    durations: list[int] = []
    current_regime = 0

    for seg in range(n_segments):
        # Sample duration from lognormal
        dur = int(np.clip(rng.lognormal(mean=np.log(mean_dwell), sigma=0.3), 8, 30))
        regime_sequence.extend([current_regime] * dur)
        durations.append(dur)
        # Transition to a different regime
        possible = [k for k in range(K) if k != current_regime]
        current_regime = rng.choice(possible)

    T = len(regime_sequence)
    true_p_t = np.array(regime_sequence, dtype=int)

    # 3. Sample continuous state trajectory m_t
    true_m_t = np.zeros((T, L))
    m_curr = rng.normal(scale=0.2, size=L)
    true_m_t[0] = m_curr

    for t in range(1, T):
        k = true_p_t[t]
        w_t = rng.normal(scale=np.sqrt(np.diag(Q_fast[k])), size=L)
        m_curr = A_fast[k] @ m_curr + b_fast[k] + w_t
        true_m_t[t] = m_curr

    # 4. Sample observed emissions y_t
    X = np.zeros((T, F))
    for t in range(T):
        k = true_p_t[t]
        v_t = rng.normal(scale=np.sqrt(R_obs[k]), size=F)
        X[t] = C_obs[k] @ true_m_t[t] + d_obs[k] + v_t

    timestamps = np.linspace(1.0, float(T), T)

    return SyntheticDualData(
        X=X,
        true_p_t=true_p_t,
        true_m_t=true_m_t,
        timestamps=timestamps,
        durations=durations,
    )
