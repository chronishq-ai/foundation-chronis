"""Synthetic surrogate-user trajectory generator (semi-Markov, log-normal dwell)."""

from __future__ import annotations
import numpy as np


def generate_surrogate_user(
    n_sessions: int = 120,
    n_features: int = 10,
    n_regimes: int = 3,
    mean_dwell_days: float = 25.0,
    dwell_lognormal_sigma: float = 0.5,
    missing_session_rate: float = 0.08,
    seed: int | None = None,
) -> dict:
    """Generate one synthetic user's raw feature matrix (T x F) with calendar-date
    indexed sessions, ground-truth regime sequence sampled from a log-normal-duration
    semi-Markov process (NOT geometric), and random missing sessions (NaN rows)."""
    rng = np.random.default_rng(seed)

    regime_means = rng.normal(loc=0.0, scale=1.0, size=(n_regimes, n_features))
    regime_means += rng.normal(loc=0.0, scale=2.5, size=(n_regimes, n_features))
    regime_cov_scale = rng.uniform(0.5, 1.2, size=n_regimes)

    calendar_day = 0
    regime_path = []
    current_regime = int(rng.integers(0, n_regimes))
    total_days_target = int(n_sessions * 1.15)

    trans = rng.dirichlet(alpha=np.ones(n_regimes - 1) * 2.0, size=n_regimes)

    while calendar_day < total_days_target:
        mu = np.log(mean_dwell_days) - 0.5 * dwell_lognormal_sigma**2
        duration = max(1, int(round(rng.lognormal(mean=mu, sigma=dwell_lognormal_sigma))))
        end_day = calendar_day + duration
        regime_path.append((calendar_day, end_day, current_regime))
        calendar_day = end_day

        others = [r for r in range(n_regimes) if r != current_regime]
        probs = trans[current_regime]
        current_regime = others[rng.choice(len(others), p=probs)]

    total_days = regime_path[-1][1]
    day_regime = np.zeros(total_days, dtype=int)
    for start, end, r in regime_path:
        day_regime[start:end] = r

    all_days = np.arange(total_days)
    session_days = np.sort(rng.choice(all_days, size=min(n_sessions, len(all_days)), replace=False))

    X = np.zeros((len(session_days), n_features))
    true_regimes = day_regime[session_days]
    for i, r in enumerate(true_regimes):
        cov = np.eye(n_features) * regime_cov_scale[r]
        X[i] = rng.multivariate_normal(regime_means[r], cov)

    missing_mask = rng.random(len(session_days)) < missing_session_rate
    X_with_missing = X.copy()
    X_with_missing[missing_mask] = np.nan

    return {
        "dates": session_days,
        "X": X_with_missing,
        "X_complete": X,
        "true_regimes": true_regimes,
        "regime_means": regime_means,
        "missing_mask": missing_mask,
    }
