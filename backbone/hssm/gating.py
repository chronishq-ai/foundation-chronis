"""
Sprint 3, Day 9 — Cold-start gate + missing-session marginalization.

Spec:
  - Below 30 present sessions: literally NO output. Not low-confidence output.
  - Missing sessions marginalized via NULL extension in model.py's emission
    log-likelihood (NaN rows contribute 0, never imputed) — this module is
    the GATE that decides whether fitting is even attempted, it does not
    itself do the marginalization (that lives in model.py's _emission_loglik,
    exercised here only through count-based gating logic).
"""

from __future__ import annotations
import numpy as np

from backbone.hssm.config import DEFAULT_COLD_START_CONFIG, ColdStartConfig
from backbone.hssm.fitting import fit_with_random_restarts


class ColdStartError(Exception):
    """Raised when HSSM output is requested below the session gate. This is an
    intentional, caught exception — never a silent low-confidence fallback."""
    pass


def count_present_sessions(X: np.ndarray) -> int:
    """Count sessions with NO missing feature (a row is 'present' only if
    every feature in it is non-NaN). Missingness cannot be used to pad the count."""
    return int((~np.isnan(X).any(axis=1)).sum())


def fit_hssm_gated(
    X: np.ndarray,
    n_regimes: int,
    n_features: int,
    n_present_sessions: int | None = None,
    config: ColdStartConfig = DEFAULT_COLD_START_CONFIG,
    n_init: int = 10,
    max_duration: int = 45,
    base_seed: int = 0,
):
    """Wraps fit_with_random_restarts with the cold-start gate. Raises
    ColdStartError below config.min_present_sessions; fits normally at/above it."""
    if n_present_sessions is None:
        n_present_sessions = count_present_sessions(X)

    if n_present_sessions < config.min_present_sessions:
        raise ColdStartError(
            f"{n_present_sessions} present sessions < cold-start minimum "
            f"({config.min_present_sessions}). No HSSM output produced. This "
            f"is the correct, silent, no-output state per Bible 5.1 doctrine, "
            f"not a low-confidence result."
        )

    return fit_with_random_restarts(
        X, n_regimes=n_regimes, n_features=n_features, n_init=n_init,
        max_duration=max_duration, base_seed=base_seed,
    )
