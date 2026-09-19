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


class ColdStartError(Exception):
    """Raised when HSSM output is requested below the session gate. This is an
    intentional, caught exception — never a silent low-confidence fallback."""
    pass


def count_present_sessions(X: np.ndarray) -> int:
    """Count eligible sessions using the confidence contract tiers (>= 50% feature presence).
    Rows with >= 50% observed features (tiers 1.0, 0.75, 0.5) are eligible; tier < 0.5 is excluded."""
    if X.size == 0 or X.ndim < 2 or X.shape[1] == 0:
        return 0
    obs_fraction = (~np.isnan(X)).sum(axis=1) / X.shape[1]
    return int((obs_fraction >= 0.5).sum())


def fit_hssm_gated(
    X: np.ndarray,
    n_regimes: int = 3,
    n_features: int = 1,
    n_present_sessions: int | None = None,
    config: ColdStartConfig = DEFAULT_COLD_START_CONFIG,
    n_init: int = 10,
    max_duration: int = 45,
    base_seed: int = 0,
    timestamps: np.ndarray | None = None,
    min_present_sessions: int | None = None,
):
    """Wraps canonical fit_hssm for backward compatibility.
    ColdStartError is raised by canonical fit_hssm below config.min_present_sessions."""
    if n_features <= 0:
        raise ValueError("Feature count must be strictly greater than zero (F > 0)")

    if min_present_sessions is not None:
        config = ColdStartConfig(min_present_sessions=min_present_sessions)

    from backbone.hssm.fitting import fit_hssm

    res = fit_hssm(
        matrix=X,
        candidate_ks=(n_regimes,),
        n_initializations=n_init,
        random_seed=base_seed,
        allow_fast_test_fit=(n_init < 10),
        timestamps=timestamps,
        config=config,
        n_present_sessions=n_present_sessions,
        max_duration=max_duration,
    )
    return res.model, res.convergence_metadata.get("run_log", [])

