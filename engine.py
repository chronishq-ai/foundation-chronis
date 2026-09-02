from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

import numpy as np

from .state import DivergenceState

AMBIGUITY_GAP = 0.15


@dataclass
class DivergenceInputs:
    user_id: str
    domain_id: str
    window_start: datetime
    window_end: datetime
    p_t: np.ndarray
    q_t: np.ndarray
    m_t: np.ndarray
    n_t: np.ndarray
    behavioral_regime_id: int
    narrative_regime_id: int
    n_domain_pairs_tested: int
    behavioral_attractor_weakening: bool
    narrative_conformal_confidence: float


def _recover_lag(b: np.ndarray, n: np.ndarray, max_lag: int = 12) -> tuple[int, float]:
    db = -np.diff(np.asarray(b, dtype=float))
    dn = np.diff(np.asarray(n, dtype=float))
    best_lag = 0
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = db[-lag:], dn[: len(dn) + lag]
        elif lag > 0:
            x, y = db[: len(db) - lag], dn[lag:]
        else:
            x, y = db, dn
        if len(x) < 10:
            continue
        if np.std(x) < 1e-6 or np.std(y) < 1e-6:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if np.isnan(c):
            continue
        if c > best:
            best, best_lag = c, lag
    return best_lag, best


def _score_types(
    m_t: np.ndarray,
    n_t: np.ndarray,
    *,
    behavioral_attractor_weakening: bool,
) -> dict[str, float]:
    """Type scores from fitted latents (Bible-shaped heuristics).

    NOTE: This path uses the available divergence package API. The Sprint 8
    Granger estimator in production remains OLS-VAR-limited (S79.1 open).
    Do not describe this as Bayesian MS-VAR validation.
    """
    b = np.asarray(m_t, dtype=float)
    n = np.asarray(n_t, dtype=float)
    if len(b) == 0 or len(n) == 0:
        return {
            "Ignorance": 0.0,
            "Aspiration": 0.0,
            "Self-Protection": 0.0,
            "ActiveTransition": 0.0,
        }
    dn = np.diff(n) if len(n) > 1 else np.array([0.0])
    b_mean = float(np.mean(b))
    b_drop = float(np.clip(b[0] - b[-1], 0, 1))
    b_var = float(np.std(b))
    n_mean = float(np.mean(n))
    n_rise = float(np.clip(n[-1] - n[0], 0, 1))
    lag, corr = _recover_lag(b, n)

    # Prefer the explicit weakening flag when callers set it from fit diagnostics.
    weaken = 1.0 if behavioral_attractor_weakening else b_drop

    ignorance = b_mean * (1.0 - n_mean) * (1.0 - n_rise)
    aspiration = weaken * n_mean * (1.0 - n_rise)
    self_prot = (1.0 - min(b_var * 6, 1.0)) * (1.0 - abs(n_mean - 0.35)) * (1.0 - n_rise)
    lag_ok = 1.0 if lag >= 2 else 0.15
    at = weaken * n_rise * max(corr, 0.0) * lag_ok

    raw = {
        "Ignorance": ignorance,
        "Aspiration": aspiration,
        "Self-Protection": self_prot,
        "ActiveTransition": at,
    }
    return {k: float(np.clip(v, 0, 1)) for k, v in raw.items()}


def compute_divergence_state(
    inputs: DivergenceInputs,
    previous_state_id: str | None = None,
) -> DivergenceState:
    """Sprint 8 API surface used by Sprint 14 gates + Sprint 15 harness."""
    scores = _score_types(
        inputs.m_t,
        inputs.n_t,
        behavioral_attractor_weakening=inputs.behavioral_attractor_weakening,
    )
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, second = ordered[0], ordered[1]
    ambiguous = (top[1] - second[1]) < AMBIGUITY_GAP
    dominant = None if ambiguous else top[0]
    return DivergenceState(
        user_id=inputs.user_id,
        domain_id=inputs.domain_id,
        state_id=uuid4().hex,
        previous_state_id=previous_state_id,
        type_scores=scores,
        dominant_type=dominant,
        ambiguous=ambiguous,
    )
