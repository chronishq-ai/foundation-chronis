# SCRATCH / EXPLORATION ONLY — NEVER CITE AS VALIDATION EVIDENCE.
#
# These formulas were the pre-audit disconnected toy simulation (closure
# ticket S15.1 / §14.1). Kept for historical reference and unit-level
# debugging. Production validation must use
# divergence_engine.engine.compute_divergence_state → DivergenceState.type_scores.
from __future__ import annotations

import numpy as np


def recover_lag(b: np.ndarray, n: np.ndarray, max_lag: int = 12) -> tuple[int, float]:
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


def type_scores(b: np.ndarray, n: np.ndarray) -> dict[str, float]:
    """Hand-invented scratch formula — not the production divergence path."""
    b = np.asarray(b, dtype=float)
    n = np.asarray(n, dtype=float)
    dn = np.diff(n)
    b_mean = float(np.mean(b))
    b_drop = float(np.clip(b[0] - b[-1], 0, 1))
    b_var = float(np.std(b))
    n_mean = float(np.mean(n))
    n_rise = float(np.clip(n[-1] - n[0], 0, 1))
    lag, corr = recover_lag(b, n)

    ignorance = b_mean * (1.0 - n_mean) * (1.0 - n_rise)
    aspiration = b_drop * n_mean * (1.0 - n_rise)
    self_prot = (1.0 - min(b_var * 6, 1.0)) * (1.0 - abs(n_mean - 0.35)) * (1.0 - n_rise)
    lag_ok = 1.0 if lag >= 2 else 0.15
    at = b_drop * n_rise * max(corr, 0.0) * lag_ok

    raw = {
        "Ignorance": ignorance,
        "Aspiration": aspiration,
        "Self-Protection": self_prot,
        "ActiveTransition": at,
    }
    return {k: float(np.clip(v, 0, 1)) for k, v in raw.items()}
