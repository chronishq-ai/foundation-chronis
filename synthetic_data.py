from __future__ import annotations

import numpy as np


def generate_surrogate_user(n_sessions=40, n_features=6, n_regimes=2, seed=None):
    rng = np.random.default_rng(seed)
    t = max(n_sessions, 8) * 4
    x = rng.normal(size=(t, n_features))
    regimes = rng.integers(0, n_regimes, size=t)
    return {"X_complete": x, "true_regimes": regimes}
