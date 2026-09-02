from __future__ import annotations

import numpy as np


def build_synthetic_validation_set(sessions_per_pattern: int = 6):
    labels = np.tile(np.array([0, 1], dtype=int), sessions_per_pattern * 4)
    return [], labels
