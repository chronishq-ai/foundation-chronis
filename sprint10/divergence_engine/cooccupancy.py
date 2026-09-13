"""
divergence_engine/cooccupancy.py

Sprint 8, Day 23 — Condition 1: regime co-occupancy.

Generalizes to a windowed contingency table between p_t (behavioral regime)
and q_t (Sprint 7's NSSM narrative regime) via the identical Fisher's-exact,
Bonferroni-corrected procedure already built in Sprint 6 Day 17.

This module does not reimplement Sprint 6's alignment code — in the real repo
this would `import` it directly. It's reproduced here (structurally identical)
so Sprint 8 is runnable standalone.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CooccupancyResult:
    contingency_table: np.ndarray  # shape (n_behavioral_regimes_in_window, n_narrative_regimes_in_window)
    p_value: float
    bonferroni_alpha: float
    significant: bool
    n_sessions: int


def windowed_contingency_table(
    p_t: np.ndarray,
    q_t: np.ndarray,
    behavioral_regime_id: int,
    narrative_regime_id: int,
) -> np.ndarray:
    """
    Build a 2x2 contingency table for "session is in behavioral_regime_id" x
    "session is in narrative_regime_id", over the aligned (same-length,
    same-timestamp-grid) p_t / q_t arrays.
    """
    if len(p_t) != len(q_t):
        raise ValueError("p_t and q_t must be aligned to the same session/time grid")

    in_behavioral = (p_t == behavioral_regime_id)
    in_narrative = (q_t == narrative_regime_id)

    a = int(np.sum(in_behavioral & in_narrative))
    b = int(np.sum(in_behavioral & ~in_narrative))
    c = int(np.sum(~in_behavioral & in_narrative))
    d = int(np.sum(~in_behavioral & ~in_narrative))
    return np.array([[a, b], [c, d]])


def fisher_cooccupancy_test(
    p_t: np.ndarray,
    q_t: np.ndarray,
    behavioral_regime_id: int,
    narrative_regime_id: int,
    n_pairs_tested: int,
    alpha: float = 0.05,
) -> CooccupancyResult:
    """
    Fisher's exact test on regime co-occupancy, Bonferroni-corrected for the
    number of behavioral x narrative regime pairs tested for this user.

    `n_pairs_tested` must be the count of ALL pairs being tested in this run,
    not just this one — Bonferroni correction is only valid computed that way.
    """
    if n_pairs_tested < 1:
        raise ValueError("n_pairs_tested must be >= 1")

    table = windowed_contingency_table(p_t, q_t, behavioral_regime_id, narrative_regime_id)
    _, p_value = stats.fisher_exact(table)

    bonferroni_alpha = alpha / n_pairs_tested
    significant = p_value < bonferroni_alpha

    return CooccupancyResult(
        contingency_table=table,
        p_value=float(p_value),
        bonferroni_alpha=bonferroni_alpha,
        significant=significant,
        n_sessions=len(p_t),
    )
