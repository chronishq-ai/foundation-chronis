"""
divergence_engine/granger.py

Sprint 8, Day 23 — Condition 2: within-regime Granger predictability, and the
hard 20-session-per-regime power gate (MP-09).

Spec calls for Bayesian Markov-Switching VAR Granger causality (Droumaguet,
Warne & Wozniak 2017). statsmodels does not ship a Bayesian MS-VAR estimator
out of the box. This module implements the AIC-lag-selected VAR + Granger-F
test statsmodels DOES support, applied to (m_t, n_t) *within* a single fitted
regime window (i.e., already regime-conditioned by construction — same effect
Droumaguet et al. achieve via the switching prior, but via pre-segmentation
instead of joint estimation).

*** HONESTY FLAG — do not silently treat this as equivalent to the Bible spec ***
If your team needs the actual Bayesian MS-VAR (joint regime + VAR estimation,
not pre-segmented), swap `_fit_var_and_test` below for a real implementation
(e.g. via `statsmodels.tsa.regime_switching` composed with a VAR likelihood,
or an external Bayesian MS-VAR package). The gate logic, Bonferroni correction,
and MP-09 enforcement below are correct regardless of which estimator backs it.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from statsmodels.tsa.api import VAR

MIN_SESSIONS_PER_REGIME = 20  # MP-09: hard gate, no exceptions, no overrides.


@dataclass(frozen=True)
class GrangerResult:
    ran: bool  # False if the power gate blocked the test entirely
    f_statistic_m_causes_n: Optional[float]
    p_value_m_causes_n: Optional[float]
    f_statistic_n_causes_m: Optional[float]
    p_value_n_causes_m: Optional[float]
    lag_order: Optional[int]
    bonferroni_alpha: Optional[float]
    significant_m_causes_n: bool
    significant_n_causes_m: bool
    n_sessions_in_regime: int
    power_gate_passed: bool


def power_gate(n_sessions_in_regime: int) -> bool:
    """MP-09. Hard boolean. No soft score, ever."""
    return n_sessions_in_regime >= MIN_SESSIONS_PER_REGIME


def within_regime_granger_test(
    m_t: np.ndarray,
    n_t: np.ndarray,
    n_pairs_tested: int,
    max_lag: int = 5,
    alpha: float = 0.05,
) -> GrangerResult:
    """
    m_t: behavioral fast state, shape (T_regime, d1) — already sliced to sessions
         within the regime window being tested.
    n_t: narrative fast state, shape (T_regime, d2) — same session alignment.

    Enforces MP-09 FIRST: if n_sessions_in_regime < 20, the test does not run
    at all and this function returns immediately with ran=False.
    """
    if len(m_t) != len(n_t):
        raise ValueError("m_t and n_t must be aligned to the same session grid")

    n_sessions = len(m_t)
    gate_passed = power_gate(n_sessions)

    if not gate_passed:
        return GrangerResult(
            ran=False,
            f_statistic_m_causes_n=None,
            p_value_m_causes_n=None,
            f_statistic_n_causes_m=None,
            p_value_n_causes_m=None,
            lag_order=None,
            bonferroni_alpha=None,
            significant_m_causes_n=False,
            significant_n_causes_m=False,
            n_sessions_in_regime=n_sessions,
            power_gate_passed=False,
        )

    # Reduce multivariate m_t/n_t to their first principal / mean dimension for
    # a simple bivariate Granger test if d>1. (Full multivariate MS-VAR Granger
    # is a real extension point — flagged above.)
    m_series = m_t.mean(axis=1) if m_t.ndim > 1 else m_t
    n_series = n_t.mean(axis=1) if n_t.ndim > 1 else n_t

    data = np.column_stack([m_series, n_series])
    var_model = VAR(data)

    max_lag_allowed = min(max_lag, n_sessions // 3 - 1)
    max_lag_allowed = max(1, max_lag_allowed)
    aic_order = var_model.select_order(max_lag_allowed).aic
    lag_order = max(1, aic_order)

    fitted = var_model.fit(lag_order)

    gc_m_to_n = fitted.test_causality(caused=1, causing=0, kind="f")
    gc_n_to_m = fitted.test_causality(caused=0, causing=1, kind="f")

    bonferroni_alpha = alpha / max(1, n_pairs_tested)

    return GrangerResult(
        ran=True,
        f_statistic_m_causes_n=float(gc_m_to_n.test_statistic),
        p_value_m_causes_n=float(gc_m_to_n.pvalue),
        f_statistic_n_causes_m=float(gc_n_to_m.test_statistic),
        p_value_n_causes_m=float(gc_n_to_m.pvalue),
        lag_order=int(lag_order),
        bonferroni_alpha=bonferroni_alpha,
        significant_m_causes_n=bool(gc_m_to_n.pvalue < bonferroni_alpha),
        significant_n_causes_m=bool(gc_n_to_m.pvalue < bonferroni_alpha),
        n_sessions_in_regime=n_sessions,
        power_gate_passed=True,
    )
