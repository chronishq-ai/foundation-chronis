"""
MP-02 fix: canonical post-hoc regime alignment. HMMs/SSMs with multiple regimes
can label the same underlying regime differently across fitting runs, making
cross-session/cross-user comparison meaningless without correction.

Solution (locked convention, never to be changed after first real fit): sort
regimes by mean behavioral activity level, ascending. Regime 0 = lowest
activity, regime K-1 = highest.
"""

from __future__ import annotations
import numpy as np

ACTIVITY_LEVEL_CONVENTION = "mean_L2_norm_of_regime_mean_vector"


def canonicalize_labels(model) -> object:
    """Reorders pi, A, mu, var, dur_mu, dur_sigma in place by ascending
    activity level (L2 norm of each regime's mean vector in reduced feature
    space). Returns the same model object, mutated, plus a `_label_order_applied`
    attribute recording the permutation used (for audit)."""
    activity = np.linalg.norm(model.mu, axis=1)
    order = np.argsort(activity)

    model.pi = model.pi[order]
    model.A = model.A[np.ix_(order, order)]
    model.mu = model.mu[order]
    model.var = model.var[order]
    model.dur_mu = model.dur_mu[order]
    model.dur_sigma = model.dur_sigma[order]
    model._label_order_applied = order.tolist()
    return model


def canonicalize_regime_order(model) -> object:
    """Legacy alias matching foundation's old label switching interface."""
    return canonicalize_labels(model)


def activity_levels(model) -> np.ndarray:
    """Returns the activity-level metric per regime, in current label order.
    Exposed separately so tests/callers can assert ascending order without
    recomputing the convention inline."""
    return np.linalg.norm(model.mu, axis=1)
