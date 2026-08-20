import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from backbone.attractors.detector import compute_attractor_stats, is_attractor


def _hand_built_attractor_trajectory(seed=0):
    rng = np.random.default_rng(seed)
    T = 200
    modal = np.array([2.0, -1.0, 0.5])
    m_t = np.zeros((T, 3))
    regime_labels = np.zeros(T, dtype=int)
    t = 0
    while t < T:
        if rng.random() < 0.6:
            dwell = rng.integers(5, 15)
            m_t[t:t+dwell] = modal + rng.normal(scale=0.15, size=(min(dwell, T - t), 3))
        else:
            dwell = rng.integers(3, 8)
            m_t[t:t+dwell] = rng.normal(scale=3.0, size=(min(dwell, T - t), 3))
        t += dwell
    return m_t, regime_labels


def test_attractor_declared_on_obvious_structure():
    m_t, regime_labels = _hand_built_attractor_trajectory()
    stats = compute_attractor_stats(m_t, regime_labels, target_regime=0)
    assert stats["revisit_count"] > 0
    assert is_attractor(stats, N=(3, 3), T=1.0)


def test_insufficient_data_never_declares_attractor():
    m_t = np.zeros((2, 3))
    regime_labels = np.zeros(2, dtype=int)
    stats = compute_attractor_stats(m_t, regime_labels, target_regime=0)
    assert stats["insufficient_data"] is True
    assert is_attractor(stats, N=(0, 0), T=1e9) is False  # even absurdly loose thresholds must not admit it


def test_hard_and_rule_not_weighted():
    # construct stats that pass 2 of 3 conditions but fail the third -> must be False
    stats = {"revisit_count": 100, "mean_dwell_time": 100, "transition_stability": 5.0, "insufficient_data": False}
    assert is_attractor(stats, N=(3, 3), T=1.0) is False  # stability fails (5.0 not < 1.0)
    stats2 = {"revisit_count": 1, "mean_dwell_time": 100, "transition_stability": 0.01, "insufficient_data": False}
    assert is_attractor(stats2, N=(3, 3), T=1.0) is False  # revisit_count fails


def test_scalar_N_broadcasts_to_both():
    stats = {"revisit_count": 5, "mean_dwell_time": 5, "transition_stability": 0.5, "insufficient_data": False}
    assert is_attractor(stats, N=4, T=1.0) is True
    assert is_attractor(stats, N=6, T=1.0) is False


def test_neighborhood_radius_uses_norm_not_mean():
    # regression test for the found-and-fixed bug: radius must scale with
    # sqrt(F)-ish norm of per-dim std, not the raw mean, or revisit_count
    # collapses to 0 in higher dimensions.
    rng = np.random.default_rng(1)
    F = 8
    T = 150
    modal = rng.normal(size=F)
    m_t = np.zeros((T, F))
    regime_labels = np.zeros(T, dtype=int)
    t = 0
    while t < T:
        dwell = rng.integers(8, 20)
        m_t[t:t+dwell] = modal + rng.normal(scale=0.2, size=(min(dwell, T - t), F))
        t += dwell
    stats = compute_attractor_stats(m_t, regime_labels, target_regime=0)
    assert stats["revisit_count"] > 0, "regression: neighborhood radius bug reintroduced (mean(std) undershoot)"
