import numpy as np
from phase_transition.stability import RegimeStability


def test_stabilizing_regime_is_met():
    np.random.seed(7)
    pre = np.random.normal(0, 1, 50)
    # sigma drops from 3 -> 0.3 over 20 days: clear stabilizing trend
    post = np.array([np.random.normal(5, max(0.3, 3 - 0.15 * i)) for i in range(20)])
    signal = np.concatenate([pre, post]).tolist()

    stab = RegimeStability(min_days=20)
    result = stab.is_stabilizing(signal, candidate_t=50)

    assert result["valid"]
    assert result["met"], f"expected stabilizing regime to be met: {result}"


def test_transient_noise_is_not_met():
    np.random.seed(8)
    pre = np.random.normal(0, 1, 50)
    post = np.random.normal(5, 3, 20)  # constant variance, no trend
    signal = np.concatenate([pre, post]).tolist()

    stab = RegimeStability(min_days=20)
    result = stab.is_stabilizing(signal, candidate_t=50)

    assert result["valid"]
    assert not result["met"], f"expected transient noise to reset: {result}"
    assert result["reset"]


def test_insufficient_data_is_invalid():
    signal = list(np.random.normal(0, 1, 55))  # only 5 days post-candidate
    stab = RegimeStability(min_days=20)
    result = stab.is_stabilizing(signal, candidate_t=50)
    assert not result["valid"]