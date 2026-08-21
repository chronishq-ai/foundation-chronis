import numpy as np
from phase_transition.degradation import PredictiveFitDegradation


def test_degrades_on_real_mean_shift():
    np.random.seed(42)
    pre = np.random.normal(0, 1, 100)
    post = np.random.normal(5, 1, 100)
    signal = np.concatenate([pre, post]).tolist()

    deg = PredictiveFitDegradation()
    assert deg.is_degraded(signal, candidate_t=100, threshold=2.0)


def test_no_degradation_on_stationary_noise():
    np.random.seed(1)
    signal = np.random.normal(0, 1, 200).tolist()

    deg = PredictiveFitDegradation()
    assert not deg.is_degraded(signal, candidate_t=100, threshold=2.0)