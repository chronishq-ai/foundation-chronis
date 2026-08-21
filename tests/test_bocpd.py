import numpy as np
from phase_transition.bocpd import ChangepointDetector


def test_detects_candidate_on_synthetic_shift():
    np.random.seed(42)
    pre = np.random.normal(0, 1, 100)
    post = np.random.normal(5, 1, 100)
    signal = np.concatenate([pre, post]).tolist()

    det = ChangepointDetector(hazard=1/250)
    cps = det.candidate_changepoints(signal, window=3, threshold=0.5)

    assert any(90 <= t <= 115 for t in cps), f"expected candidate near t=100, got {cps}"


def test_no_candidate_on_stationary_noise():
    np.random.seed(1)
    signal = np.random.normal(0, 1, 200).tolist()
    det = ChangepointDetector(hazard=1/250)
    cps = det.candidate_changepoints(signal, window=3, threshold=0.5)
    assert len(cps) < 10, f"stationary noise shouldn't spam candidates, got {cps}"