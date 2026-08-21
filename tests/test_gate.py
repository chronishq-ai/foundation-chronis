import numpy as np
from phase_transition.gate import PhaseTransitionGate


def test_genuine_transition_declared():
    """All 3 conditions met -> should declare exactly around t=100."""
    np.random.seed(42)
    pre = np.random.normal(0, 1, 100)
    # sigma drops from 3 -> 0.3 over 20 days post-shift: clear stabilizing trend
    post = np.array([np.random.normal(5, max(0.3, 3 - 0.15 * i)) for i in range(80)])
    signal = np.concatenate([pre, post]).tolist()

    gate = PhaseTransitionGate()
    declared = gate.declared_transitions(signal)

    assert len(declared) > 0, "expected at least one declared transition"
    assert any(90 <= t <= 125 for t in declared), f"expected near t=100, got {declared}"

def test_transient_spike_not_declared():
    """A brief spike that reverts back -- condition 3 should block it."""
    np.random.seed(3)
    pre = np.random.normal(0, 1, 100)
    spike = np.random.normal(5, 1, 10)   # short-lived spike
    revert = np.random.normal(0, 1, 90)  # reverts to old regime
    signal = np.concatenate([pre, spike, revert]).tolist()

    gate = PhaseTransitionGate()
    declared = gate.declared_transitions(signal)

    # spike reverts, so stability condition should fail -> nothing declared near t=100-110
    assert not any(95 <= t <= 115 for t in declared), \
        f"transient spike should not be declared, got {declared}"


def test_stationary_noise_declares_nothing():
    np.random.seed(1)
    signal = np.random.normal(0, 1, 200).tolist()

    gate = PhaseTransitionGate()
    declared = gate.declared_transitions(signal)

    assert len(declared) == 0, f"stationary noise should declare nothing, got {declared}"