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


def test_timestamps_forwarded_to_condition3_sparse_user():
    """S56.1 gate-wiring fix: a sparse-session user (1 sample every 3
    calendar days) should still evaluate condition 3 over a real
    stability_min_days-CALENDAR-DAY window when timestamps are
    supplied, not a sample-count window. Without timestamps, the same
    signal falls back to sample-offset behavior (backward compatible),
    which for a sparse user pulls stability data from a much shorter
    calendar span. This proves is_stabilizing actually receives the
    gate's timestamps rather than them being silently dropped."""
    np.random.seed(7)
    pre = np.random.normal(0, 1, 100)
    post = np.array([np.random.normal(5, max(0.3, 3 - 0.15 * i)) for i in range(80)])
    signal = np.concatenate([pre, post]).tolist()

    # 1 sample every 3 calendar days -> stability_min_days=14 calendar
    # days spans ~5 samples without timestamps-awareness, not enough
    # for the >=10-sample minimum condition 3 requires.
    timestamps = [float(i * 3) for i in range(len(signal))]

    gate = PhaseTransitionGate(stability_min_days=14)

    result_no_ts = gate.evaluate_candidate(signal, 100)
    assert result_no_ts["condition_3_detail"]["valid"] is True, \
        "sanity check: sample-offset path should still be valid here"

    result_with_ts = gate.evaluate_candidate(signal, 100, timestamps=timestamps)
    # With timestamps threaded through, condition 3's window is a real
    # 14-calendar-day span (~5 samples at this cadence) -> insufficient
    # data for the >=10-sample floor -> explicitly invalid, not silently
    # evaluated over the wrong (sample-offset) window.
    assert result_with_ts["condition_3_detail"]["valid"] is False
    assert result_with_ts["condition_3_detail"]["reason"] == \
        "insufficient post-candidate data in calendar window"


def test_declared_transitions_accepts_timestamps_end_to_end():
    """detect_transitions/declared_transitions also forward timestamps
    all the way through, not just evaluate_candidate."""
    np.random.seed(42)
    pre = np.random.normal(0, 1, 100)
    post = np.array([np.random.normal(5, max(0.3, 3 - 0.15 * i)) for i in range(80)])
    signal = np.concatenate([pre, post]).tolist()
    timestamps = [float(i) for i in range(len(signal))]  # 1/day, dense

    gate = PhaseTransitionGate()
    declared_no_ts = gate.declared_transitions(signal)
    declared_with_ts = gate.declared_transitions(signal, timestamps=timestamps)

    # Dense (1/day) user: calendar-day window == sample-offset window,
    # so results should match -- proves the plumbing doesn't change
    # behavior for the common dense case, only for sparse ones.
    assert declared_no_ts == declared_with_ts