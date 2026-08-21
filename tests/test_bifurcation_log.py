import numpy as np
from phase_transition.gate import PhaseTransitionGate
from phase_transition.bifurcation_log import BifurcationLog
from phase_transition.rupture import RuptureDetector, SensorSnapshot


def _declared_rupture_result(timestamp):
    det = RuptureDetector()
    snap = SensorSnapshot(
        timestamp=timestamp,
        voice_energy_sigma=3.5,
        ppg_hr_pct_above_baseline=45.0,
        cse_salience_level=5,
        cse_salience_duration_min=12.0,
        imu_motion_disruption=True,
    )
    return det.is_rupture(snap)


def test_nearby_rupture_satisfies_cond2_even_when_degradation_weak():
    """A candidate with WEAK degradation score but a nearby declared
    rupture should still pass condition 2 via the OR."""
    np.random.seed(11)
    # flat signal, no real mean shift -> degradation score will be weak/False
    signal = np.random.normal(0, 1, 200).tolist()

    log = BifurcationLog()
    log.append(_declared_rupture_result(timestamp=100.0))

    gate = PhaseTransitionGate(bifurcation_log=log,
                                bifurcation_evidence_window=5.0)

    result = gate.evaluate_candidate(signal, candidate_t=100)

    assert not result["condition_2_degradation_score"], \
        "expected degradation score alone to be weak on flat noise"
    assert result["condition_2_rupture_evidence"], \
        "expected rupture evidence to be found near candidate_t=100"
    assert result["condition_2_degradation"], \
        "expected cond2 to pass via OR even though degradation score failed"


def test_distant_rupture_does_not_leak_into_condition2():
    """A rupture far away in time from the candidate must NOT satisfy
    condition 2 for this candidate."""
    np.random.seed(12)
    signal = np.random.normal(0, 1, 300).tolist()

    log = BifurcationLog()
    log.append(_declared_rupture_result(timestamp=10.0))  # far from t=200

    gate = PhaseTransitionGate(bifurcation_log=log,
                                bifurcation_evidence_window=5.0)

    result = gate.evaluate_candidate(signal, candidate_t=200)

    assert not result["condition_2_rupture_evidence"], \
        "distant rupture (t=10) should not count as evidence for t=200"
    assert not result["condition_2_degradation"], \
        "cond2 should fail: no real degradation AND no nearby rupture"


def test_gate_without_bifurcation_log_still_works_as_before():
    """Backward compatibility: gate with no bifurcation_log passed
    behaves exactly like the original degradation-only condition 2."""
    np.random.seed(42)
    pre = np.random.normal(0, 1, 100)
    post = np.array([np.random.normal(5, max(0.3, 3 - 0.15 * i)) for i in range(80)])
    signal = np.concatenate([pre, post]).tolist()

    gate = PhaseTransitionGate()  # no bifurcation_log
    declared = gate.declared_transitions(signal)

    assert len(declared) > 0, "expected genuine transition still declared without log"
    assert any(90 <= t <= 125 for t in declared)