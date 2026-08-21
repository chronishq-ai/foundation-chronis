from phase_transition.bocpd import ChangepointDetector
from phase_transition.degradation import PredictiveFitDegradation
from phase_transition.stability import RegimeStability


class PhaseTransitionGate:
    """
    Composes all 3 conditions into a single hard gate.
    A phase transition is declared ONLY if all 3 conditions pass.
    BOCPD change-point probability alone can NEVER declare a transition
    on its own -- it only ever contributes condition 1 of 3.

    Condition 2 (predictive fit degradation) can ALSO be satisfied by a
    nearby declared rupture in the BifurcationLog (Module 4.11) -- this
    is the "feeds phase-transition condition 2 as additional evidence"
    rule from Bible Part 5.23 / Sprint 5 Day 15. Rupture evidence is
    ORed with the degradation-score check, never a replacement for it.
    """

    def __init__(self,
                 hazard: float = 1/250,
                 cp_window: int = 3,
                 cp_threshold: float = 0.5,
                 degradation_threshold: float = 2.0,
                 stability_min_days: int = 14,
                 bifurcation_log=None,
                 bifurcation_evidence_window: float = 5.0):
        self.bocpd = ChangepointDetector(hazard=hazard)
        self.degradation = PredictiveFitDegradation()
        self.stability = RegimeStability(min_days=stability_min_days)
        self.cp_window = cp_window
        self.cp_threshold = cp_threshold
        self.degradation_threshold = degradation_threshold
        self.bifurcation_log = bifurcation_log
        self.bifurcation_evidence_window = bifurcation_evidence_window

    def evaluate_candidate(self, data: list[float], candidate_t: int) -> dict:
        """
        Evaluate one candidate changepoint against conditions 2 and 3.
        (Condition 1 -- being a candidate at all -- is assumed already
        true, since candidate_t normally comes from
        ChangepointDetector.candidate_changepoints().)
        """
        degradation_met = self.degradation.is_degraded(
            data, candidate_t, threshold=self.degradation_threshold)

        rupture_evidence = False
        if self.bifurcation_log is not None:
            rupture_evidence = self.bifurcation_log.as_condition2_evidence(
                candidate_t, window=self.bifurcation_evidence_window)

        cond2 = degradation_met or rupture_evidence

        cond3_result = self.stability.is_stabilizing(data, candidate_t)
        cond3 = cond3_result["met"] if cond3_result["valid"] else False

        declared = cond2 and cond3

        return {
            "candidate_t": candidate_t,
            "condition_1_candidate": True,
            "condition_2_degradation": cond2,
            "condition_2_degradation_score": degradation_met,
            "condition_2_rupture_evidence": rupture_evidence,
            "condition_3_stability": cond3,
            "condition_3_detail": cond3_result,
            "declared_transition": declared,
        }

    def detect_transitions(self, data: list[float]) -> list[dict]:
        """
        Full pipeline: find condition-1 candidates via BOCPD, then test
        each against conditions 2 and 3. Only candidates passing all 3
        are declared transitions.
        """
        candidates = self.bocpd.candidate_changepoints(
            data, window=self.cp_window, threshold=self.cp_threshold)

        results = []
        for t in candidates:
            result = self.evaluate_candidate(data, t)
            results.append(result)
        return results

    def declared_transitions(self, data: list[float]) -> list[int]:
        """Just the timesteps where all 3 conditions passed."""
        results = self.detect_transitions(data)
        return [r["candidate_t"] for r in results if r["declared_transition"]]