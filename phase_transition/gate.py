from phase_transition.bocpd import ChangepointDetector
from phase_transition.degradation import PredictiveFitDegradation
from phase_transition.hssm_degradation import evaluate_regime_conditional_degradation
from phase_transition.stability import RegimeStability, AmbiguousStabilityMetricError


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
                 bifurcation_evidence_window: float = 5.0,
                 require_regime_probabilities: bool = False):
        """
        require_regime_probabilities (item 3 mechanical sub-fix, added
        this pass): when True, `evaluate_candidate`/`detect_transitions`/
        `declared_transitions` raise `AmbiguousStabilityMetricError`
        instead of silently falling back to the raw-variance
        `RegimeStability.is_stabilizing` whenever a call omits
        `regime_probabilities`. Default False preserves the exact
        existing fallback behavior unchanged.

        This flag does NOT make the doctrine-correct entropy metric the
        only production path -- it only lets a caller who has already
        decided to require it say so and fail loudly instead of
        silently reusing old semantics. Making entropy the *default* /
        *only* path for every caller is the metric-swap decision
        `stability.py`'s HONESTY FLAG marks Senior-owned, with an
        explicit "DO NOT MERGE without that review" -- not done here,
        and not something this flag does either way.
        """
        self.bocpd = ChangepointDetector(hazard=hazard)
        self.degradation = PredictiveFitDegradation()
        self.stability = RegimeStability(min_days=stability_min_days)
        self.cp_window = cp_window
        self.cp_threshold = cp_threshold
        self.degradation_threshold = degradation_threshold
        self.bifurcation_log = bifurcation_log
        self.bifurcation_evidence_window = bifurcation_evidence_window
        self.require_regime_probabilities = require_regime_probabilities

    def evaluate_candidate(self, data: list[float], candidate_t: int,
                             timestamps: list[float] | None = None,
                             regime_probabilities: list | None = None,
                             regime_sequence: list | None = None,
                             hssm_observations: list | None = None) -> dict:
        """
        Evaluate one candidate changepoint against conditions 2 and 3.
        (Condition 1 -- being a candidate at all -- is assumed already
        true, since candidate_t normally comes from
        ChangepointDetector.candidate_changepoints().)

        timestamps: optional, same length as `data`. Forwarded to
        condition 3 for calendar-day windowing (mechanical fix,
        Yes/Yes-owned). Omit for unchanged sample-offset behavior.

        regime_probabilities (S56.1 metric swap -- SENIOR SIGN-OFF NOT
        YET OBTAINED, see stability.py docstring): optional, shape
        (T, K), the per-timestep regime-posterior vector from
        backbone.hssm's HSSMResult. When supplied, condition 3 uses
        RegimeStability.is_stabilizing_entropy (doctrine-correct
        posterior-entropy metric) INSTEAD OF the raw-variance
        is_stabilizing. When omitted (default), falls back to
        raw-variance is_stabilizing, unchanged. This lets a caller who
        actually has real HSSM posteriors opt into the doctrine metric
        without forcing it on callers who only have raw scalar data --
        but per the pack's Ownership Model this metric choice itself
        still needs Mandatory senior review before being made the
        default / used to gate a real release decision.

        regime_sequence, hssm_observations (S56.2 metric swap -- SAME
        SENIOR-SIGN-OFF CAVEAT, see phase_transition/hssm_degradation.py
        for the full honesty flag): optional, `regime_sequence` shape
        (T,) and `hssm_observations` shape (T, F), the decoded regime
        labels + observation features from HSSMAdapterOutput. When
        BOTH are supplied, condition 2 uses
        `is_regime_conditional_degraded` (regime-conditional Gaussian
        emission model INSTEAD OF the generic scalar-Gaussian
        `PredictiveFitDegradation`. When either is omitted (default),
        falls back to the generic scalar model, unchanged. As with
        regime_probabilities above, this is narrower than true HSSM
        predictive likelihood (see hssm_degradation.py docstring) and
        opting in does not constitute the Mandatory senior review the
        pack requires before this becomes the production default.
        """
        if regime_sequence is not None and hssm_observations is not None:
            degradation_detail = evaluate_regime_conditional_degradation(
                regime_sequence, hssm_observations, candidate_t,
                timestamps=timestamps)
            degradation_met = (
                degradation_detail.get("valid", False)
                and degradation_detail["degradation"] > self.degradation_threshold
            )
        else:
            degradation_detail = self.degradation.degradation_score(
                data, candidate_t)
            degradation_met = (
                degradation_detail.get("valid", False)
                and degradation_detail["degradation"] > self.degradation_threshold
            )

        rupture_evidence = False
        if self.bifurcation_log is not None:
            rupture_evidence = self.bifurcation_log.as_condition2_evidence(
                candidate_t, window=self.bifurcation_evidence_window)

        cond2 = degradation_met or rupture_evidence

        if regime_probabilities is not None:
            cond3_result = self.stability.is_stabilizing_entropy(
                regime_probabilities, candidate_t, timestamps=timestamps)
        elif self.require_regime_probabilities:
            raise AmbiguousStabilityMetricError(
                "PhaseTransitionGate(require_regime_probabilities=True) "
                "but evaluate_candidate was called without "
                "regime_probabilities -- refusing to silently fall back "
                "to the raw-variance stability metric. Pass "
                "regime_probabilities, or construct the gate with "
                "require_regime_probabilities=False (the default) to "
                "allow the raw-variance fallback."
            )
        else:
            cond3_result = self.stability.is_stabilizing(
                data, candidate_t, timestamps=timestamps)
        cond3 = cond3_result["met"] if cond3_result["valid"] else False

        declared = cond2 and cond3

        return {
            "candidate_t": candidate_t,
            "condition_1_candidate": True,
            "condition_2_degradation": cond2,
            "condition_2_degradation_score": degradation_met,
            "condition_2_rupture_evidence": rupture_evidence,
            # item 4 / R2-S56.4 mechanical sub-fix (this pass): decision
            # record always names which model produced condition 2 --
            # never an anonymous Gaussian -- plus the exact windows and
            # (when the regime-conditional path ran) the null-baseline
            # comparison. Real fitted-HSSM predictive likelihood as the
            # PRODUCTION DEFAULT still requires backbone.hssm (item 1,
            # not in this zip) -- this only makes whichever model DID
            # run traceable and reproducible, it does not swap the
            # default per S56.4's full ask.
            "condition_2_detail": degradation_detail,
            "condition_3_stability": cond3,
            "condition_3_detail": cond3_result,
            "declared_transition": declared,
        }

    def detect_transitions(self, data: list[float],
                             timestamps: list[float] | None = None,
                             regime_probabilities: list | None = None,
                             regime_sequence: list | None = None,
                             hssm_observations: list | None = None) -> list[dict]:
        """
        Full pipeline: find condition-1 candidates via BOCPD, then test
        each against conditions 2 and 3. Only candidates passing all 3
        are declared transitions.

        timestamps, regime_probabilities, regime_sequence,
        hssm_observations: forwarded to evaluate_candidate (see its
        docstring for the S56.1 entropy-metric and S56.2
        regime-conditional-degradation opt-ins).
        """
        candidates = self.bocpd.candidate_changepoints(
            data, window=self.cp_window, threshold=self.cp_threshold)

        results = []
        for t in candidates:
            result = self.evaluate_candidate(
                data, t, timestamps=timestamps,
                regime_probabilities=regime_probabilities,
                regime_sequence=regime_sequence,
                hssm_observations=hssm_observations)
            results.append(result)
        return results

    def declared_transitions(self, data: list[float],
                               timestamps: list[float] | None = None,
                               regime_probabilities: list | None = None,
                               regime_sequence: list | None = None,
                               hssm_observations: list | None = None) -> list[int]:
        """Just the timesteps where all 3 conditions passed."""
        results = self.detect_transitions(
            data, timestamps=timestamps, regime_probabilities=regime_probabilities,
            regime_sequence=regime_sequence, hssm_observations=hssm_observations)
        return [r["candidate_t"] for r in results if r["declared_transition"]]