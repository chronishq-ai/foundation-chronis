"""
cold_start.py
Sprint 10 — Threshold Calibration II & Cold Start Compass

Implements the 5-stage Cold Start Protocol as an explicit state machine,
and the person-specific D*-based divergence observation window.

Bible refs: Part 5.10 (Cold Start Protocol), Part 5.11 (Threshold Calibration)
Closes: MP-04 (cold start produces no output for 30-90 days)

D* formula
----------
The HSSM (Sprint 3) fits a log-normal dwell-time distribution per regime.
After label canonicalization (ascending activity), regime 0 is always the
slow regime. The mean dwell time of the slow regime is:

    D* = exp(dur_mu + dur_sigma^2 / 2)       [log-normal mean]

where dur_mu and dur_sigma come from HSSMFit.duration_parameters[slow_regime_id],
i.e. directly from the EM-fitted model parameters, NOT from regime posteriors.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from upstream_interfaces import HSSMFit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

class ColdStartStage(Enum):
    """
    The five Cold Start stages, exactly as specified in Sprint 10 Day 29.

    Transition is evidence-gated, not schedule-gated at Stage 4.
    Early regime estimates computed internally during Stage 2 are NEVER
    surfaced — violating this rule destroys user trust.
    """
    STAGE_0 = 0   # Days  1– 7  — zero inference, nothing computed or surfaced
    STAGE_1 = 1   # Days  8–29  — tentative internal patterns, no claims
    STAGE_2 = 2   # Days 30–59  — first HSSM fit, Level-1 output only
    STAGE_3 = 3   # Days 60–89  — DivergenceState accumulation begins, no claims yet
    STAGE_4 = 4   # Day  90+    — claims begin, evidence-gated


# Day boundaries that trigger automatic stage advancement.
# Advancement to STAGE_4 additionally requires evidence (see StateMachine).
_STAGE_DAY_BOUNDARIES: dict[ColdStartStage, int] = {
    ColdStartStage.STAGE_0: 1,   # active from Day 1
    ColdStartStage.STAGE_1: 8,   # enters on Day 8
    ColdStartStage.STAGE_2: 30,  # enters on Day 30
    ColdStartStage.STAGE_3: 60,  # enters on Day 60
    ColdStartStage.STAGE_4: 90,  # eligible from Day 90 (evidence still required)
}


# ---------------------------------------------------------------------------
# State snapshot
# ---------------------------------------------------------------------------

@dataclass
class ColdStartState:
    """
    Immutable snapshot of a single user's cold-start status.

    Returned by ColdStartStateMachine.evaluate(); callers must never
    surface any claim or inference when can_surface_claims is False.
    """
    user_id: str
    day: int
    stage: ColdStartStage

    # Whether the product layer is allowed to surface any claim at all.
    # Hard False for stages 0-3; True only when stage == STAGE_4 AND
    # evidence_gate_passed is True.
    can_surface_claims: bool

    # Whether an HSSM fit has been performed AND the result is safe to
    # expose to downstream systems.
    # IMPORTANT (S10.2): This is False for Stage-2 states even when the
    # HSSM was fitted internally, because exposing hssm_fitted=True would
    # allow downstream callers to infer that estimates are available and
    # act on them — leaking internal Stage-2 estimates through a side channel.
    hssm_fitted: bool

    # Honest, specific message the product layer MAY show the user.
    # Never "coming soon." Always a concrete day estimate.
    user_facing_message: str

    # Whether internal regime estimates exist but must stay internal.
    # Callers must assert this is True only in Stage 2 (Days 30-59)
    # and must guarantee the estimates are never passed downstream.
    internal_estimates_only: bool = False

    # ------------------------------------------------------------------
    # Output contract enforcement  (S10.2)
    # ------------------------------------------------------------------

    def assert_downstream_safe(self) -> None:
        """
        Raise RuntimeError if this state must not cross a pipeline boundary.

        Call this at every handoff point before passing ColdStartState to:
          - The Claims Engine (Sprint 9)
          - The Divergence accumulator (Sprint 8)
          - The Mirror pipeline (Sprint 12)
          - Any output or product layer

        Why this matters:
        Stage-2 (Days 30-59) is "internal model fitting only". No estimate,
        partial claim, or HSSM result may reach the Claims Engine or any
        output-producing system. This method is the code-enforced contract;
        documentation alone is not sufficient.

        Raises:
            RuntimeError: if internal_estimates_only is True, i.e. stage == STAGE_2.
        """
        if self.internal_estimates_only:
            raise RuntimeError(
                f"user={self.user_id} day={self.day}: ColdStartState is Stage-2 "
                f"(internal_estimates_only=True). This state MUST NOT cross a pipeline "
                f"boundary into the Claims Engine, Mirror, or any output layer. "
                f"Stage-2 estimates are internal only — call this method at every "
                f"downstream handoff to enforce the contract in code, not just in docs."
            )


# ---------------------------------------------------------------------------
# D* estimator  (Day 28) — uses REAL HSSM duration parameters
# ---------------------------------------------------------------------------

def estimate_slow_phase_duration(
    fitted_hssm: "HSSMFit",
    slow_regime_id: int,
) -> float:
    """
    Compute D* — the mean dwell time of the slow regime — from the HSSM's
    fitted log-normal duration parameters.

    Formula (log-normal mean):
        D* = exp(dur_mu + dur_sigma^2 / 2)

    This is mathematically exact for the log-normal duration prior used by
    the Sprint 3 backbone HSSM (GaussianHSMM / KimHSSMModel). The parameters
    dur_mu and dur_sigma come directly from the EM M-step fitting of the
    dwell-time distribution — they are NOT derived from regime posteriors.

    Why this is correct:
        The backbone HSSM uses an explicit semi-Markov duration model with a
        log-normal prior (backbone/hssm/model.py). After the EM algorithm
        converges, model.dur_mu[k] and model.dur_sigma[k] are the log-space
        mean and standard deviation of the dwell-time distribution for regime k.
        The mean of a log-normal random variable X ~ LogNormal(mu, sigma) is:
            E[X] = exp(mu + sigma^2 / 2)
        This is the quantity we want: expected number of sessions spent in the
        slow regime before switching out.

    Args:
        fitted_hssm:    HSSMFit from Sprint 3, containing duration_parameters
                        per regime. Must have been through canonicalize_labels()
                        so that slow_regime_id=0 is the actual slowest regime.
        slow_regime_id: Which regime is the slow one (always 0 after label
                        canonicalization, passed explicitly for clarity).

    Returns:
        D* as a float (expected dwell time in session-index units).

    Raises:
        KeyError:   if slow_regime_id is not in duration_parameters.
        ValueError: if parameters produce a degenerate or non-positive result.
    """
    if slow_regime_id not in fitted_hssm.duration_parameters:
        raise KeyError(
            f"slow_regime_id={slow_regime_id} not found in HSSMFit.duration_parameters. "
            f"Available regime IDs: {list(fitted_hssm.duration_parameters.keys())}"
        )

    params   = fitted_hssm.duration_parameters[slow_regime_id]
    dur_mu   = float(params["dur_mu"])
    dur_sigma = float(params["dur_sigma"])

    if dur_sigma <= 0:
        raise ValueError(
            f"dur_sigma={dur_sigma} must be positive (log-normal scale parameter)."
        )

    d_hat = math.exp(dur_mu + (dur_sigma ** 2) / 2.0)

    if d_hat <= 0 or not math.isfinite(d_hat):
        raise ValueError(
            f"D* = {d_hat} is non-positive or non-finite. "
            f"Check HSSM fit quality: dur_mu={dur_mu}, dur_sigma={dur_sigma}."
        )

    logger.debug(
        "D* estimated: dur_mu=%.4f  dur_sigma=%.4f  D*=%.2f sessions",
        dur_mu, dur_sigma, d_hat,
    )
    return d_hat


def compute_observation_window(d_hat: float, sessions_per_day: float) -> float:
    """
    Compute the minimum divergence observation window in calendar days.

    D* from the HSSM is expressed in session-index units (expected sessions
    in the slow regime). To compare against calendar-day stage boundaries
    the window must be converted to days:

        observation_window_days = 2 * D* / sessions_per_day

    where sessions_per_day = n_present_sessions / days_since_onboarding.

    The baked-in assumption of 1 session/day was the original bug: a user
    with D*=45 sessions who attends 0.5 sessions/day needs 90 days of
    calendar time, not 45.

    Worked examples (sessions_per_day=0.5):
        D*=45  -> window=180 days    (not 90)
        D*=20  -> window= 80 days    (not 40)
        D*=90  -> window=360 days    (not 180)

    Args:
        d_hat:            D* in session-index units (log-normal mean dwell time).
        sessions_per_day: Average user session frequency (sessions / calendar day).
                          Compute as n_present_sessions / day.
                          Must be positive.

    Returns:
        Minimum observation window in calendar days.

    Raises:
        ValueError: if d_hat or sessions_per_day are non-positive.
    """
    if d_hat <= 0:
        raise ValueError(f"D* must be positive, got {d_hat}.")
    if sessions_per_day <= 0:
        raise ValueError(
            f"sessions_per_day must be positive, got {sessions_per_day}. "
            f"Compute as n_present_sessions / day (both must be > 0)."
        )
    window_days = (2.0 * d_hat) / sessions_per_day
    logger.debug(
        "observation_window: D*=%.2f sessions  sessions_per_day=%.4f  -> %.2f calendar days",
        d_hat, sessions_per_day, window_days,
    )
    return window_days


# ---------------------------------------------------------------------------
# Main state machine
# ---------------------------------------------------------------------------

class ColdStartStateMachine:
    """
    Explicit 5-stage Cold Start Protocol state machine (Sprint 10 Day 29).

    Usage
    -----
    >>> sm = ColdStartStateMachine(user_id="u_001")
    >>> state = sm.evaluate(day=1)
    >>> assert not state.can_surface_claims          # Stage 0 — silent
    >>> state = sm.evaluate(day=95, evidence_gate_passed=True)
    >>> assert state.can_surface_claims              # Stage 4 — claims OK

    MLflow logging
    --------------
    Call sm.log_to_mlflow(d_hat, window) after fitting D* to satisfy
    the Definition of Done's auditability requirement.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self._current_stage = ColdStartStage.STAGE_0

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        day: int,
        evidence_gate_passed: bool = False,
        hssm_fitted: bool = False,
    ) -> ColdStartState:
        """
        Advance the state machine to the correct stage for the given day
        and return a ColdStartState snapshot.

        Args:
            day:                  Calendar day since user onboarding (1-indexed).
            evidence_gate_passed: True only when the upstream HSSM/Divergence
                                  pipeline has confirmed sufficient evidence to
                                  allow claim generation. Ignored before Stage 4.
            hssm_fitted:          True if an HSSM fit has been performed this
                                  session.

        Returns:
            ColdStartState — the product layer must consult can_surface_claims
            before emitting any insight or claim.
        """
        if day < 1:
            raise ValueError(f"day must be >= 1, got {day}.")

        stage = self._resolve_stage(day)
        self._current_stage = stage

        can_surface = self._can_surface_claims(stage, evidence_gate_passed)
        internal_only = (stage == ColdStartStage.STAGE_2)

        # S10.2 output contract: Stage-2 is internal-only.
        # Even when the HSSM was fitted (hssm_fitted=True was passed in),
        # we redact it to False for Stage-2 states so that downstream callers
        # cannot inspect hssm_fitted to infer that estimates exist internally.
        # A caller checking `state.hssm_fitted` and acting on it would leak
        # Stage-2 estimates through a side channel — redacting it closes that.
        effective_hssm_fitted = hssm_fitted and not internal_only

        return ColdStartState(
            user_id=self.user_id,
            day=day,
            stage=stage,
            can_surface_claims=can_surface,
            hssm_fitted=effective_hssm_fitted,
            user_facing_message=self._user_facing_message(day, stage),
            internal_estimates_only=internal_only,
        )

    # ------------------------------------------------------------------
    # Stage resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_stage(day: int) -> ColdStartStage:
        """
        Determine stage from day alone.
        Advancement to STAGE_4 is day-eligible at Day 90 but
        still evidence-gated separately in _can_surface_claims().
        """
        if day >= _STAGE_DAY_BOUNDARIES[ColdStartStage.STAGE_4]:
            return ColdStartStage.STAGE_4
        if day >= _STAGE_DAY_BOUNDARIES[ColdStartStage.STAGE_3]:
            return ColdStartStage.STAGE_3
        if day >= _STAGE_DAY_BOUNDARIES[ColdStartStage.STAGE_2]:
            return ColdStartStage.STAGE_2
        if day >= _STAGE_DAY_BOUNDARIES[ColdStartStage.STAGE_1]:
            return ColdStartStage.STAGE_1
        return ColdStartStage.STAGE_0

    @staticmethod
    def _can_surface_claims(
        stage: ColdStartStage,
        evidence_gate_passed: bool,
    ) -> bool:
        """
        Hard gate: claims are NEVER surfaced before Stage 4 AND evidence
        gate is passed. No exceptions.
        """
        if stage != ColdStartStage.STAGE_4:
            return False
        return evidence_gate_passed

    # ------------------------------------------------------------------
    # User-facing messaging  (honest, specific — never "coming soon")
    # ------------------------------------------------------------------

    @staticmethod
    def _user_facing_message(day: int, stage: ColdStartStage) -> str:
        """
        Return a structured, honest message the product layer may present.
        Never vague. Always a concrete day estimate.

        Per Sprint 10 Day 30: "After 45 days we can start building your
        behavioral model", never "coming soon."
        """
        days_to_stage1 = max(0, _STAGE_DAY_BOUNDARIES[ColdStartStage.STAGE_1] - day)
        days_to_stage2 = max(0, _STAGE_DAY_BOUNDARIES[ColdStartStage.STAGE_2] - day)
        days_to_stage4 = max(0, _STAGE_DAY_BOUNDARIES[ColdStartStage.STAGE_4] - day)

        messages = {
            ColdStartStage.STAGE_0: (
                f"Chronis is getting to know you. "
                f"Your first behavioral patterns will start forming in {days_to_stage1} day(s)."
            ),
            ColdStartStage.STAGE_1: (
                f"Chronis is building your behavioral model. "
                f"Your first model fit begins in {days_to_stage2} day(s)."
            ),
            ColdStartStage.STAGE_2: (
                "Your behavioral model is being refined. "
                "No insights yet — Chronis needs more time to be confident."
            ),
            ColdStartStage.STAGE_3: (
                f"Chronis is accumulating evidence. "
                f"Personalized insights will be ready in approximately {days_to_stage4} day(s)."
            ),
            ColdStartStage.STAGE_4: (
                "Chronis has enough data to share personalized insights with you."
            ),
        }
        return messages[stage]

    # ------------------------------------------------------------------
    # MLflow logging  (DoD requirement: Day 28)
    # ------------------------------------------------------------------

    def log_to_mlflow(
        self,
        d_hat: float,
        observation_window_days: float,
        sessions_per_day: float,
        run_id: Optional[str] = None,
    ) -> None:
        """
        Log D* and the derived observation window to MLflow.

        Definition of Done (Sprint 10):
            "D* and the derived window are logged per user, per fit, to MLflow."

        S10.2 constraint: Stage 2 estimates are INTERNAL ONLY.
        This method must never be called when the current stage is STAGE_2.
        The pipeline enforces this — calling here in Stage 2 raises to
        make the invariant explicit and catch any accidental future path.

        Args:
            d_hat:                   D* in session-index units.
            observation_window_days: Converted observation window in calendar days.
            sessions_per_day:        Session frequency used for conversion.
            run_id:                  Existing MLflow run ID to log into, or None
                                     to use the active run.

        Raises:
            RuntimeError: if current stage is STAGE_2 (internal estimates must
                          never be logged to an external audit system).
        """
        if self._current_stage == ColdStartStage.STAGE_2:
            raise RuntimeError(
                f"user={self.user_id}: Attempted to log Stage-2 internal estimates to "
                f"MLflow. Stage-2 estimates are INTERNAL ONLY and must never reach an "
                f"externally observable audit trail. This is a code path bug — the "
                f"pipeline must skip MLflow logging in Stage 2."
            )

        import mlflow  # lazy — avoids hard dep at module collection time

        params = {
            "cold_start.user_id":                   self.user_id,
            "cold_start.d_hat_sessions":            d_hat,
            "cold_start.sessions_per_day":          sessions_per_day,
            "cold_start.observation_window_days":   observation_window_days,
        }

        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_params(params)
        else:
            mlflow.log_params(params)

        logger.info(
            "Logged D*=%.1f sessions  sessions/day=%.4f  window=%.1f days for user=%s",
            d_hat, sessions_per_day, observation_window_days, self.user_id,
        )
