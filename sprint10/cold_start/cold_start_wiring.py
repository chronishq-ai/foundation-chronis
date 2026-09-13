"""
cold_start_wiring.py
Sprint 10 — Adapter between real Sprint 3/8 types and cold_start_pipeline.

Translates:
  HSSMFit       (upstream_interfaces.py, Sprint 3)  → fitted_hssm (for D*)
  DivergenceState (divergence_engine/state.py, Sprint 8) → evidence_gate_passed

Then calls run_cold_start_pipeline() and returns a ColdStartState.

D* source
---------
D* = exp(dur_mu + dur_sigma^2 / 2)  [log-normal mean]

where dur_mu and dur_sigma come from HSSMFit.duration_parameters[slow_regime_id].
These are the EM-fitted parameters from the Sprint 3 backbone HSSM.
They are NOT derived from regime_posteriors or any occupancy proxy.

Bridge from backbone to HSSMFit
--------------------------------
If you have a raw backbone.hssm.model.GaussianHSMM object, convert it via:

    from upstream_interfaces import hssm_fit_from_backbone
    fit = hssm_fit_from_backbone(model, user_id="u_001", fit_id="...")
"""

from __future__ import annotations

import logging
from typing import Optional

from upstream_interfaces import HSSMFit
from divergence_engine.state import DivergenceState
from cold_start.cold_start import ColdStartStage, ColdStartState, ColdStartStateMachine
from cold_start.cold_start_pipeline import run_cold_start_pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evidence gate — derived from Sprint 8 DivergenceState
# ---------------------------------------------------------------------------

def evidence_gate_passed(divergence_state: Optional[DivergenceState]) -> bool:
    """
    The Cold Start Stage 4 evidence gate is satisfied when:
      1. Sprint 8's MP-09 power gate passed (>=20 sessions/regime for Granger).
      2. The four divergence type scores are not ambiguous — dominant() is not None
         (top two scores differ by more than AMBIGUITY_THRESHOLD = 0.15).

    If no DivergenceState exists yet (user hasn't accumulated enough data for
    Sprint 8 to run), the gate is unconditionally False — Stage 4 cannot open
    without a real divergence result.
    """
    if divergence_state is None:
        return False

    power_ok = divergence_state.provenance.power_gate_passed
    unambiguous = divergence_state.type_scores.dominant() is not None

    gate = power_ok and unambiguous

    logger.debug(
        "evidence_gate: user=%s  power_gate=%s  unambiguous=%s  -> gate=%s",
        divergence_state.user_id, power_ok, unambiguous, gate,
    )
    return gate


# ---------------------------------------------------------------------------
# Single entry point for the rest of the system
# ---------------------------------------------------------------------------

def evaluate_cold_start(
    *,
    day: int,
    fitted_hssm: HSSMFit,
    n_sessions: int = 0,
    divergence_state: Optional[DivergenceState] = None,
    mlflow_run_id: Optional[str] = None,
) -> ColdStartState:
    """
    Full Sprint 10 evaluation wired to real Sprint 3 / Sprint 8 types.

    D* is computed from fitted_hssm.duration_parameters[slow_regime_id]
    using the log-normal mean formula. This is the ONLY correct source —
    not regime posteriors, not occupancy proxies.

    Args:
        day:              Days since onboarding, 1-indexed.
        fitted_hssm:      HSSMFit from Sprint 3 backbone (after canonicalize_labels).
                          Use hssm_fit_from_backbone() to convert a raw GaussianHSMM.
        n_sessions:       Number of present sessions observed. Used to convert
                          D* (session units) to calendar-day observation window.
                          Pass 0 to use the conservative fallback (day/2).
        divergence_state: Most recent Sprint 8 DivergenceState, or None.
        mlflow_run_id:    Active MLflow run to log D* into.

    Returns:
        ColdStartState — the authoritative gate checked by the Claims Engine.

    Raises:
        TypeError:  if fitted_hssm is not an HSSMFit instance.
        ValueError: if the HSSMFit fails structural validation (missing slow
                    regime params, n_regimes < 2, or empty user_id).
    """
    # S10.3: structural HSSM model/class validation.
    # The original tautological check (user_id != user_id) was a placeholder.
    # The correct fix validates the type and key invariants of the fit object,
    # not just its identity metadata.
    #
    # Check 1: type guard — must be an HSSMFit, not a raw backbone model
    if not isinstance(fitted_hssm, HSSMFit):
        raise TypeError(
            f"fitted_hssm must be an HSSMFit instance (from upstream_interfaces.py), "
            f"got {type(fitted_hssm).__name__!r}. "
            "Convert a raw backbone GaussianHSMM/KimHSSMModel via "
            "hssm_fit_from_backbone(model, user_id=...) before calling evaluate_cold_start."
        )
    # Check 2: minimum regime count
    if fitted_hssm.n_regimes < 2:
        raise ValueError(
            f"fitted_hssm.n_regimes={fitted_hssm.n_regimes} must be >= 2. "
            "The behavioral HSSM requires at least a slow regime and a fast regime. "
            "Re-fit with n_regimes >= 2."
        )
    # Check 3: slow regime must be present in duration_parameters
    if fitted_hssm.slow_regime_id not in fitted_hssm.duration_parameters:
        raise ValueError(
            f"fitted_hssm.slow_regime_id={fitted_hssm.slow_regime_id} is not in "
            f"duration_parameters (keys: {list(fitted_hssm.duration_parameters.keys())}). "
            "Ensure canonicalize_labels() was applied before hssm_fit_from_backbone() "
            "so that slow_regime_id=0 is always valid."
        )
    # Check 4: non-empty user_id
    if not fitted_hssm.user_id or not fitted_hssm.user_id.strip():
        raise ValueError(
            "fitted_hssm.user_id is empty or blank. "
            "Use hssm_fit_from_backbone(model, user_id=...) to build a valid fit."
        )

    if not fitted_hssm.converged:
        logger.warning(
            "user=%s: HSSMFit.converged=False — EM did not converge. "
            "D* estimate may be unreliable. Proceeding with available parameters.",
            fitted_hssm.user_id,
        )

    gate_passed = evidence_gate_passed(divergence_state)

    return run_cold_start_pipeline(
        user_id=fitted_hssm.user_id,
        day=day,
        n_sessions=n_sessions,
        fitted_hssm=fitted_hssm,
        evidence_gate_passed=gate_passed,
        mlflow_run_id=mlflow_run_id,
    )


def evaluate_cold_start_gated(
    *,
    day: int,
    user_id: str,
    n_present_sessions: int,
    fitted_hssm: Optional[HSSMFit] = None,
    divergence_state: Optional[DivergenceState] = None,
    mlflow_run_id: Optional[str] = None,
    min_sessions: int = 30,
) -> ColdStartState:
    """
    Gate-aware entry point: handles the case where the user hasn't yet cleared
    the Sprint 3 cold-start gate (< 30 present sessions).

    If n_present_sessions < min_sessions OR fitted_hssm is None, returns
    Stage 0 immediately without running D* estimation.

    Args:
        day:                  Days since onboarding, 1-indexed.
        user_id:              User identifier.
        n_present_sessions:   Number of non-missing sessions observed so far.
                              Use backbone.hssm.gating.count_present_sessions(X).
        fitted_hssm:          HSSMFit if the user has cleared the gate and a fit
                              exists, else None.
        divergence_state:     Sprint 8 DivergenceState, or None.
        mlflow_run_id:        Active MLflow run ID.
        min_sessions:         Cold-start gate threshold (default 30, per Sprint 3 spec).

    Returns:
        ColdStartState — Stage 0 if gated, otherwise full evaluation.
    """
    if fitted_hssm is None or n_present_sessions < min_sessions:
        logger.info(
            "user=%s: n_present_sessions=%d < min=%d or no fit — returning Stage 0.",
            user_id, n_present_sessions, min_sessions,
        )
        sm = ColdStartStateMachine(user_id=user_id)
        return ColdStartState(
            user_id=user_id,
            day=day,
            stage=ColdStartStage.STAGE_0,
            can_surface_claims=False,
            hssm_fitted=False,
            user_facing_message=(
                f"Chronis is still learning your patterns. "
                f"Check back in {max(0, 8 - day)} day(s)."
                if day < 8 else
                "Chronis needs more data before it can personalise insights for you."
            ),
            internal_estimates_only=False,
        )

    return evaluate_cold_start(
        day=day,
        n_sessions=n_present_sessions,
        fitted_hssm=fitted_hssm,
        divergence_state=divergence_state,
        mlflow_run_id=mlflow_run_id,
    )
