"""
cold_start_pipeline.py
Sprint 10 — Integration Entry Point

Wires together:
  1. Stage check             (Stage 0 = zero inference — checked FIRST)
  2. D* estimator            (Day 28, uses HSSMFit.duration_parameters directly)
  3. Observation window      (Day 28, = 2 * D*)
  4. Cold Start state machine (Day 29)
  5. MLflow logging          (DoD)

Callers: Claims Engine (Sprint 9), DivergenceState accumulator (Sprint 8),
         product surface layer (Sprint 12+).

Usage
-----
    from cold_start.cold_start_pipeline import run_cold_start_pipeline
    from upstream_interfaces import HSSMFit, hssm_fit_from_backbone

    # Convert Sprint 3 backbone model to HSSMFit contract
    fit = hssm_fit_from_backbone(backbone_model, user_id="u_001", fit_id="...")

    state = run_cold_start_pipeline(
        user_id          = "u_001",
        day              = 35,
        fitted_hssm      = fit,
        evidence_gate_passed = divergence_engine.gate_passed(),
        mlflow_run_id    = active_run.info.run_id,
    )

    if state.can_surface_claims:
        yield_claims(state)
    else:
        yield_user_message(state.user_facing_message)
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from cold_start.cold_start import (
    ColdStartStage,
    ColdStartState,
    ColdStartStateMachine,
    compute_observation_window,
    estimate_slow_phase_duration,
)

if TYPE_CHECKING:
    from upstream_interfaces import HSSMFit

logger = logging.getLogger(__name__)


def run_cold_start_pipeline(
    user_id: str,
    day: int,
    fitted_hssm: "HSSMFit",
    evidence_gate_passed: bool = False,
    mlflow_run_id: Optional[str] = None,
) -> ColdStartState:
    """
    Full Sprint 10 pipeline for a single user evaluation.

    Flow (stage-first — Stage 0 short-circuits before any inference runs)
    -----------------------------------------------------------------------
    1. Determine Cold Start Stage from day.
    2. If Stage 0 (Days 1-7): return immediately. D* is NOT computed.
       "Stage 0 = zero inference" is non-negotiable.
    3. Estimate D* from fitted_hssm.duration_parameters[slow_regime_id]
       using the log-normal mean formula: exp(dur_mu + dur_sigma^2 / 2).
    4. Derive the minimum observation window (2 * D*).
    5. Advance the Cold Start state machine to the correct stage.
    6. Log D* and window to MLflow for auditability (DoD requirement).
    7. Return the ColdStartState snapshot.

    Args:
        user_id:              Unique user identifier.
        day:                  Days since onboarding, 1-indexed.
        fitted_hssm:          HSSMFit from Sprint 3 backbone, containing the
                              fitted log-normal duration parameters for each
                              regime. This is the ONLY valid source of D*.
        evidence_gate_passed: True only when DivergenceState accumulator
                              has confirmed enough evidence for claims.
                              Ignored by the state machine before Stage 4.
        mlflow_run_id:        Active MLflow run to log into. Pass None to
                              log into whatever run is currently active.

    Returns:
        ColdStartState — the authoritative gate for the product layer.

    Raises:
        ValueError: propagated from estimator/window if inputs are degenerate.
    """
    # --- Step 1: Determine Cold Start Stage (stage-first, inference-second) --
    # Per Sprint 10 spec: "Stage 0 = zero inference."
    # D* must NOT be computed before the stage check — doing so would run
    # HSSM-derived inference on Day 1-7 users, which violates the protocol.
    sm = ColdStartStateMachine(user_id=user_id)
    stage = ColdStartStateMachine._resolve_stage(day)

    if stage == ColdStartStage.STAGE_0:
        # Stage 0: no inference, no D*, no MLflow logging. Return immediately.
        logger.info(
            "user=%s  day=%d  Stage 0 — zero inference. Skipping D* calculation.",
            user_id, day,
        )
        return sm.evaluate(day=day, evidence_gate_passed=False, hssm_fitted=False)

    # --- Step 2: Estimate D* from fitted HSSM duration parameters -----------
    # Uses the log-normal mean: exp(dur_mu + dur_sigma^2 / 2)
    # Source: HSSMFit.duration_parameters[slow_regime_id] — EM-fitted values,
    # NOT regime posteriors.
    d_hat = estimate_slow_phase_duration(
        fitted_hssm=fitted_hssm,
        slow_regime_id=fitted_hssm.slow_regime_id,
    )
    logger.debug("user=%s  day=%d  D*=%.2f sessions", user_id, day, d_hat)

    # --- Step 3: Observation window ------------------------------------------
    window = compute_observation_window(d_hat)
    logger.debug("user=%s  observation_window=%.2f sessions", user_id, window)

    # --- Step 4: Advance state machine to correct stage ----------------------
    state = sm.evaluate(
        day=day,
        evidence_gate_passed=evidence_gate_passed,
        hssm_fitted=True,
    )
    logger.info(
        "user=%s  day=%d  stage=%s  can_surface_claims=%s",
        user_id, day, state.stage.name, state.can_surface_claims,
    )

    # --- Step 5: MLflow logging (DoD: logged per user, per fit) --------------
    sm.log_to_mlflow(
        d_hat=d_hat,
        observation_window=window,
        run_id=mlflow_run_id,
    )

    return state
