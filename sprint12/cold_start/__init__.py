"""
cold_start — Sprint 10 Cold Start Compass & Threshold Calibration II

Public API
----------
Core:
    ColdStartStage                — 5-stage enum (STAGE_0 … STAGE_4)
    ColdStartState                — immutable snapshot returned to callers
    ColdStartStateMachine         — explicit state machine
    estimate_slow_phase_duration  — D* from HSSMFit.duration_parameters (log-normal mean)
    compute_observation_window    — 2 * D*

Pipeline (orchestration):
    run_cold_start_pipeline       — single user evaluation (requires HSSMFit)

Wiring (Sprint 3/8 adapter):
    evaluate_cold_start           — wired to HSSMFit + DivergenceState
    evaluate_cold_start_gated     — gate-aware variant (handles < 30 session users)
    evidence_gate_passed          — derives bool from DivergenceState
"""

from cold_start.cold_start import (
    ColdStartStage,
    ColdStartState,
    ColdStartStateMachine,
    compute_observation_window,
    estimate_slow_phase_duration,
)
from cold_start.cold_start_pipeline import run_cold_start_pipeline
from cold_start.cold_start_wiring import (
    evaluate_cold_start,
    evaluate_cold_start_gated,
    evidence_gate_passed,
)

__all__ = [
    "ColdStartStage",
    "ColdStartState",
    "ColdStartStateMachine",
    "estimate_slow_phase_duration",
    "compute_observation_window",
    "run_cold_start_pipeline",
    "evaluate_cold_start",
    "evaluate_cold_start_gated",
    "evidence_gate_passed",
]
