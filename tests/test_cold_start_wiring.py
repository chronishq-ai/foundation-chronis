"""
test_cold_start_wiring.py
Sprint 10 — Regression suite for cold_start_wiring.py

Tests the adapter layer between real Sprint 3 HSSMFit / Sprint 8 DivergenceState
and the cold_start_pipeline.

Tests:
  - evidence_gate_passed: gate logic from DivergenceState
  - evaluate_cold_start: full wiring with HSSMFit
  - evaluate_cold_start_gated: handles not-yet-fitted users (< 30 sessions)
  - 180-day simulation through wiring
"""

import math
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cold_start.cold_start import ColdStartStage
from cold_start.cold_start_wiring import (
    evaluate_cold_start,
    evaluate_cold_start_gated,
    evidence_gate_passed,
)
from upstream_interfaces import HSSMFit, hssm_fit_from_backbone


# ---------------------------------------------------------------------------
# Helpers — build real-ish type instances
# ---------------------------------------------------------------------------

def _make_hssm_fit(
    d_hat_target: float = 45.0,
    dur_sigma: float = 0.5,
    n_regimes: int = 2,
    user_id: str = "user_001",
    converged: bool = True,
) -> HSSMFit:
    """Build an HSSMFit whose slow regime gives D* ≈ d_hat_target."""
    dur_mu = math.log(d_hat_target) - (dur_sigma ** 2) / 2.0
    duration_parameters = {
        0: {"dur_mu": dur_mu, "dur_sigma": dur_sigma},          # slow regime
        **{k: {"dur_mu": dur_mu - 1.0, "dur_sigma": dur_sigma}  # faster regimes
           for k in range(1, n_regimes)},
    }
    return HSSMFit(
        user_id=user_id,
        fit_id="test-fit",
        slow_regime_id=0,
        n_regimes=n_regimes,
        duration_parameters=duration_parameters,
        log_likelihood=-200.0,
        converged=converged,
    )


def _make_divergence_state(
    power_gate_passed: bool = True,
    dominant_type: str = "temporal",
) -> MagicMock:
    """Build a mock DivergenceState that satisfies the evidence gate."""
    ds = MagicMock()
    ds.user_id = "user_001"
    ds.provenance.power_gate_passed = power_gate_passed
    if dominant_type is not None:
        ds.type_scores.dominant.return_value = dominant_type
    else:
        ds.type_scores.dominant.return_value = None
    return ds


# ---------------------------------------------------------------------------
# hssm_fit_from_backbone bridge
# ---------------------------------------------------------------------------

class TestHSSMFitFromBackbone:

    def test_extracts_dur_mu_and_sigma_correctly(self):
        """hssm_fit_from_backbone must copy dur_mu / dur_sigma from backbone model."""
        mock_model = MagicMock()
        mock_model.K = 2
        mock_model.dur_mu = np.array([2.5, 1.5])
        mock_model.dur_sigma = np.array([0.4, 0.3])
        mock_model.log_likelihood_ = -150.0
        mock_model.converged_ = True

        fit = hssm_fit_from_backbone(mock_model, user_id="u1", fit_id="f1")

        assert fit.slow_regime_id == 0
        assert fit.n_regimes == 2
        assert abs(fit.duration_parameters[0]["dur_mu"] - 2.5) < 1e-9
        assert abs(fit.duration_parameters[0]["dur_sigma"] - 0.4) < 1e-9
        assert abs(fit.duration_parameters[1]["dur_mu"] - 1.5) < 1e-9
        assert fit.converged is True

    def test_slow_regime_id_is_always_zero(self):
        """After canonicalize_labels, slow_regime_id is always 0."""
        mock_model = MagicMock()
        mock_model.K = 3
        mock_model.dur_mu = np.array([3.0, 2.0, 1.0])
        mock_model.dur_sigma = np.array([0.5, 0.4, 0.3])
        mock_model.log_likelihood_ = -300.0
        mock_model.converged_ = True

        fit = hssm_fit_from_backbone(mock_model, user_id="u2", fit_id="f2")
        assert fit.slow_regime_id == 0
        assert fit.n_regimes == 3


# ---------------------------------------------------------------------------
# Evidence gate
# ---------------------------------------------------------------------------

class TestEvidenceGate:

    def test_none_divergence_state_is_false(self):
        assert not evidence_gate_passed(None)

    def test_power_gate_failed_blocks_evidence(self):
        ds = _make_divergence_state(power_gate_passed=False, dominant_type="temporal")
        assert not evidence_gate_passed(ds)

    def test_ambiguous_scores_block_evidence(self):
        ds = _make_divergence_state(power_gate_passed=True, dominant_type=None)
        assert not evidence_gate_passed(ds)

    def test_clear_dominant_and_power_gate_passed(self):
        ds = _make_divergence_state(power_gate_passed=True, dominant_type="temporal")
        assert evidence_gate_passed(ds)

    def test_power_gate_passed_but_all_scores_equal(self):
        ds = _make_divergence_state(power_gate_passed=True, dominant_type=None)
        assert not evidence_gate_passed(ds)


# ---------------------------------------------------------------------------
# evaluate_cold_start_gated — below-gate users
# ---------------------------------------------------------------------------

class TestEvaluateColdStartGated:

    def test_gated_user_always_stage_0(self):
        """User with < 30 sessions and no fit must get Stage 0 unconditionally."""
        state = evaluate_cold_start_gated(
            day=35,
            user_id="gated_user",
            n_present_sessions=20,
            fitted_hssm=None,
        )
        assert state.stage == ColdStartStage.STAGE_0
        assert not state.can_surface_claims

    def test_gated_user_no_mlflow_logging(self):
        """Gated users must not trigger any MLflow logging."""
        mock_mlflow = MagicMock()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            evaluate_cold_start_gated(
                day=35,
                user_id="gated_user",
                n_present_sessions=10,
                fitted_hssm=None,
            )
        mock_mlflow.log_params.assert_not_called()

    def test_fitted_user_above_gate_goes_through_pipeline(self):
        """User with >= 30 sessions and a fit must go through the real pipeline."""
        fit = _make_hssm_fit(d_hat_target=45.0, user_id="u_ready")
        mock_mlflow = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            state = evaluate_cold_start_gated(
                day=35,
                user_id="u_ready",
                n_present_sessions=50,
                fitted_hssm=fit,
            )

        assert state.stage == ColdStartStage.STAGE_2
        mock_mlflow.log_params.assert_called_once()


# ---------------------------------------------------------------------------
# Full simulation through wiring
# ---------------------------------------------------------------------------

class TestFullSimulationThroughWiring:

    def test_180_day_simulation_with_real_types(self):
        """Walk a synthetic user through 180 days using evaluate_cold_start."""
        fit = _make_hssm_fit(d_hat_target=45.0, user_id="sim_user")

        expected = {
            range(1, 8):    ColdStartStage.STAGE_0,
            range(8, 30):   ColdStartStage.STAGE_1,
            range(30, 60):  ColdStartStage.STAGE_2,
            range(60, 90):  ColdStartStage.STAGE_3,
            range(90, 181): ColdStartStage.STAGE_4,
        }

        mock_mlflow = MagicMock()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            for day in range(1, 181):
                divergence_state = (
                    _make_divergence_state(power_gate_passed=True, dominant_type="temporal")
                    if day >= 90 else None
                )
                state = evaluate_cold_start(
                    day=day,
                    fitted_hssm=fit,
                    divergence_state=divergence_state,
                )
                for day_range, expected_stage in expected.items():
                    if day in day_range:
                        assert state.stage == expected_stage, \
                            f"day={day}: expected {expected_stage}, got {state.stage}"
                        break

    def test_no_claims_surface_when_divergence_state_absent(self):
        """Without a DivergenceState, claims must never surface even on Day 90+."""
        fit = _make_hssm_fit(d_hat_target=45.0, user_id="no_div_user")
        mock_mlflow = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            for day in range(90, 121):
                state = evaluate_cold_start(
                    day=day,
                    fitted_hssm=fit,
                    divergence_state=None,
                )
                assert not state.can_surface_claims, \
                    f"claims surfaced on day {day} without DivergenceState"

    def test_claims_surface_on_day_90_with_valid_evidence(self):
        """Day 90 + valid DivergenceState = can_surface_claims = True."""
        fit = _make_hssm_fit(d_hat_target=45.0, user_id="ready_user")
        div_state = _make_divergence_state(power_gate_passed=True, dominant_type="temporal")
        mock_mlflow = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            state = evaluate_cold_start(
                day=90,
                fitted_hssm=fit,
                divergence_state=div_state,
            )

        assert state.stage == ColdStartStage.STAGE_4
        assert state.can_surface_claims
