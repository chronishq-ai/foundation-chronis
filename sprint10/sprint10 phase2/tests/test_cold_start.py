"""
test_cold_start.py
Sprint 10 — Regression suite for cold_start.py

Tests:
  - estimate_slow_phase_duration: log-normal mean formula with real HSSMFit inputs
  - compute_observation_window: 2 * D*
  - Stage boundaries (0-4)
  - Stage 0 silence (no claims, no inference)
  - Evidence gate (Stage 4 + gate required)
  - internal_estimates_only flag (Stage 2 only)
  - 180-day synthetic simulation
  - User-facing messages (never "coming soon")
  - MLflow logging (D* and window logged per fit)
"""

import math
import sys
from unittest.mock import MagicMock, patch

import pytest

from cold_start.cold_start import (
    ColdStartStage,
    ColdStartState,
    ColdStartStateMachine,
    compute_observation_window,
    estimate_slow_phase_duration,
)
from cold_start.cold_start_pipeline import run_cold_start_pipeline
from upstream_interfaces import HSSMFit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hssm_fit(
    slow_dur_mu: float,
    slow_dur_sigma: float,
    n_regimes: int = 2,
    user_id: str = "test_user",
) -> HSSMFit:
    """
    Build a synthetic HSSMFit with controlled duration parameters.

    D* = exp(slow_dur_mu + slow_dur_sigma^2 / 2)
    """
    duration_parameters = {
        k: {
            "dur_mu":    slow_dur_mu if k == 0 else slow_dur_mu - 1.0,
            "dur_sigma": slow_dur_sigma,
        }
        for k in range(n_regimes)
    }
    return HSSMFit(
        user_id=user_id,
        fit_id="test-fit-id",
        slow_regime_id=0,
        n_regimes=n_regimes,
        duration_parameters=duration_parameters,
        log_likelihood=-100.0,
        converged=True,
    )


def _dur_mu_for_d_hat(d_hat: float, dur_sigma: float = 0.5) -> float:
    """Inverse of log-normal mean: dur_mu = log(d_hat) - dur_sigma^2 / 2"""
    return math.log(d_hat) - (dur_sigma ** 2) / 2.0


# ---------------------------------------------------------------------------
# D* estimator
# ---------------------------------------------------------------------------

class TestDHatEstimator:

    @pytest.mark.parametrize("d_hat_target,sessions_per_day,expected_window_days", [
        (45, 1.0, 90),    # 1 session/day: D*=45 -> 90 days
        (20, 1.0, 40),    # 1 session/day: D*=20 -> 40 days
        (90, 1.0, 180),   # 1 session/day: D*=90 -> 180 days
        (45, 0.5, 180),   # 0.5 sessions/day: D*=45 -> 180 days (unit bug was here)
        (45, 2.0, 45),    # 2 sessions/day: D*=45 -> 45 days
    ])
    def test_worked_examples(self, d_hat_target, sessions_per_day, expected_window_days):
        """
        Sprint 10 Day 28 worked examples + unit-conversion cases.
        D* is in session units; observation window must be in calendar days.
        """
        dur_sigma = 0.5
        dur_mu = _dur_mu_for_d_hat(d_hat_target, dur_sigma)
        fit = _make_hssm_fit(slow_dur_mu=dur_mu, slow_dur_sigma=dur_sigma)

        d_hat = estimate_slow_phase_duration(fit, slow_regime_id=0)
        window = compute_observation_window(d_hat, sessions_per_day=sessions_per_day)

        assert abs(d_hat - d_hat_target) < 0.5, f"D*={d_hat} not close to target {d_hat_target}"
        assert abs(window - expected_window_days) < 1.0, (
            f"window={window:.2f} expected {expected_window_days} days "
            f"(D*={d_hat_target}, spd={sessions_per_day})"
        )

    def test_lognormal_mean_formula_exact(self):
        """D* = exp(dur_mu + dur_sigma^2/2) — verify exact formula."""
        dur_mu, dur_sigma = 2.0, 0.3
        expected = math.exp(dur_mu + dur_sigma**2 / 2.0)
        fit = _make_hssm_fit(slow_dur_mu=dur_mu, slow_dur_sigma=dur_sigma)
        d_hat = estimate_slow_phase_duration(fit, slow_regime_id=0)
        assert abs(d_hat - expected) < 1e-9

    def test_missing_regime_id_raises_key_error(self):
        fit = _make_hssm_fit(slow_dur_mu=2.0, slow_dur_sigma=0.5, n_regimes=2)
        with pytest.raises(KeyError):
            estimate_slow_phase_duration(fit, slow_regime_id=99)

    def test_zero_dur_sigma_raises_value_error(self):
        fit = _make_hssm_fit(slow_dur_mu=2.0, slow_dur_sigma=0.0)
        with pytest.raises(ValueError):
            estimate_slow_phase_duration(fit, slow_regime_id=0)

    def test_uses_slow_regime_id_from_fit(self):
        """slow_regime_id=0 has a larger D* than regime 1 (lower dur_mu for regime 1)."""
        fit = _make_hssm_fit(slow_dur_mu=3.0, slow_dur_sigma=0.5)
        d_hat_slow = estimate_slow_phase_duration(fit, slow_regime_id=0)
        d_hat_fast = estimate_slow_phase_duration(fit, slow_regime_id=1)
        assert d_hat_slow > d_hat_fast


# ---------------------------------------------------------------------------
# Stage boundaries
# ---------------------------------------------------------------------------

class TestStageBoundaries:

    @pytest.mark.parametrize("day,expected_stage", [
        (1,   ColdStartStage.STAGE_0),
        (7,   ColdStartStage.STAGE_0),
        (8,   ColdStartStage.STAGE_1),
        (29,  ColdStartStage.STAGE_1),
        (30,  ColdStartStage.STAGE_2),
        (59,  ColdStartStage.STAGE_2),
        (60,  ColdStartStage.STAGE_3),
        (89,  ColdStartStage.STAGE_3),
        (90,  ColdStartStage.STAGE_4),
        (180, ColdStartStage.STAGE_4),
    ])
    def test_stage_at_day(self, day, expected_stage):
        sm = ColdStartStateMachine(user_id="u")
        state = sm.evaluate(day=day)
        assert state.stage == expected_stage


# ---------------------------------------------------------------------------
# Stage 0 silence
# ---------------------------------------------------------------------------

class TestStage0Silence:

    @pytest.mark.parametrize("day", [1, 2, 3, 4, 5, 6, 7])
    def test_no_output_before_day_8(self, day):
        sm = ColdStartStateMachine(user_id="u")
        state = sm.evaluate(day=day)
        assert not state.can_surface_claims
        assert state.stage == ColdStartStage.STAGE_0

    def test_first_allowed_output_is_day_8(self):
        sm = ColdStartStateMachine(user_id="u")
        state = sm.evaluate(day=8)
        assert state.stage == ColdStartStage.STAGE_1


# ---------------------------------------------------------------------------
# Stage 0 pipeline short-circuit — D* must NOT be called on Stage 0 days
# ---------------------------------------------------------------------------

class TestStage0NoDStarComputation:

    def test_pipeline_does_not_call_d_star_on_stage_0_days(self):
        """
        The pipeline must return Stage 0 immediately on Day 1-7 without
        calling estimate_slow_phase_duration at all.
        """
        fit = _make_hssm_fit(slow_dur_mu=3.0, slow_dur_sigma=0.5)

        with patch("cold_start.cold_start_pipeline.estimate_slow_phase_duration") as mock_d_star:
            with patch("cold_start.cold_start_pipeline.ColdStartStateMachine.log_to_mlflow"):
                state = run_cold_start_pipeline(
                    user_id="u_001",
                    day=5,
                    fitted_hssm=fit,
                )
        mock_d_star.assert_not_called()
        assert state.stage == ColdStartStage.STAGE_0

    def test_pipeline_calls_d_star_on_stage_1_days(self):
        """D* IS computed on Day 8+."""
        fit = _make_hssm_fit(slow_dur_mu=3.0, slow_dur_sigma=0.5)

        with patch("cold_start.cold_start_pipeline.estimate_slow_phase_duration",
                   return_value=20.0) as mock_d_star:
            with patch("cold_start.cold_start_pipeline.ColdStartStateMachine.log_to_mlflow"):
                run_cold_start_pipeline(
                    user_id="u_001",
                    day=10,
                    fitted_hssm=fit,
                )
        mock_d_star.assert_called_once()


# ---------------------------------------------------------------------------
# Evidence gate
# ---------------------------------------------------------------------------

class TestEvidenceGate:

    def test_stage_4_without_evidence_no_claims(self):
        sm = ColdStartStateMachine(user_id="u")
        state = sm.evaluate(day=95, evidence_gate_passed=False, hssm_fitted=True)
        assert state.stage == ColdStartStage.STAGE_4
        assert not state.can_surface_claims

    def test_stage_4_with_evidence_allows_claims(self):
        sm = ColdStartStateMachine(user_id="u")
        state = sm.evaluate(day=95, evidence_gate_passed=True, hssm_fitted=True)
        assert state.stage == ColdStartStage.STAGE_4
        assert state.can_surface_claims


# ---------------------------------------------------------------------------
# internal_estimates_only flag
# ---------------------------------------------------------------------------

class TestInternalEstimatesFlag:

    @pytest.mark.parametrize("day,expected_internal_only", [
        (29, False),
        (30, True),
        (59, True),
        (60, False),
    ])
    def test_internal_estimates_only_flag(self, day, expected_internal_only):
        sm = ColdStartStateMachine(user_id="u")
        state = sm.evaluate(day=day)
        assert state.internal_estimates_only == expected_internal_only

    @pytest.mark.parametrize("day", [30, 45, 59])
    def test_stage2_hssm_fitted_is_redacted_to_false(self, day):
        """
        S10.2: Even when hssm_fitted=True is passed in, Stage-2 states must
        expose hssm_fitted=False so downstream callers cannot infer an internal
        fit ran and act on it as a side channel.
        """
        sm = ColdStartStateMachine(user_id="u")
        state = sm.evaluate(day=day, hssm_fitted=True)  # internal fit DID run
        assert state.stage == ColdStartStage.STAGE_2
        assert state.internal_estimates_only is True
        # hssm_fitted must be redacted — downstream must not see it as True
        assert state.hssm_fitted is False, (
            f"Stage-2 state exposed hssm_fitted=True on day={day}. "
            "This leaks the existence of an internal fit to downstream callers."
        )

    @pytest.mark.parametrize("day", [8, 29, 60, 90])
    def test_non_stage2_hssm_fitted_is_not_redacted(self, day):
        """Outside Stage 2, hssm_fitted=True should be preserved."""
        sm = ColdStartStateMachine(user_id="u")
        state = sm.evaluate(day=day, hssm_fitted=True, evidence_gate_passed=(day >= 90))
        assert state.hssm_fitted is True


class TestAssertDownstreamSafe:
    """S10.2: assert_downstream_safe() is the code-enforced output contract."""

    def test_raises_for_stage2_state(self):
        """Stage-2 state must raise when assert_downstream_safe() is called."""
        sm = ColdStartStateMachine(user_id="u")
        state = sm.evaluate(day=35)  # Stage 2
        assert state.internal_estimates_only is True
        with pytest.raises(RuntimeError, match="Stage-2"):
            state.assert_downstream_safe()

    @pytest.mark.parametrize("day,gate", [
        (1, False),    # Stage 0
        (10, False),   # Stage 1
        (65, False),   # Stage 3
        (95, True),    # Stage 4
    ])
    def test_does_not_raise_outside_stage2(self, day, gate):
        """Non-Stage-2 states must not raise assert_downstream_safe()."""
        sm = ColdStartStateMachine(user_id="u")
        state = sm.evaluate(day=day, evidence_gate_passed=gate)
        # Should not raise
        state.assert_downstream_safe()


# ---------------------------------------------------------------------------
# 180-day synthetic simulation
# ---------------------------------------------------------------------------

class TestFullSimulation:

    def test_180_day_synthetic_user(self):
        """Walk a synthetic user through 180 days and verify all stage transitions."""
        fit = _make_hssm_fit(
            slow_dur_mu=_dur_mu_for_d_hat(45),
            slow_dur_sigma=0.5,
        )

        expected_stages = {
            range(1, 8):    ColdStartStage.STAGE_0,
            range(8, 30):   ColdStartStage.STAGE_1,
            range(30, 60):  ColdStartStage.STAGE_2,
            range(60, 90):  ColdStartStage.STAGE_3,
            range(90, 181): ColdStartStage.STAGE_4,
        }

        with patch("cold_start.cold_start_pipeline.ColdStartStateMachine.log_to_mlflow"):
            for day in range(1, 181):
                state = run_cold_start_pipeline(
                    user_id="sim_user",
                    day=day,
                    fitted_hssm=fit,
                    evidence_gate_passed=(day >= 90),
                )
                for day_range, expected_stage in expected_stages.items():
                    if day in day_range:
                        assert state.stage == expected_stage, \
                            f"day={day}: expected {expected_stage}, got {state.stage}"
                        break

    def test_no_early_surface_in_any_stage(self):
        """can_surface_claims must be False for every day before 90."""
        fit = _make_hssm_fit(
            slow_dur_mu=_dur_mu_for_d_hat(45),
            slow_dur_sigma=0.5,
        )
        with patch("cold_start.cold_start_pipeline.ColdStartStateMachine.log_to_mlflow"):
            for day in range(1, 90):
                state = run_cold_start_pipeline(
                    user_id="u",
                    day=day,
                    fitted_hssm=fit,
                    evidence_gate_passed=True,  # even with gate passed, must not surface
                )
                assert not state.can_surface_claims, f"claims surfaced on day {day}"


# ---------------------------------------------------------------------------
# User-facing messages
# ---------------------------------------------------------------------------

class TestUserFacingMessages:

    @pytest.mark.parametrize("day", [1, 10, 35, 65, 95])
    def test_no_vague_messaging(self, day):
        """Messages must never be vague — "coming soon" is explicitly banned."""
        sm = ColdStartStateMachine(user_id="u")
        state = sm.evaluate(day=day)
        assert "coming soon" not in state.user_facing_message.lower()

    def test_stage_0_message_contains_day_estimate(self):
        sm = ColdStartStateMachine(user_id="u")
        state = sm.evaluate(day=1)
        assert "day" in state.user_facing_message.lower()


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

class TestMLflowLogging:

    def test_d_hat_and_window_logged(self):
        """D* and window_days must be logged when estimate_slow_phase_duration is called."""
        mock_mlflow = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            sm = ColdStartStateMachine(user_id="mlflow_user")
            sm._current_stage = ColdStartStage.STAGE_4  # Stage 4 is allowed to log
            d_hat = 45.0
            sessions_per_day = 0.5
            window = compute_observation_window(d_hat, sessions_per_day=sessions_per_day)
            sm.log_to_mlflow(
                d_hat=d_hat,
                observation_window_days=window,
                sessions_per_day=sessions_per_day,
            )

        mock_mlflow.log_params.assert_called_once()
        logged = mock_mlflow.log_params.call_args[0][0]

        assert logged["cold_start.d_hat_sessions"] == d_hat
        assert logged["cold_start.observation_window_days"] == window
        assert logged["cold_start.sessions_per_day"] == sessions_per_day
        assert logged["cold_start.user_id"] == "mlflow_user"

    def test_pipeline_logs_on_stage_1_plus_not_stage_2(self):
        """
        Pipeline logs MLflow for Stage 1, 3, 4 but NOT for Stage 2.
        Stage 2 estimates are INTERNAL ONLY (S10.2).
        """
        fit = _make_hssm_fit(slow_dur_mu=_dur_mu_for_d_hat(45), slow_dur_sigma=0.5)
        mock_mlflow = MagicMock()

        # Stage 1 (Day 10) — should log
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            run_cold_start_pipeline(
                user_id="log_test",
                day=10,
                n_sessions=5,
                fitted_hssm=fit,
                evidence_gate_passed=False,
            )
        mock_mlflow.log_params.assert_called_once()

        # Stage 2 (Day 35) — must NOT log
        mock_mlflow.reset_mock()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            run_cold_start_pipeline(
                user_id="log_test",
                day=35,
                n_sessions=18,
                fitted_hssm=fit,
                evidence_gate_passed=False,
            )
        mock_mlflow.log_params.assert_not_called()

    def test_pipeline_skips_log_on_stage_0(self):
        """Stage 0 short-circuits before any logging."""
        fit = _make_hssm_fit(slow_dur_mu=_dur_mu_for_d_hat(45), slow_dur_sigma=0.5)
        mock_mlflow = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            state = run_cold_start_pipeline(
                user_id="no_log_test",
                day=5,
                fitted_hssm=fit,
            )

        mock_mlflow.log_params.assert_not_called()
        assert state.stage == ColdStartStage.STAGE_0

    def test_log_to_mlflow_raises_in_stage_2(self):
        """S10.2: log_to_mlflow must raise RuntimeError if current stage is STAGE_2."""
        mock_mlflow = MagicMock()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            sm = ColdStartStateMachine(user_id="u_002")
            sm._current_stage = ColdStartStage.STAGE_2
            with pytest.raises(RuntimeError, match="Stage-2"):
                sm.log_to_mlflow(
                    d_hat=45.0,
                    observation_window_days=90.0,
                    sessions_per_day=0.5,
                )
        mock_mlflow.log_params.assert_not_called()


# ---------------------------------------------------------------------------
# S10.3: real validation check (no tautological self-comparison)
# ---------------------------------------------------------------------------

class TestTautologicalValidationFix:
    """S10.3: structural HSSM model/class validation (type + invariants, not just user_id)."""

    def test_non_hssm_fit_object_raises_type_error(self):
        """S10.3 Check 1: passing a non-HSSMFit raises TypeError."""
        from cold_start.cold_start_wiring import evaluate_cold_start
        mock_model = MagicMock()  # raw backbone model, not an HSSMFit
        mock_model.user_id = "u_raw"
        mock_model.n_regimes = 2
        mock_mlflow = MagicMock()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with pytest.raises(TypeError, match="HSSMFit instance"):
                evaluate_cold_start(day=35, fitted_hssm=mock_model)

    def test_n_regimes_less_than_2_raises_value_error(self):
        """S10.3 Check 2: n_regimes < 2 raises ValueError."""
        from cold_start.cold_start_wiring import evaluate_cold_start
        # Build an HSSMFit with n_regimes=1 (degenerate)
        bad_fit = HSSMFit(
            user_id="u_bad",
            fit_id="fit-bad",
            slow_regime_id=0,
            n_regimes=1,
            duration_parameters={0: {"dur_mu": 2.0, "dur_sigma": 0.5}},
            log_likelihood=-100.0,
            converged=True,
        )
        mock_mlflow = MagicMock()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with pytest.raises(ValueError, match="n_regimes"):
                evaluate_cold_start(day=35, fitted_hssm=bad_fit)

    def test_slow_regime_missing_from_duration_params_raises(self):
        """S10.3 Check 3: slow_regime_id not in duration_parameters raises ValueError."""
        from cold_start.cold_start_wiring import evaluate_cold_start
        bad_fit = HSSMFit(
            user_id="u_bad",
            fit_id="fit-bad",
            slow_regime_id=0,       # claims 0 is slow
            n_regimes=2,
            duration_parameters={1: {"dur_mu": 2.0, "dur_sigma": 0.5}},  # 0 is missing!
            log_likelihood=-100.0,
            converged=True,
        )
        mock_mlflow = MagicMock()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with pytest.raises(ValueError, match="slow_regime_id"):
                evaluate_cold_start(day=35, fitted_hssm=bad_fit)

    def test_empty_user_id_raises_value_error(self):
        """S10.3 Check 4: empty user_id raises ValueError."""
        from cold_start.cold_start_wiring import evaluate_cold_start
        fit = _make_hssm_fit(slow_dur_mu=2.0, slow_dur_sigma=0.5, user_id="")
        mock_mlflow = MagicMock()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with pytest.raises(ValueError, match="user_id is empty"):
                evaluate_cold_start(day=35, fitted_hssm=fit)

    def test_blank_user_id_raises_value_error(self):
        """S10.3: whitespace-only user_id is also rejected."""
        from cold_start.cold_start_wiring import evaluate_cold_start
        fit = _make_hssm_fit(slow_dur_mu=2.0, slow_dur_sigma=0.5, user_id="   ")
        mock_mlflow = MagicMock()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with pytest.raises(ValueError, match="user_id is empty"):
                evaluate_cold_start(day=35, fitted_hssm=fit)

    def test_valid_fit_passes_all_checks(self):
        """S10.3: a structurally valid HSSMFit passes all validation checks."""
        from cold_start.cold_start_wiring import evaluate_cold_start
        fit = _make_hssm_fit(slow_dur_mu=2.0, slow_dur_sigma=0.5, user_id="u_valid")
        mock_mlflow = MagicMock()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            state = evaluate_cold_start(day=35, fitted_hssm=fit)
        assert state.user_id == "u_valid"
