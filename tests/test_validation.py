import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backbone.attractors.validation import (
    validate_planted_recovery, check_no_coincidental_NT_sharing, check_silence_below_gate,
)
from backbone.attractors.config import CalibrationConfig


def test_silence_below_cold_start_gate():
    assert check_silence_below_gate() is True


def test_planted_recovery_meets_bar_for_multiple_users():
    config = CalibrationConfig(n_timesteps=250, n_trials=15, target_precision=0.8)
    results = validate_planted_recovery(n_users=3, config=config, n_test_trials=20)
    assert results["all_users_pass"], f"precision/recall bar not met: {results['recovery']}"


def test_NT_coincidence_check_runs():
    config = CalibrationConfig(n_timesteps=250, n_trials=15, target_precision=0.8)
    results = validate_planted_recovery(n_users=3, config=config, n_test_trials=10)
    outcome = check_no_coincidental_NT_sharing(results["calibrations"])
    assert isinstance(outcome, bool)
