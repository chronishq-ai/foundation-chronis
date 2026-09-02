import pytest
import numpy as np
from backbone.hssm.model import GaussianHSMM
from backbone.hssm.gating import fit_hssm_gated, ColdStartError, count_present_sessions


def test_s34_6_t1_duration_prior_units_mismatch():
    rng = np.random.default_rng(42)
    
    # User A: Daily sessions for 40 days (no missingness)
    # Regime 0 for first 20 days, Regime 1 for next 20 days
    X_a = np.zeros((40, 2))
    X_a[:20] = rng.normal(0.0, 0.1, size=(20, 2))
    X_a[20:] = rng.normal(5.0, 0.1, size=(20, 2))
    ts_a = np.arange(40, dtype=float)

    # User B: Every second day is missing, so 20 sessions over 40 days (50% missingness density)
    # Regime 0 for first 20 calendar days (10 sessions), Regime 1 for next 20 calendar days (10 sessions)
    X_b = np.zeros((20, 2))
    X_b[:10] = rng.normal(0.0, 0.1, size=(10, 2))
    X_b[10:] = rng.normal(5.0, 0.1, size=(10, 2))
    ts_b = np.arange(0, 40, 2, dtype=float)

    # Pre-fix regression test (without timestamps): session-index units cause parameter discrepancy
    model_a_old = GaussianHSMM(n_regimes=2, n_features=2, seed=42)
    model_a_old.fit(X_a)  # no timestamps -> sessions
    assert model_a_old.duration_unit == "sessions"

    model_b_old = GaussianHSMM(n_regimes=2, n_features=2, seed=42)
    model_b_old.fit(X_b)  # no timestamps -> sessions
    assert model_b_old.duration_unit == "sessions"

    diff_old_0 = abs(model_a_old.duration_mu[0] - model_b_old.duration_mu[0])
    diff_old_1 = abs(model_a_old.duration_mu[1] - model_b_old.duration_mu[1])
    assert diff_old_0 > 0.4, f"Pre-fix: expected parameters to differ, got difference={diff_old_0}"
    assert diff_old_1 > 0.4, f"Pre-fix: expected parameters to differ, got difference={diff_old_1}"

    # Post-fix acceptance test (with timestamps): calendar-day units are statistically equivalent
    model_a = GaussianHSMM(n_regimes=2, n_features=2, seed=42)
    model_a.fit(X_a, timestamps=ts_a)
    assert model_a.duration_unit == "calendar_days"

    model_b = GaussianHSMM(n_regimes=2, n_features=2, seed=42)
    model_b.fit(X_b, timestamps=ts_b)
    assert model_b.duration_unit == "calendar_days"

    diff_regime0 = abs(model_a.duration_mu[0] - model_b.duration_mu[0])
    diff_regime1 = abs(model_a.duration_mu[1] - model_b.duration_mu[1])

    # Core acceptance criterion: parameters are statistically equivalent within tolerance (< 0.15)
    assert diff_regime0 < 0.15, f"Post-fix: expected parameters to match, got difference={diff_regime0}"
    assert diff_regime1 < 0.15, f"Post-fix: expected parameters to match, got difference={diff_regime1}"


def test_s34_6_t2_cold_start_boundary_with_partial_missingness():
    # Cold-start gate checks min_present_sessions (default 30).
    # Create datasets with exactly 29, 30, and 31 valid sessions.
    # Valid sessions: rows with no NaNs.
    
    rng = np.random.default_rng(42)
    
    # Total rows = 35
    # 29 valid sessions: 6 rows fully NaN, others present (with some partial missingness containing NaN in 1 feature,
    # which also doesn't count as a present session)
    X_29 = rng.normal(size=(35, 2))
    X_29[0:6, :] = np.nan  # 6 fully missing rows
    # Out of the remaining 29, let's keep all present so we have exactly 29 present sessions
    ts_29 = np.arange(35, dtype=float)
    assert count_present_sessions(X_29) == 29

    with pytest.raises(ColdStartError):
        fit_hssm_gated(X_29, n_regimes=2, n_features=2, timestamps=ts_29, n_init=10)

    # 30 valid sessions: 5 rows fully NaN, others present
    X_30 = rng.normal(size=(35, 2))
    X_30[0:5, :] = np.nan  # 5 fully missing rows
    ts_30 = np.arange(35, dtype=float)
    assert count_present_sessions(X_30) == 30

    # With exactly 30 present sessions, fitting should proceed normally
    model_30, _ = fit_hssm_gated(X_30, n_regimes=2, n_features=2, timestamps=ts_30, n_init=10)
    assert model_30.duration_unit == "calendar_days"

    # 31 valid sessions: 4 rows fully NaN, others present
    X_31 = rng.normal(size=(35, 2))
    X_31[0:4, :] = np.nan  # 4 fully missing rows
    ts_31 = np.arange(35, dtype=float)
    assert count_present_sessions(X_31) == 31

    # With exactly 31 present sessions, fitting should proceed normally
    model_31, _ = fit_hssm_gated(X_31, n_regimes=2, n_features=2, timestamps=ts_31, n_init=10)
    assert model_31.duration_unit == "calendar_days"
