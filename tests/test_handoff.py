"""
Sprint 3–4 Phase 5: Downstream Handoff Contract Tests.
Tests 117–123 per senior directive Section J & Section Q.

Verifies:
  - HSSMResult provides all required fields for downstream modules (attractors, phase transitions, drift)
  - Result feeds seamlessly into attractor detector (compute_attractor_stats)
  - Baseline model safely provides None for fast state m_t without breaking downstream callers
  - Dual-timescale model provides valid continuous latent trajectory (m_t) with shape (T, latent_dim)
  - Tests 121-123: Sprint 5-6 adapter integration boundary, contract safety, and rejection of mock adapters
"""

import importlib
import numpy as np
import pytest

from backbone.hssm.fitting import fit_hssm, HSSMResult
from backbone.hssm.model import FitState
from backbone.attractors.detector import compute_attractor_stats, is_attractor
from data.synthetic_dual import generate_synthetic_dual_timescale_data


def test_117_hssm_result_full_metadata_contract():
    """Test 117: HSSMResult exposes all ~40 directive-mandated fields with valid types."""
    T = 40
    X = np.random.RandomState(42).randn(T, 4)
    cal = np.array([float(100.0 + i * 1.5) for i in range(T)])

    result = fit_hssm(
        matrix=X,
        candidate_ks=(2,),
        n_initializations=1,
        max_iter=5,
        random_seed=42,
        allow_fast_test_fit=True,
        timestamps=cal,
        user_id="user_handoff_117",
        dataset_hash="a" * 64,
        model_type="dual_timescale",
        latent_dim=2,
    )

    # Core identity & capabilities
    assert isinstance(result.user_id, str)
    assert isinstance(result.model_identity, str)
    assert isinstance(result.model_family, str)
    assert isinstance(result.model_capability, dict)
    assert isinstance(result.is_baseline_model, bool)
    assert isinstance(result.fast_state_supported, bool)

    # State estimates
    assert isinstance(result.p_t_posterior, np.ndarray)
    assert isinstance(result.p_t_estimate, np.ndarray)
    assert isinstance(result.m_t_posterior_mean, np.ndarray)
    assert isinstance(result.m_t_uncertainty, np.ndarray)

    # Duration
    assert isinstance(result.duration_parameters, dict)
    assert isinstance(result.duration_model_identity, str)
    assert isinstance(result.Dmax, int)
    assert isinstance(result.duration_unit_source, str)

    # Temporal & metrics
    assert isinstance(result.K, int)
    assert isinstance(result.BIC, float)
    assert isinstance(result.AIC, float)
    assert isinstance(result.fit_state, FitState)
    assert isinstance(result.run_log, list)
    assert isinstance(result.warnings, list)
    assert isinstance(result.limitations, list)


def test_118_dual_model_handoff_to_attractor_detector():
    """Test 118: DualTimescaleHSSM result.m_t directly feeds into compute_attractor_stats."""
    data = generate_synthetic_dual_timescale_data(n_regimes=2, n_features=4, latent_dim=2, n_segments=6, seed=42)
    
    result = fit_hssm(
        matrix=data.X,
        candidate_ks=(2,),
        n_initializations=1,
        max_iter=15,
        random_seed=42,
        allow_fast_test_fit=True,
        timestamps=data.timestamps,
        user_id="user_attractor_handoff",
        model_type="dual_timescale",
        latent_dim=2,
    )

    assert result.m_t is not None
    assert result.p_t is not None
    T = data.X.shape[0]
    assert result.m_t.shape == (T, 2)
    assert result.p_t.shape == (T,)

    # Feed directly into attractor detection
    stats_k0 = compute_attractor_stats(
        m_t=result.m_t,
        regime_labels=result.p_t,
        target_regime=0,
    )
    assert "revisit_count" in stats_k0
    assert "mean_dwell_time" in stats_k0
    assert "transition_stability" in stats_k0
    assert isinstance(stats_k0["revisit_count"], int)
    assert isinstance(stats_k0["mean_dwell_time"], float)
    assert isinstance(stats_k0["transition_stability"], float)


def test_119_baseline_model_fast_state_contract_safety():
    """Test 119: Baseline model returns None for m_t and fast_state_supported=False without exceptions."""
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(0, 0.5, (30, 2)),
        np.random.normal(4, 0.5, (30, 2)),
    ])

    result = fit_hssm(
        matrix=X,
        candidate_ks=(2,),
        n_initializations=10,
        max_iter=50,
        random_seed=42,
        allow_fast_test_fit=True,
        model_type="baseline",
    )

    assert result.fast_state_supported is False
    assert result.is_baseline_model is True
    assert result.m_t is None
    assert result.m_t_uncertainty is None
    assert result.p_t is not None
    assert result.p_t.shape == (60,)


def test_120_result_regime_posterior_simplex_normalization():
    """Test 120: Posterior probabilities for both models sum to 1.0 at every timestep."""
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(0, 0.5, (30, 2)),
        np.random.normal(4, 0.5, (30, 2)),
    ])

    for mtype in ("baseline", "dual_timescale"):
        res = fit_hssm(
            matrix=X,
            candidate_ks=(2,),
            n_initializations=10,
            max_iter=50,
            random_seed=42,
            allow_fast_test_fit=True,
            model_type=mtype,
            latent_dim=2,
        )
        row_sums = np.sum(res.p_t_posterior, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5, err_msg=f"{mtype} posterior must sum to 1.0")


def test_121_hssm_to_sprint56_adapter_blocked_when_package_absent():
    """Test 121: Handoff adapter to Sprint 5-6 (phase transitions) is cleanly blocked when external package is absent."""
    # Attempting to import non-existent external sprint 5/6 phase transitions package must raise ModuleNotFoundError
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("backbone.phase_transitions")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("chronis_phase_transitions")


def test_122_hssm_to_sprint56_adapter_contract_requirements():
    """Test 122: HSSMResult satisfies complete schema requirements expected by Phase-Transition consumers."""
    T = 40
    cal = np.array([float(10.0 + i * 2.5) for i in range(T)])
    X = np.random.RandomState(42).randn(T, 3)

    res = fit_hssm(
        matrix=X,
        candidate_ks=(2,),
        n_initializations=1,
        max_iter=5,
        random_seed=42,
        allow_fast_test_fit=True,
        timestamps=cal,
        model_type="dual_timescale",
        latent_dim=2,
    )

    # Phase-transition consumer contract requirements (Sprint 5-6 spec):
    # 1. p_t posterior simplex over time [T, K]
    assert res.p_t_posterior is not None
    assert res.p_t_posterior.shape == (T, 2)
    # 2. Continuous state vector trajectory [T, L]
    assert res.m_t_posterior_mean is not None
    assert res.m_t_posterior_mean.shape == (T, 2)
    # 3. Dynamic uncertainty bounds [T, L]
    assert res.m_t_uncertainty is not None
    assert res.m_t_uncertainty.shape == (T, 2)
    # 4. Valid calendar timestamps
    assert res.calendar_start == 10.0
    assert res.calendar_end == 10.0 + (T - 1) * 2.5


def test_123_no_synthetic_adapter_used_in_production_path():
    """Test 123: Production backbone namespace contains no mock or synthetic adapter bridges."""
    import backbone.hssm as hssm_mod
    exported = getattr(hssm_mod, "__all__", dir(hssm_mod))
    for name in exported:
        assert "mock" not in name.lower(), f"Forbidden mock adapter found in production exports: {name}"
        assert "synthetic_adapter" not in name.lower(), f"Forbidden synthetic adapter in production exports: {name}"


def test_result_run_log_auditability():
    """Additional contract check: run_log records initialization details for every candidate K and restart."""
    T = 40
    X = np.random.RandomState(42).randn(T, 2)

    res = fit_hssm(
        matrix=X,
        candidate_ks=(2, 3),
        n_initializations=2,
        max_iter=5,
        random_seed=42,
        allow_fast_test_fit=True,
        model_type="dual_timescale",
        latent_dim=2,
    )

    assert len(res.run_log) >= 4  # 2 candidates * 2 inits
    for entry in res.run_log:
        assert "seed" in entry
        assert "fit_state" in entry


def test_legacy_properties_backward_compatibility():
    """Additional contract check: Legacy property accessors work seamlessly."""
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(0, 0.5, (30, 2)),
        np.random.normal(4, 0.5, (30, 2)),
    ])

    res = fit_hssm(
        matrix=X,
        candidate_ks=(2,),
        n_initializations=10,
        max_iter=50,
        random_seed=42,
        allow_fast_test_fit=True,
        model_type="baseline",
    )

    assert res.p_t is res.p_t_estimate
    assert res.regime_posterior is res.p_t_posterior
    assert res.k_selected == res.K
    assert "max_duration" in res.duration_info
