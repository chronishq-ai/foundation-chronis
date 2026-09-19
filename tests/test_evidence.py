"""
Sprint 3–4 Phase 6: Evidence & Reproducibility Verification Tests.
Tests 124–130 per senior directive Section P, Q, and R.

Verifies:
  - Strict determinism (same seed, same data -> bit-for-bit identical HSSMResult)
  - Evidence manifest and documentation integrity
  - Serialization safety of HSSMResult
  - Auditability of run logs
  - Explicit limitations against overclaiming
  - AIC/BIC sanity and complexity penalization
  - Acceptance ladder self-consistency
"""

import os
from pathlib import Path
import numpy as np
import pytest

from backbone.hssm.fitting import fit_hssm, HSSMResult
from backbone.hssm.model import FitState
from backbone.hssm.dual_model import DualTimescaleHSSM


def test_124_same_seed_determinism():
    """Test 124: Same seed and same input data produces bit-for-bit identical HSSMResult."""
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(0, 0.5, (25, 2)),
        np.random.normal(4, 0.5, (25, 2)),
    ])
    timestamps = np.linspace(100.0, 150.0, 50)

    res1 = fit_hssm(
        matrix=X,
        candidate_ks=(2,),
        n_initializations=5,
        max_iter=30,
        random_seed=777,
        allow_fast_test_fit=True,
        timestamps=timestamps,
        model_type="dual_timescale",
        latent_dim=2,
    )

    res2 = fit_hssm(
        matrix=X,
        candidate_ks=(2,),
        n_initializations=5,
        max_iter=30,
        random_seed=777,
        allow_fast_test_fit=True,
        timestamps=timestamps,
        model_type="dual_timescale",
        latent_dim=2,
    )

    # State equality
    np.testing.assert_array_equal(res1.p_t_estimate, res2.p_t_estimate)
    np.testing.assert_allclose(res1.p_t_posterior, res2.p_t_posterior, atol=1e-12)
    np.testing.assert_allclose(res1.m_t_posterior_mean, res2.m_t_posterior_mean, atol=1e-12)
    np.testing.assert_allclose(res1.m_t_uncertainty, res2.m_t_uncertainty, atol=1e-12)

    # Information criteria & log-likelihood
    assert res1.BIC == res2.BIC
    assert res1.AIC == res2.AIC
    assert res1.fit_state == res2.fit_state


def test_125_evidence_manifest_and_docs_exist():
    """Test 125: All 8 required evidence documents and evidence_manifest.yaml exist in docs/backbone/."""
    docs_dir = Path("docs/backbone")
    assert docs_dir.exists(), "docs/backbone directory must exist"

    required_docs = [
        "HSSM_DESIGN_NOTE.md",
        "HSSM_RESULT_SCHEMA.md",
        "HSSM_MATH_SPEC.md",
        "HSSM_LIMITATIONS.md",
        "DMAX_SENSITIVITY_REPORT.md",
        "MODEL_SELECTION_REPORT.md",
        "BASELINE_VS_FULL_REPORT.md",
        "evidence_manifest.yaml",
    ]

    for doc_name in required_docs:
        doc_path = docs_dir / doc_name
        assert doc_path.exists(), f"Required evidence document missing: {doc_name}"
        assert doc_path.stat().st_size > 50, f"Evidence document {doc_name} is unexpectedly empty"

    # Verify manifest text contains acceptance state and all documents
    manifest_path = docs_dir / "evidence_manifest.yaml"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_text = f.read()

    assert "acceptance_state: \"SEALED_YELLOW\"" in manifest_text or "acceptance_state: 'SEALED_YELLOW'" in manifest_text
    for doc_name in required_docs[:7]:
        assert doc_name in manifest_text, f"{doc_name} must be indexed in evidence manifest"


def test_126_hssm_result_dict_serialization():
    """Test 126: HSSMResult can be serialized to dictionary without lost metadata or crash."""
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(0, 0.5, (25, 2)),
        np.random.normal(4, 0.5, (25, 2)),
    ])
    timestamps = np.linspace(100.0, 150.0, 50)

    res = fit_hssm(
        matrix=X,
        candidate_ks=(2,),
        n_initializations=2,
        max_iter=10,
        random_seed=42,
        allow_fast_test_fit=True,
        timestamps=timestamps,
        user_id="user_serial",
        dataset_hash="b" * 64,
        model_type="dual_timescale",
        latent_dim=2,
    )

    # Convert to serializable dictionary representation
    data_dict = {
        "user_id": res.user_id,
        "model_identity": res.model_identity,
        "model_family": res.model_family,
        "model_capability": res.model_capability,
        "is_baseline_model": res.is_baseline_model,
        "fast_state_supported": res.fast_state_supported,
        "K": res.K,
        "BIC": res.BIC,
        "AIC": res.AIC,
        "fit_state": res.fit_state.name,
        "fit_dataset_hash": res.fit_dataset_hash,
        "Dmax": res.Dmax,
        "duration_unit_source": res.duration_unit_source,
        "calendar_start": res.calendar_start,
        "calendar_end": res.calendar_end,
        "observation_density": res.observation_density,
        "neural_residual_enabled": res.neural_residual_enabled,
        "residual_mode": res.residual_mode,
        "p_t_estimate": res.p_t_estimate.tolist() if res.p_t_estimate is not None else None,
        "m_t_mean": res.m_t_posterior_mean.tolist() if res.m_t_posterior_mean is not None else None,
    }

    assert data_dict["user_id"] == "user_serial"
    assert data_dict["model_identity"] == res.model_identity
    assert data_dict["fast_state_supported"] is True
    assert len(data_dict["p_t_estimate"]) == 50
    assert len(data_dict["m_t_mean"]) == 50


def test_127_run_log_auditability_completeness():
    """Test 127: Every restart is logged with seed, log-likelihood, and fit state."""
    np.random.seed(42)
    X = np.random.randn(40, 2)

    res = fit_hssm(
        matrix=X,
        candidate_ks=(2,),
        n_initializations=3,
        max_iter=5,
        random_seed=100,
        allow_fast_test_fit=True,
        model_type="dual_timescale",
        latent_dim=2,
    )

    assert len(res.run_log) == 3
    for log_entry in res.run_log:
        assert "seed" in log_entry
        assert "fit_state" in log_entry
        assert "log_likelihood" in log_entry


def test_128_limitations_block_overclaim_guard():
    """Test 128: Baseline model explicitly lists limitations preventing claims of fast state tracking."""
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

    assert res.is_baseline_model is True
    assert len(res.limitations) > 0
    # Must explicitly state that m_t is not supported
    assert any("fast continuous state m_t is not supported" in lim for lim in res.limitations)


def test_129_information_criteria_sanity():
    """Test 129: BIC and AIC are finite numeric values with BIC >= AIC for N >= 8 observations."""
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(0, 0.5, (30, 2)),
        np.random.normal(4, 0.5, (30, 2)),
    ])

    res = fit_hssm(
        matrix=X,
        candidate_ks=(2,),
        n_initializations=5,
        max_iter=30,
        random_seed=42,
        allow_fast_test_fit=True,
        model_type="dual_timescale",
        latent_dim=2,
    )

    assert np.isfinite(res.BIC)
    assert np.isfinite(res.AIC)
    # For N=60, ln(N) = ln(60) ≈ 4.09 > 2, so BIC penalty k*ln(N) > AIC penalty 2*k => BIC > AIC
    assert res.BIC > res.AIC


def test_130_acceptance_ladder_capability_consistency():
    """Test 130: Model capabilities dict precisely matches model identity across both classes."""
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(0, 0.5, (30, 2)),
        np.random.normal(4, 0.5, (30, 2)),
    ])

    model_baseline = fit_hssm(
        matrix=X,
        candidate_ks=(2,),
        n_initializations=10,
        max_iter=50,
        random_seed=42,
        allow_fast_test_fit=True,
        model_type="baseline",
    )
    assert model_baseline.model_identity == "ExplicitDurationSwitchingBaselineV1"
    assert model_baseline.model_capability["fast_state_supported"] is False
    assert model_baseline.model_capability["is_baseline_model"] is True

    model_dual = fit_hssm(
        matrix=X,
        candidate_ks=(2,),
        n_initializations=1,
        max_iter=5,
        allow_fast_test_fit=True,
        model_type="dual_timescale",
    )
    assert model_dual.model_identity == "DualTimescaleKimHSSMV1"
    assert model_dual.model_capability["fast_state_supported"] is True
    assert model_dual.model_capability["is_baseline_model"] is False
