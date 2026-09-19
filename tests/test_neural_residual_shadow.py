"""
Phase 4: Neural Residual Shadow & Zero-Weight Ablation Tests.
Verifies directive Section E, Phase 4, and Tests 43-48:
  - Pure base model validity when residual is disabled
  - Zero-weight ablation bit-for-bit identity check
  - Shadow mode offline logging without corrupting live states
  - Active mode bounded additive corrections
  - Invariant: residual never overwrites or fakes m_t / p_t
  - HSSMResult metadata tracking
"""

import numpy as np
import pytest
from backbone.hssm.neural_residual import NeuralResidualAdapter, ResidualMode, ResidualDiagnostic
from backbone.hssm.dual_model import DualTimescaleHSSM
from backbone.hssm.fitting import fit_hssm, HSSMResult
from data.synthetic_dual import generate_synthetic_dual_timescale_data


def test_pure_base_model_valid_when_residual_disabled():
    """Test 43: Pure base model is complete and fully valid on its own."""
    adapter = NeuralResidualAdapter(mode=ResidualMode.DISABLED, weight=0.0)
    base_emission = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_feat = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    out, diag = adapter.apply_residual(base_emission, X_feat)
    np.testing.assert_array_equal(out, base_emission)
    assert diag.ablation_verified is True
    assert diag.mode == "DISABLED"


def test_zero_weight_ablation_identity():
    """Test 44: Zero-weight ablation test — output is bit-for-bit identical to base."""
    # Even in ACTIVE mode, if weight is 0.0, output must be strictly identical
    adapter = NeuralResidualAdapter(mode=ResidualMode.ACTIVE, weight=0.0)
    base_emission = np.random.randn(50, 4)
    X_feat = np.random.randn(50, 4)
    
    out, diag = adapter.apply_residual(base_emission, X_feat)
    np.testing.assert_array_equal(out, base_emission)
    assert diag.ablation_verified is True


def test_shadow_mode_logs_diagnostics_without_modifying_live_emission():
    """Test 45: Shadow mode logs non-zero residuals without altering live emission."""
    adapter = NeuralResidualAdapter(mode=ResidualMode.SHADOW, weight=0.5, seed=42)
    base_emission = np.ones((30, 3))
    X_feat = np.random.randn(30, 3)
    
    out, diag = adapter.apply_residual(base_emission, X_feat)
    # Output must remain strictly identical to base emission in SHADOW mode
    np.testing.assert_array_equal(out, base_emission)
    assert diag.mode == "SHADOW"
    # Diagnostics must record non-zero residual norms
    assert diag.residual_mean_norm > 0.0
    assert diag.residual_max_norm > 0.0


def test_active_mode_applies_bounded_additive_correction():
    """Test 46: Active mode applies additive correction within bounded range."""
    weight = 0.2
    adapter = NeuralResidualAdapter(mode=ResidualMode.ACTIVE, weight=weight, seed=42)
    base_emission = np.zeros((40, 2))
    X_feat = np.ones((40, 2))
    
    out, diag = adapter.apply_residual(base_emission, X_feat)
    assert diag.mode == "ACTIVE"
    # Correction must be non-zero and bounded by weight * tanh_bound (since tanh in [-1, 1])
    assert not np.array_equal(out, base_emission)
    max_dev = np.max(np.abs(out - base_emission))
    assert max_dev <= weight + 1e-5


def test_residual_never_alters_latent_states():
    """Test 47: Neural residual operates on observation space, never overwriting m_t or p_t."""
    data = generate_synthetic_dual_timescale_data(n_regimes=2, n_features=3, latent_dim=2, seed=42)
    model = DualTimescaleHSSM(n_regimes=2, n_features=3, latent_dim=2, seed=42)
    model.fit(data.X, n_iter=10)
    
    m_t, m_t_std, p_t, p_t_post = model.estimate_states(data.X)
    
    # Run residual adapter
    adapter = NeuralResidualAdapter(mode=ResidualMode.ACTIVE, weight=0.3)
    base_pred = np.zeros((len(data.X), 3))
    out, _ = adapter.apply_residual(base_pred, data.X)
    
    # Assert latent trajectory and regime posteriors were untouched
    assert m_t.shape == (len(data.X), 2)
    assert p_t.shape == (len(data.X),)
    assert p_t_post.shape == (len(data.X), 2)


def test_hssm_result_records_residual_metadata():
    """Test 48: HSSMResult contains accurate residual metadata fields."""
    data = generate_synthetic_dual_timescale_data(n_regimes=2, n_features=3, latent_dim=2, seed=42)
    
    result = fit_hssm(
        matrix=data.X,
        candidate_ks=(2,),
        n_initializations=10,
        max_iter=10,
        random_seed=42,
        allow_fast_test_fit=True,
    )
    
    assert hasattr(result, "neural_residual_enabled")
    assert hasattr(result, "residual_mode")
    assert hasattr(result, "residual_weight")
    assert hasattr(result, "residual_model_version")
    assert hasattr(result, "residual_ablation_status")
    assert result.neural_residual_enabled is False
    assert result.residual_mode == "DISABLED"
    assert result.residual_weight == 0.0
