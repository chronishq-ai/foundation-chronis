import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from backbone.shared.synthetic_data import generate_surrogate_user
from backbone.shared.feature_reduction import per_person_zscore, reduce_dimensionality
from backbone.hssm.fitting import fit_with_random_restarts
from backbone.attractors.calibration import sample_from_fitted_hssm, grid_search_NT
from backbone.attractors.config import CalibrationConfig


def _fit_small_model(seed=5):
    user = generate_surrogate_user(n_sessions=120, n_features=15, n_regimes=2, seed=seed)
    Z, _, _ = per_person_zscore(user["X"])
    feat_names = [f"f{i}" for i in range(15)]
    Z_red, _, _ = reduce_dimensionality(Z, feat_names, target_dims=8)
    model, _ = fit_with_random_restarts(Z_red, n_regimes=2, n_features=8, n_init=10, base_seed=1)
    return model


def test_sample_from_fitted_hssm_shapes():
    model = _fit_small_model()
    out = sample_from_fitted_hssm(model, n_timesteps=100, seed_attractor=False, seed=0)
    assert out["m_t"].shape == (100, model.F)
    assert out["regime_labels"].shape == (100,)
    assert set(np.unique(out["regime_labels"])).issubset(set(range(model.K)))


def test_seeded_condition_has_lower_variance_in_target_regime():
    model = _fit_small_model()
    target = model.K - 1
    seeded = sample_from_fitted_hssm(model, 300, seed_attractor=True, attractor_regime=target, seed=1)
    noise = sample_from_fitted_hssm(model, 300, seed_attractor=False, attractor_regime=target, seed=2)
    seeded_var = np.var(seeded["m_t"][seeded["regime_labels"] == target], axis=0).mean()
    noise_var = np.var(noise["m_t"][noise["regime_labels"] == target], axis=0).mean()
    assert seeded_var < noise_var, "seeded attractor condition should be tighter than pure noise"


def test_grid_search_finds_a_working_operating_point():
    model = _fit_small_model()
    target = model.K - 1
    config = CalibrationConfig(n_timesteps=250, n_trials=15, target_precision=0.8)
    result = grid_search_NT(model, target_regime=target, config=config, base_seed=0)
    assert result["target_met"] is True, f"calibration failed to meet target precision: {result}"
    assert result["achieved_precision"] >= 0.8


def test_NT_split_not_shared_scalar():
    # regression test for the found-and-fixed bug: revisit_count and
    # mean_dwell_time cannot share one literal N.
    model = _fit_small_model()
    target = model.K - 1
    config = CalibrationConfig(n_timesteps=250, n_trials=15, target_precision=0.8,
                                N_revisit_grid=(2,), N_dwell_grid=(0.5,))  # force equal-ish grids for the check
    result = grid_search_NT(model, target_regime=target, config=config, base_seed=0)
    assert "selected_N_revisit" in result and "selected_N_dwell" in result
