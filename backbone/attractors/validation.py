"""
Sprint 4, Day 12 — Validation against planted structure + cold-start silence check.

Spec:
  - Validate attractor detection against synthetic trajectories with known
    planted structure; verify recovery above target precision/recall bar,
    measured on a HELD-OUT batch (different seeds than calibration itself —
    otherwise this just re-reports the calibration search's own numbers).
  - Below the 30-session gate, verify the system stays completely silent on
    attractor claims (MP-11) — delegates to hssm.gating's ColdStartError.
  - Spot-check: no two users should share an identical (N,T) purely by
    coincidence of CODE (a hardcoded default), only by coincidence of DATA.
"""

from __future__ import annotations
import numpy as np

from backbone.shared.synthetic_data import generate_surrogate_user
from backbone.shared.feature_reduction import per_person_zscore, reduce_dimensionality
from backbone.hssm.gating import fit_hssm_gated, ColdStartError, count_present_sessions
from backbone.attractors.detector import compute_attractor_stats, is_attractor
from backbone.attractors.calibration import grid_search_NT, sample_from_fitted_hssm
from backbone.attractors.config import DEFAULT_CALIBRATION_CONFIG, CalibrationConfig


def fit_user_pipeline(n_sessions: int, n_features_raw: int = 15, n_regimes: int = 2,
                       target_dims: int = 8, seed: int = 0):
    """Full pipeline: synthetic user -> zscore -> PCA reduce -> gated HSSM fit."""
    user = generate_surrogate_user(n_sessions=n_sessions, n_features=n_features_raw,
                                    n_regimes=n_regimes, seed=seed)
    Z, _, _ = per_person_zscore(user["X"])
    feat_names = [f"f{i}" for i in range(n_features_raw)]
    Z_red, _, _ = reduce_dimensionality(Z, feat_names, target_dims=target_dims)
    n_present = count_present_sessions(user["X"])
    model, run_log = fit_hssm_gated(Z_red, n_regimes=n_regimes, n_features=target_dims,
                                     n_present_sessions=n_present, n_init=10, base_seed=seed * 10)
    return model, run_log, user


def validate_planted_recovery(
    n_users: int = 3,
    n_regimes: int = 2,
    target_dims: int = 8,
    config: CalibrationConfig = DEFAULT_CALIBRATION_CONFIG,
    n_test_trials: int = 20,
) -> dict:
    """Fit n_users users, calibrate N/T per user (Day 11), then run a FRESH
    independent batch of planted-vs-noise trajectories per user to measure
    recovery precision/recall at the selected operating point. This batch is
    held out from the trials used to select the threshold in the first place."""
    all_calibrations = {}
    all_recovery = {}

    for u in range(n_users):
        model, run_log, _ = fit_user_pipeline(n_sessions=130, n_regimes=n_regimes,
                                               target_dims=target_dims, seed=u * 7 + 1)
        target_regime = n_regimes - 1  # highest-activity regime, canonical post-sort

        calib = grid_search_NT(model, target_regime=target_regime, config=config, base_seed=u * 500)
        Nr, Nd, T = calib["selected_N_revisit"], calib["selected_N_dwell"], calib["selected_T"]
        all_calibrations[f"user_{u}"] = {"N_revisit": Nr, "N_dwell": Nd, "T": T}

        fn, fp = 0, 0
        for trial in range(n_test_trials):
            seeded = sample_from_fitted_hssm(model, config.n_timesteps, seed_attractor=True,
                                              attractor_regime=target_regime, seed=u * 9000 + trial)
            noise = sample_from_fitted_hssm(model, config.n_timesteps, seed_attractor=False,
                                             attractor_regime=target_regime, seed=u * 9000 + 5000 + trial)
            s_seeded = compute_attractor_stats(seeded["m_t"], seeded["regime_labels"], target_regime)
            s_noise = compute_attractor_stats(noise["m_t"], noise["regime_labels"], target_regime)
            if not is_attractor(s_seeded, (Nr, Nd), T):
                fn += 1
            if is_attractor(s_noise, (Nr, Nd), T):
                fp += 1

        recall = (n_test_trials - fn) / n_test_trials
        tp = n_test_trials - fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        all_recovery[f"user_{u}"] = {
            "precision": precision, "recall": recall,
            "meets_bar": precision >= config.target_precision and recall >= config.target_precision,
        }

    all_pass = all(v["meets_bar"] for v in all_recovery.values())
    return {"calibrations": all_calibrations, "recovery": all_recovery, "all_users_pass": all_pass}


def check_no_coincidental_NT_sharing(calibrations: dict) -> bool:
    """DoD spot-check: duplicate (N,T) triples across users are only
    acceptable if genuinely explained by similar underlying data density
    (coincidence of DATA). This check flags duplicates for manual review —
    it cannot itself distinguish 'coincidence of data' from 'a bug that
    silently returns a hardcoded default', so a False here means REVIEW,
    not necessarily FAIL."""
    pairs = [(v["N_revisit"], v["N_dwell"], v["T"]) for v in calibrations.values()]
    unique_pairs = set(pairs)
    return len(unique_pairs) == len(pairs)


def check_silence_below_gate(seed: int = 999) -> bool:
    """DoD: below 30-session gate, attractor detection pipeline must be
    completely silent (MP-11) — verified via ColdStartError propagating up
    through the full pipeline, not just the gating module in isolation."""
    try:
        fit_user_pipeline(n_sessions=25, n_regimes=2, seed=seed)
        return False  # should have raised
    except ColdStartError:
        return True
