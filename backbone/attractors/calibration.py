"""
Sprint 4, Day 11 — Threshold calibration harness (Bible 5.11).

Sample synthetic trajectories FROM a user's fitted HSSM under:
  (a) seeded attractor structure (known recurring basin injected)
  (b) pure regime noise (no structure)
Grid-search N (split N_revisit/N_dwell) and T across both, compute FP/FN,
select the operating point hitting target precision for THIS user's own
data density. Thresholds are person-specific and data-derived, never
hardcoded constants.
"""

from __future__ import annotations
import numpy as np

from backbone.attractors.detector import compute_attractor_stats, is_attractor
from backbone.attractors.config import DEFAULT_CALIBRATION_CONFIG, CalibrationConfig


def sample_from_fitted_hssm(
    model,
    n_timesteps: int,
    seed_attractor: bool = False,
    attractor_regime: int | None = None,
    attractor_strength: float = 0.85,
    seed: int | None = None,
) -> dict:
    """Sample a synthetic trajectory from a FITTED model's own regime
    means/variances/durations, for calibration purposes.

    seed_attractor=False: pure regime noise — draw regime sequence from the
      model's own transition+duration structure, emit per-regime Gaussian
      noise, no extra structure injected.
    seed_attractor=True: within attractor_regime, additionally bias emissions
      toward the regime mean with reduced variance, simulating a genuine
      recurring basin on top of the base regime-switching process."""
    rng = np.random.default_rng(seed)
    K, F = model.K, model.F
    if attractor_regime is None:
        attractor_regime = K - 1

    d_vals = np.arange(1, model.Dmax + 1)
    dur_logpmf = model._duration_logpmf()

    regime_path = []
    t = 0
    current_regime = rng.choice(K, p=model.pi)
    while t < n_timesteps:
        probs = np.exp(dur_logpmf[current_regime])
        duration = rng.choice(d_vals, p=probs / probs.sum())
        end = min(t + duration, n_timesteps)
        regime_path.append((t, end, current_regime))
        t = end
        if t >= n_timesteps:
            break
        row = model.A[current_regime]
        if row.sum() > 0:
            current_regime = rng.choice(K, p=row / row.sum())

    regime_labels = np.zeros(n_timesteps, dtype=int)
    for start, end, r in regime_path:
        regime_labels[start:end] = r

    m_t = np.zeros((n_timesteps, F))
    for i in range(n_timesteps):
        r = regime_labels[i]
        if seed_attractor and r == attractor_regime:
            var = model.var[r] * (1 - attractor_strength)
        else:
            var = model.var[r]
        m_t[i] = rng.normal(model.mu[r], np.sqrt(np.clip(var, 1e-6, None)))

    return {"m_t": m_t, "regime_labels": regime_labels, "attractor_regime": attractor_regime, "seeded": seed_attractor}


def grid_search_NT(
    model,
    target_regime: int,
    config: CalibrationConfig = DEFAULT_CALIBRATION_CONFIG,
    N_revisit_grid: np.ndarray | None = None,
    N_dwell_grid: np.ndarray | None = None,
    T_grid: np.ndarray | None = None,
    base_seed: int = 0,
) -> dict:
    """Grid-search (N_revisit, N_dwell, T) using paired seeded/noise trial
    batches (reused across grid points — only the threshold varies, so
    threshold sensitivity isn't conflated with resampling noise). Selects
    the point meeting config.target_precision with lowest false-negative
    rate, tie-broken by lowest false-positive rate."""
    N_revisit_grid = np.array(N_revisit_grid if N_revisit_grid is not None else config.N_revisit_grid)
    N_dwell_grid = np.array(N_dwell_grid if N_dwell_grid is not None else config.N_dwell_grid)
    if T_grid is None:
        base_var = float(np.mean(model.var[target_regime]))
        T_grid = np.array([base_var * m for m in config.T_grid_multipliers])
    else:
        T_grid = np.array(T_grid)

    n_timesteps, n_trials = config.n_timesteps, config.n_trials

    seeded_trials = [
        sample_from_fitted_hssm(model, n_timesteps, seed_attractor=True, attractor_regime=target_regime,
                                 attractor_strength=config.attractor_strength, seed=base_seed + i)
        for i in range(n_trials)
    ]
    noise_trials = [
        sample_from_fitted_hssm(model, n_timesteps, seed_attractor=False, attractor_regime=target_regime,
                                 seed=base_seed + 1000 + i)
        for i in range(n_trials)
    ]
    seeded_stats = [compute_attractor_stats(s["m_t"], s["regime_labels"], target_regime) for s in seeded_trials]
    noise_stats = [compute_attractor_stats(s["m_t"], s["regime_labels"], target_regime) for s in noise_trials]

    results = []
    for Nr in N_revisit_grid:
        for Nd in N_dwell_grid:
            for T in T_grid:
                fn_count = sum(1 for s in seeded_stats if not is_attractor(s, (Nr, Nd), T))
                fp_count = sum(1 for s in noise_stats if is_attractor(s, (Nr, Nd), T))
                fpr = fp_count / n_trials
                fnr = fn_count / n_trials
                tp = n_trials - fn_count
                precision = tp / (tp + fp_count) if (tp + fp_count) > 0 else 0.0
                results.append({"N_revisit": float(Nr), "N_dwell": float(Nd), "T": float(T),
                                 "fpr": fpr, "fnr": fnr, "precision": precision})

    candidates = [r for r in results if r["precision"] >= config.target_precision]
    if not candidates:
        best = max(results, key=lambda r: r["precision"])
        return {
            "selected_N_revisit": best["N_revisit"], "selected_N_dwell": best["N_dwell"],
            "selected_T": best["T"], "achieved_precision": best["precision"],
            "target_met": False, "all_results": results,
            "warning": (
                f"No grid point met target_precision={config.target_precision} for "
                f"this user's data density. Returning best available point. Must be "
                f"surfaced, not silently accepted."
            ),
        }

    best = min(candidates, key=lambda r: (r["fnr"], r["fpr"]))
    return {
        "selected_N_revisit": best["N_revisit"], "selected_N_dwell": best["N_dwell"],
        "selected_T": best["T"], "achieved_precision": best["precision"],
        "target_met": True, "all_results": results,
    }
