"""
Sprint 4, Day 10 — Attractor detection (Bible 5.2).

Three statistics per user/regime: revisit_count, mean_dwell_time,
transition_stability. Attractor declared ONLY if all three exceed
person-calibrated thresholds — hard AND, never a weighted score.
DTW+HDBSCAN shape clustering is exploratory only, never proof, and is
deliberately NOT implemented in this admissibility path.
"""

from __future__ import annotations
import numpy as np

from backbone.attractors.config import DEFAULT_ATTRACTOR_CONFIG, AttractorConfig


def compute_attractor_stats(
    m_t: np.ndarray,
    regime_labels: np.ndarray,
    target_regime: int,
    neighborhood_radius: float | None = None,
    config: AttractorConfig = DEFAULT_ATTRACTOR_CONFIG,
) -> dict:
    """Compute revisit_count, mean_dwell_time, transition_stability for one
    user/regime, restricted to timesteps assigned to target_regime.

    neighborhood_radius default uses norm(std), not mean(std) — an earlier
    version used mean(std) and undershot by ~sqrt(F) in multi-dimensional
    space, making revisit_count come back zero for everything. Fixed."""
    mask = regime_labels == target_regime
    m_sub = m_t[mask]

    if len(m_sub) < 3:
        return {
            "revisit_count": 0, "mean_dwell_time": 0.0, "transition_stability": np.inf,
            "n_observations": len(m_sub), "insufficient_data": True,
        }

    modal_value = np.median(m_sub, axis=0)

    if neighborhood_radius is None:
        neighborhood_radius = float(config.neighborhood_radius_multiplier * np.linalg.norm(np.std(m_sub, axis=0)))

    dist_to_modal = np.linalg.norm(m_sub - modal_value, axis=1)
    inside = dist_to_modal <= neighborhood_radius

    revisit_count = 0
    dwell_times = []
    entry_exit_points = []
    run_len = 0
    for i, val in enumerate(inside):
        if val:
            if run_len == 0:
                revisit_count += 1
                entry_exit_points.append(m_sub[i])
            run_len += 1
        else:
            if run_len > 0:
                dwell_times.append(run_len)
                entry_exit_points.append(m_sub[i - 1])
            run_len = 0
    if run_len > 0:
        dwell_times.append(run_len)
        entry_exit_points.append(m_sub[-1])

    mean_dwell_time = float(np.mean(dwell_times)) if dwell_times else 0.0

    if len(entry_exit_points) >= 2:
        ee = np.array(entry_exit_points)
        transition_stability = float(np.mean(np.var(ee, axis=0)))
    else:
        transition_stability = np.inf

    assert neighborhood_radius is not None
    return {
        "revisit_count": revisit_count,
        "mean_dwell_time": mean_dwell_time,
        "transition_stability": transition_stability,
        "n_observations": len(m_sub),
        "neighborhood_radius": float(neighborhood_radius),
        "modal_value": modal_value.tolist(),
        "insufficient_data": False,
    }


def is_attractor(stats: dict, N, T: float) -> bool:
    """Hard AND admissibility rule. N accepts a scalar (broadcast to both) or
    a (N_revisit, N_dwell) tuple — split after a real bug where forcing
    revisit_count and mean_dwell_time (very different numeric scales) through
    one shared threshold meant no value of N could ever work for both.
    transition_stability is a variance — LOWER is more stable, so T governs
    that side with the opposite inequality direction."""
    if stats.get("insufficient_data"):
        return False
    if isinstance(N, tuple):
        N_revisit, N_dwell = N
    else:
        N_revisit = N_dwell = N
    return (
        stats["revisit_count"] > N_revisit
        and stats["mean_dwell_time"] > N_dwell
        and stats["transition_stability"] < T
    )
