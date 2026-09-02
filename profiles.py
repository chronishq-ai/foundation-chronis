# Day 44 / Sprint 15 — planted profiles through the REAL pipeline path.
#
# Closure S15.1 (§14.1): planted b/n trajectories are turned into a synthetic
# feature matrix, fitted via backbone.hssm + nssm_pipeline, then scored by
# divergence_engine.engine.compute_divergence_state (real DivergenceState
# type_scores). The old hand-rolled formula lives in scratch_type_scores.py
# and must never be cited as validation evidence.
#
# Honesty: the Granger path behind the available divergence package remains
# OLS-VAR-limited (S79.1 still open on the research track). Accuracy numbers
# from this harness are end-to-end-through-available-packages numbers, not a
# claim that Bayesian MS-VAR is complete.
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

# T3 static check — these imports must remain (closure §14.1).
import backbone.hssm  # noqa: F401
import divergence_engine.engine  # noqa: F401
import nssm_pipeline  # noqa: F401

from .scratch_type_scores import recover_lag

TYPES = ("Ignorance", "Aspiration", "Self-Protection", "ActiveTransition")
AMBIGUITY_GAP = 0.15
N_PER_TYPE = 20
HORIZON = 48
AT_LAG = 4  # narrative change lags behavioral weakening by this many steps


@dataclass
class PlantedProfile:
    profile_id: int
    label: str
    b: np.ndarray  # attractor strength over time (higher = stronger attractor)
    n: np.ndarray  # narrative engagement / change process
    planted_lag: int  # designed lag (n lags b if positive)
    scores: dict[str, float] | None = None
    predicted: str | None = None
    ambiguous: bool = False
    recovered_lag: int | None = None


def dominant_type(scores: dict[str, float]) -> tuple[Optional[str], bool]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, second = ordered[0], ordered[1]
    if top[1] - second[1] < AMBIGUITY_GAP:
        return None, True
    return top[0], False


def _series(kind: str, rng: np.random.Generator, lag: int = AT_LAG) -> tuple[np.ndarray, np.ndarray, int]:
    """Plant structural raw trajectories (Sprint 8 Day 24 pattern shapes)."""
    t = np.arange(HORIZON, dtype=float)
    frac = t / (HORIZON - 1)
    noise_b = rng.normal(0, 0.02, HORIZON)
    noise_n = rng.normal(0, 0.02, HORIZON)
    if kind == "Ignorance":
        b = np.clip(0.9 + noise_b, 0, 1)
        n = np.clip(0.02 + noise_n * 0.2, 0, 1)
        return b, n, 0
    if kind == "Aspiration":
        b = np.clip(0.9 - 0.6 * frac + noise_b, 0, 1)
        n = np.clip(0.8 + noise_n, 0, 1)
        return b, n, 0
    if kind == "Self-Protection":
        b = np.clip(0.72 + noise_b * 0.5, 0, 1)
        n = np.clip(0.35 + noise_n * 0.5, 0, 1)
        return b, n, 0
    # AT: n is a delayed copy of the same weakening/rise process
    motion = np.zeros(HORIZON)
    start = 10
    ramp = 8
    motion[start : start + ramp] = np.linspace(0.0, 1.0, ramp)
    motion[start + ramp :] = 1.0
    delayed = np.zeros(HORIZON)
    delayed[start + lag : start + lag + ramp] = np.linspace(0.0, 1.0, ramp)
    delayed[start + lag + ramp :] = 1.0
    b = np.clip(0.9 - 0.75 * motion + noise_b, 0, 1)
    n = np.clip(0.1 + 0.75 * delayed + noise_n, 0, 1)
    return b, n, lag


def plant_profiles(n_per_type: int = N_PER_TYPE, seed: int = 7) -> list[PlantedProfile]:
    """Fit planted patterns through HSSM → NSSM → Divergence Engine."""
    from backbone.hssm import fit_hssm
    from divergence_engine.engine import DivergenceInputs, compute_divergence_state
    from nssm_pipeline import fit_nssm

    rng = np.random.default_rng(seed)
    out: list[PlantedProfile] = []
    i = 0

    for label in TYPES:
        for _ in range(n_per_type):
            b, n, lag = _series(label, rng)
            raw_matrix = np.stack([b, n], axis=1)

            hssm_out = fit_hssm(raw_matrix, random_seed=seed + i)
            nssm_out = fit_nssm(raw_matrix, random_seed=seed + i)

            weakening = bool((hssm_out.m_t[0] - hssm_out.m_t[-1]) > 0.15)
            inputs = DivergenceInputs(
                user_id=f"user_{i}",
                domain_id="dom-synth",
                window_start=datetime.now(timezone.utc),
                window_end=datetime.now(timezone.utc),
                p_t=hssm_out.p_t,
                q_t=nssm_out.q_t,
                m_t=hssm_out.m_t,
                n_t=nssm_out.n_t,
                behavioral_regime_id=1,
                narrative_regime_id=1,
                n_domain_pairs_tested=1,
                behavioral_attractor_weakening=weakening,
                narrative_conformal_confidence=0.8,
            )
            state = compute_divergence_state(inputs)
            scores = dict(state.type_scores)
            pred = state.dominant_type
            amb = state.ambiguous
            if pred is None and not amb:
                pred, amb = dominant_type(scores)

            recovered, _corr = recover_lag(hssm_out.m_t, nssm_out.n_t)

            out.append(
                PlantedProfile(
                    profile_id=i,
                    label=label,
                    b=b,
                    n=n,
                    planted_lag=lag,
                    scores=scores,
                    predicted=pred,
                    ambiguous=amb,
                    recovered_lag=int(recovered),
                )
            )
            i += 1
    return out


def type_accuracy(profiles: list[PlantedProfile]) -> dict[str, float]:
    hits: dict[str, list[int]] = {t: [] for t in TYPES}
    for p in profiles:
        if p.ambiguous or p.predicted is None:
            continue
        hits[p.label].append(int(p.predicted == p.label))
    return {t: (sum(v) / len(v) if v else 0.0) for t, v in hits.items()}


def log_accuracy_mlflow(
    acc: dict[str, float],
    profiles: list[PlantedProfile],
    *,
    tracking_uri: str,
) -> str:
    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    n_amb = sum(1 for p in profiles if p.ambiguous)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("chronis.sprint15.divergence_types")
    with mlflow.start_run(run_name="planted-type-accuracy") as run:
        for t, v in acc.items():
            mlflow.log_metric(f"accuracy_{t}", float(v))
            n = sum(1 for p in profiles if p.label == t)
            mlflow.log_metric(f"n_{t}", n)
        mlflow.log_metric("n_ambiguous", n_amb)
        mlflow.set_tag("sprint", "15")
        mlflow.set_tag(
            "note",
            "end-to-end via backbone.hssm + nssm_pipeline + divergence_engine; "
            "OLS-VAR-limited Granger (S79.1) — not Bayesian MS-VAR; "
            "synthetic planted validation, not external validity",
        )
        mlflow.set_tag("granger_status", "OLS-VAR-limited (S79.1 open)")
        return run.info.run_id
