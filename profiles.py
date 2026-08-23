# Day 44 — planted profiles + four type scores.
#
# AT profiles have a weakening behavioral attractor AND a changing narrative
# process (q_t / n_t stand-in), with a designed lag. Lag direction is recovered
# from rate-of-change correlation of b_t and n_t.
#
# All four types are planted 20+ each. Dominant type must clear >75%
# independently. Ambiguous pairs (top-two scores within 0.15) are logged,
# never forced — Sprint 8 residual-ambiguity rule.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

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


def recover_lag(b: np.ndarray, n: np.ndarray, max_lag: int = 12) -> tuple[int, float]:
    # Narrative engagement rising as the attractor weakens → correlate -db with dn.
    db = -np.diff(np.asarray(b, dtype=float))
    dn = np.diff(np.asarray(n, dtype=float))
    best_lag = 0
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = db[-lag:], dn[: len(dn) + lag]
        elif lag > 0:
            x, y = db[: len(db) - lag], dn[lag:]
        else:
            x, y = db, dn
        if len(x) < 10:
            continue
        if np.std(x) < 1e-6 or np.std(y) < 1e-6:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if np.isnan(c):
            continue
        if c > best:
            best, best_lag = c, lag
    return best_lag, best


def type_scores(b: np.ndarray, n: np.ndarray) -> dict[str, float]:
    b = np.asarray(b, dtype=float)
    n = np.asarray(n, dtype=float)
    dn = np.diff(n)
    b_mean = float(np.mean(b))
    b_drop = float(np.clip(b[0] - b[-1], 0, 1))
    b_var = float(np.std(b))
    n_mean = float(np.mean(n))
    n_end = float(n[-1])
    n_rise = float(np.clip(n[-1] - n[0], 0, 1))
    n_change = float(np.mean(np.abs(dn)))
    lag, corr = recover_lag(b, n)

    ignorance = b_mean * (1.0 - n_mean) * (1.0 - n_rise)
    aspiration = b_drop * n_mean * (1.0 - n_rise)
    self_prot = (1.0 - min(b_var * 6, 1.0)) * (1.0 - abs(n_mean - 0.35)) * (1.0 - n_rise)
    lag_ok = 1.0 if lag >= 2 else 0.15
    at = b_drop * n_rise * max(corr, 0.0) * lag_ok

    raw = {
        "Ignorance": ignorance,
        "Aspiration": aspiration,
        "Self-Protection": self_prot,
        "ActiveTransition": at,
    }
    return {k: float(np.clip(v, 0, 1)) for k, v in raw.items()}


def dominant_type(scores: dict[str, float]) -> tuple[Optional[str], bool]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, second = ordered[0], ordered[1]
    if top[1] - second[1] < AMBIGUITY_GAP:
        return None, True
    return top[0], False


def _series(kind: str, rng: np.random.Generator, lag: int = AT_LAG) -> tuple[np.ndarray, np.ndarray, int]:
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
    rng = np.random.default_rng(seed)
    out: list[PlantedProfile] = []
    i = 0
    for label in TYPES:
        for _ in range(n_per_type):
            b, n, lag = _series(label, rng)
            scores = type_scores(b, n)
            pred, amb = dominant_type(scores)
            rec_lag, _ = recover_lag(b, n)
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
                    recovered_lag=rec_lag,
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
        mlflow.set_tag("note", "synthetic planted validation; not external validity")
        return run.info.run_id
