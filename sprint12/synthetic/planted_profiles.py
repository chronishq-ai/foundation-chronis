"""
synthetic/planted_profiles.py

Sprint 8, Day 24 — synthetic validation suite.

Generates planted-ground-truth (p_t, q_t, m_t, n_t) trajectories for three of
the four divergence types (Ignorance, Aspiration, Self-Protection — Active
Transition profiles are Sprint 15's job per the directive, but a basic AT
generator is included here too so engine.py's four-way scorer can be smoke
tested end-to-end now rather than only in Sprint 15).

Each profile generator returns a `DivergenceInputs`-ready bundle plus the
planted ground-truth label, so `run_validation_suite` can compute per-type
accuracy/precision/recall against `TypeScores.dominant()`.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Tuple
import numpy as np

from divergence_engine.engine import DivergenceInputs, compute_divergence_state

RNG_SEED = 42


def _timestamps(n: int, start: datetime) -> List[datetime]:
    return [start + timedelta(days=i) for i in range(n)]


def _make_regime_labels(n: int, regime_id: int, other_id: int, occupancy_frac: float, rng: np.random.Generator) -> np.ndarray:
    labels = np.where(rng.random(n) < occupancy_frac, regime_id, other_id)
    return labels


def _make_correlated_regime_pair(
    n: int,
    p_occupancy: float,
    q_occupancy: float,
    joint_overlap: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (p_t, q_t) with a DELIBERATE, controllable co-occupancy structure,
    rather than independent sampling (which produces no association by
    construction and can never trigger Fisher significance).

    `joint_overlap` in [0,1]: fraction of behavioral-regime-1 sessions that are
    FORCED to also be narrative-regime-1 (drives the contingency table's
    diagonal mass directly).
    """
    p_t = np.where(rng.random(n) < p_occupancy, 1, 0)
    q_t = np.zeros(n, dtype=int)
    for i in range(n):
        if p_t[i] == 1 and rng.random() < joint_overlap:
            q_t[i] = 1
        else:
            q_t[i] = 1 if rng.random() < q_occupancy else 0
    return p_t, q_t


def generate_ignorance_profile(n_sessions: int = 60, seed: int = 0) -> Tuple[DivergenceInputs, str]:
    """Strong behavioral attractor, ~zero narrative engagement in this domain."""
    rng = np.random.default_rng(RNG_SEED + seed)
    p_t, q_t = _make_correlated_regime_pair(
        n_sessions, p_occupancy=0.85, q_occupancy=0.03, joint_overlap=0.02, rng=rng
    )
    m_t = rng.normal(0, 1, size=(n_sessions, 3))
    n_t = rng.normal(0, 1, size=(n_sessions, 3)) * 0.1  # near-flat narrative signal

    inputs = DivergenceInputs(
        user_id="synthetic-ignorance",
        domain_id="domX",
        window_start=datetime(2026, 1, 1),
        window_end=datetime(2026, 1, 1) + timedelta(days=n_sessions),
        p_t=p_t, q_t=q_t, m_t=m_t, n_t=n_t,
        behavioral_regime_id=1, narrative_regime_id=1,
        n_domain_pairs_tested=1,
        behavioral_attractor_weakening=False,
        narrative_conformal_confidence=0.7,
    )
    return inputs, "ignorance"


def generate_aspiration_profile(n_sessions: int = 100, seed: int = 0) -> Tuple[DivergenceInputs, str]:
    """Weakening behavioral attractor + future-tense agentic narrative regime leading."""
    rng = np.random.default_rng(RNG_SEED + 1000 + seed)
    p_t, q_t = _make_correlated_regime_pair(
        n_sessions, p_occupancy=0.5, q_occupancy=0.6, joint_overlap=0.8, rng=rng
    )
    # narrative fast-state leads: n_t at t strongly predicts m_t at t+1, ~zero reverse coupling
    n_t = rng.normal(0, 1, size=(n_sessions, 3))
    m_t = np.zeros((n_sessions, 3))
    m_t[1:] = 0.85 * n_t[:-1] + rng.normal(0, 0.15, size=(n_sessions - 1, 3))
    m_t[0] = rng.normal(0, 1, size=3)

    inputs = DivergenceInputs(
        user_id="synthetic-aspiration",
        domain_id="domX",
        window_start=datetime(2026, 1, 1),
        window_end=datetime(2026, 1, 1) + timedelta(days=n_sessions),
        p_t=p_t, q_t=q_t, m_t=m_t, n_t=n_t,
        behavioral_regime_id=1, narrative_regime_id=1,
        n_domain_pairs_tested=1,
        behavioral_attractor_weakening=True,
        narrative_conformal_confidence=0.7,
    )
    return inputs, "aspiration"


def generate_self_protection_profile(n_sessions: int = 100, seed: int = 0) -> Tuple[DivergenceInputs, str]:
    """Stable behavioral attractor + avoidant/passive narrative regime, co-occupying but NOT predictive either direction."""
    rng = np.random.default_rng(RNG_SEED + 2000 + seed)
    p_t, q_t = _make_correlated_regime_pair(
        n_sessions, p_occupancy=0.85, q_occupancy=0.8, joint_overlap=0.92, rng=rng
    )
    m_t = rng.normal(0, 1, size=(n_sessions, 3))
    n_t = rng.normal(0, 1, size=(n_sessions, 3))  # independent noise: co-occupies, but no Granger coupling either way

    inputs = DivergenceInputs(
        user_id="synthetic-self-protection",
        domain_id="domX",
        window_start=datetime(2026, 1, 1),
        window_end=datetime(2026, 1, 1) + timedelta(days=n_sessions),
        p_t=p_t, q_t=q_t, m_t=m_t, n_t=n_t,
        behavioral_regime_id=1, narrative_regime_id=1,
        n_domain_pairs_tested=1,
        behavioral_attractor_weakening=False,
        narrative_conformal_confidence=0.7,
    )
    return inputs, "self_protection"


def generate_active_transition_profile(n_sessions: int = 100, seed: int = 0) -> Tuple[DivergenceInputs, str]:
    """Both systems shifting together, bidirectional lagged coupling, co-occupying."""
    rng = np.random.default_rng(RNG_SEED + 3000 + seed)
    p_t, q_t = _make_correlated_regime_pair(
        n_sessions, p_occupancy=0.6, q_occupancy=0.6, joint_overlap=0.85, rng=rng
    )
    base = rng.normal(0, 1, size=(n_sessions, 3))
    m_t = np.zeros((n_sessions, 3))
    n_t = np.zeros((n_sessions, 3))
    m_t[0], n_t[0] = base[0], base[0]
    for t in range(1, n_sessions):
        m_t[t] = 0.7 * n_t[t - 1] + 0.15 * m_t[t - 1] + rng.normal(0, 0.2, size=3)
        n_t[t] = 0.7 * m_t[t - 1] + 0.15 * n_t[t - 1] + rng.normal(0, 0.2, size=3)

    inputs = DivergenceInputs(
        user_id="synthetic-active-transition",
        domain_id="domX",
        window_start=datetime(2026, 1, 1),
        window_end=datetime(2026, 1, 1) + timedelta(days=n_sessions),
        p_t=p_t, q_t=q_t, m_t=m_t, n_t=n_t,
        behavioral_regime_id=1, narrative_regime_id=1,
        n_domain_pairs_tested=1,
        behavioral_attractor_weakening=True,
        narrative_conformal_confidence=0.7,
    )
    return inputs, "active_transition"


PROFILE_GENERATORS: Dict[str, Callable[[int, int], Tuple[DivergenceInputs, str]]] = {
    "ignorance": generate_ignorance_profile,
    "aspiration": generate_aspiration_profile,
    "self_protection": generate_self_protection_profile,
    "active_transition": generate_active_transition_profile,
}


@dataclass
class ValidationReport:
    per_type_accuracy: Dict[str, float]
    per_type_precision: Dict[str, float]
    per_type_recall: Dict[str, float]
    ambiguous_count: int
    n_profiles_per_type: int


def run_validation_suite(n_profiles_per_type: int = 20) -> ValidationReport:
    """
    DoD (Sprint 8 Day 24 / Sprint 15 Day 44): >75% dominant-type classification
    accuracy per type, measured independently, on >=20 planted profiles/type.
    """
    y_true: List[str] = []
    y_pred: List[str] = []
    ambiguous_count = 0

    for true_type, generator in PROFILE_GENERATORS.items():
        for i in range(n_profiles_per_type):
            inputs, label = generator(seed=i)
            state = compute_divergence_state(inputs)
            pred = state.type_scores.dominant()
            y_true.append(true_type)
            if pred is None:
                ambiguous_count += 1
                y_pred.append("AMBIGUOUS")
            else:
                y_pred.append(pred)

    types = list(PROFILE_GENERATORS.keys())
    per_type_accuracy: Dict[str, float] = {}
    per_type_precision: Dict[str, float] = {}
    per_type_recall: Dict[str, float] = {}

    for t in types:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == t and yp == t)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == t and yp != t)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != t and yp == t)
        total_t = sum(1 for yt in y_true if yt == t)

        per_type_accuracy[t] = tp / total_t if total_t else 0.0
        per_type_precision[t] = tp / (tp + fp) if (tp + fp) else 0.0
        per_type_recall[t] = tp / (tp + fn) if (tp + fn) else 0.0

    return ValidationReport(
        per_type_accuracy=per_type_accuracy,
        per_type_precision=per_type_precision,
        per_type_recall=per_type_recall,
        ambiguous_count=ambiguous_count,
        n_profiles_per_type=n_profiles_per_type,
    )


if __name__ == "__main__":
    report = run_validation_suite(n_profiles_per_type=20)
    print("Per-type accuracy:", report.per_type_accuracy)
    print("Per-type precision:", report.per_type_precision)
    print("Per-type recall:", report.per_type_recall)
    print("Ambiguous (within 0.15) count:", report.ambiguous_count)
    for t, acc in report.per_type_accuracy.items():
        status = "PASS" if acc > 0.75 else "FAIL"
        print(f"  [{status}] {t}: {acc:.1%} (target >75%)")
