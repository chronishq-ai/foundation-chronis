"""
Day 18 -- Domain confidence scoring (Bible Part 5.8).

Confidence computed from 4 factors:
  - observation count (more episodes -> more confidence, saturating)
  - persistence duration (longer-lived domain -> more confidence, saturating)
  - cross-phase survival (domains that survive a phase transition get the
    HIGHEST confidence weight -- doctrine calls this "the strongest signal
    of true stability")
  - behavioral-narrative coherence: inverted Fisher's p-value from the
    domain_alignment.py joint-domain test (lower p -> higher coherence)

Domains below MIN_CONFIDENCE_THRESHOLD are "candidate" status only -- not
usable as input to the claims engine (out of scope here, Sprint 6 doesn't
build the claims engine, but the status label is what a future integration
would gate on).
"""

from __future__ import annotations
from dataclasses import dataclass

MIN_CONFIDENCE_THRESHOLD = 0.5

DEFAULT_WEIGHTS = {
    "observation": 0.2,
    "persistence": 0.2,
    "survival": 0.4,       # highest weight -- doctrine: strongest stability signal
    "coherence": 0.2,
}


@dataclass
class DomainConfidence:
    observation_score: float
    persistence_score: float
    survival_score: float
    coherence_score: float
    confidence: float
    status: str   # "active" | "candidate"


def _saturating(x: float, scale: float) -> float:
    """Maps [0, inf) -> [0, 1), saturating -- more observations/duration
    always help but with diminishing returns, never fully caps at exactly 1."""
    x = max(x, 0.0)
    return x / (x + scale)


def compute_domain_confidence(
    observation_count: int,
    persistence_duration: float,
    n_phase_transitions_survived: int,
    fisher_p_value: float,
    weights: dict | None = None,
    obs_scale: float = 50.0,
    persistence_scale: float = 30.0,
    survival_scale: float = 1.0,
    threshold: float = MIN_CONFIDENCE_THRESHOLD,
) -> DomainConfidence:
    """Compute weighted domain confidence and derive active/candidate
    status. fisher_p_value should be the Bonferroni-corrected p from
    domain_alignment.align_domains (0 = perfectly coherent, 1 = no
    coherence). survival_scale=1.0 means even 1 survived transition already
    gives strong (0.5) survival credit, consistent with doctrine treating
    ANY cross-phase survival as a strong signal, not requiring many."""
    w = weights or DEFAULT_WEIGHTS
    if not abs(sum(w.values()) - 1.0) < 1e-9:
        raise ValueError(f"weights must sum to 1.0, got {sum(w.values())}")

    obs_score = _saturating(observation_count, obs_scale)
    persistence_score = _saturating(persistence_duration, persistence_scale)
    survival_score = _saturating(n_phase_transitions_survived, survival_scale)
    coherence_score = max(0.0, min(1.0, 1.0 - fisher_p_value))

    confidence = (
        w["observation"] * obs_score
        + w["persistence"] * persistence_score
        + w["survival"] * survival_score
        + w["coherence"] * coherence_score
    )
    confidence = max(0.0, min(1.0, confidence))

    status = "active" if confidence >= threshold else "candidate"

    return DomainConfidence(
        observation_score=obs_score,
        persistence_score=persistence_score,
        survival_score=survival_score,
        coherence_score=coherence_score,
        confidence=confidence,
        status=status,
    )