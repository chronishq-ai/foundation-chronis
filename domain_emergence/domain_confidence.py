"""
Day 18 -- Domain confidence scoring (Bible Part 5.8).

Confidence computed from 4 factors:
  - observation count (more episodes -> more confidence, saturating)
  - persistence duration (longer-lived domain -> more confidence, saturating)
  - cross-phase survival (domains that survive a phase transition get the
    HIGHEST confidence weight -- doctrine calls this "the strongest signal
    of true stability")
  - behavioral-narrative coherence: a real effect-size estimate of the
    co-occurrence relationship (see S56.4 below), NOT an inverted p-value.

Domains below MIN_CONFIDENCE_THRESHOLD are "candidate" status only -- not
usable as input to the claims engine (out of scope here, Sprint 6 doesn't
build the claims engine, but the status label is what a future integration
would gate on).
"""

from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np

MIN_CONFIDENCE_THRESHOLD = 0.5

# S56.4 METRIC SWAP -- implemented per user request, ships WITHOUT the
# Mandatory senior sign-off the pack requires for this ID (statistical-
# estimator design choice, "Senior-only per the Ownership Model"). DO
# NOT MERGE without that review; flag it explicitly in the PR.
#
# Chosen formulation: bootstrap-stability effect size (one of the pack's
# three named acceptable options -- odds ratio+interval, Bayesian
# posterior, or bootstrap stability). coherence_score is now the LOWER
# bound of a bootstrap percentile confidence interval on the observed
# co-occurrence RATE across episodes (`co_occurrence_indicator`, a 0/1
# array). This is a genuine effect-size/confidence-in-magnitude
# statement ("we're 95% confident the true co-occurrence rate is at
# least this high"), not a claim about P(the domain relationship is
# true) -- it does not have the p-value-as-probability misinterpretation
# `1 - fisher_p_value` had.
#
# Legacy path: when `co_occurrence_indicator` is NOT supplied (caller
# only has a p-value, e.g. not yet threaded per-episode indicators
# through), compute_domain_confidence falls back to the old
# `1 - fisher_p_value` formula and emits a RuntimeWarning every call --
# this fallback is flagged, not silently treated as equivalent.


def _bootstrap_coherence_effect_size(
    co_occurrence_indicator: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> dict:
    """Bootstrap percentile CI on the co-occurrence rate. Returns the
    observed rate plus the CI bounds; `ci_lower` is used as the
    conservative coherence effect-size estimate."""
    x = np.asarray(co_occurrence_indicator, dtype=float)
    n = len(x)
    if n == 0:
        return {"rate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    rng = np.random.default_rng(seed)
    boot_rates = np.array([
        x[rng.integers(0, n, size=n)].mean() for _ in range(n_bootstrap)
    ])
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(boot_rates, alpha))
    upper = float(np.quantile(boot_rates, 1.0 - alpha))
    return {
        "rate": float(x.mean()),
        "ci_lower": max(0.0, lower),
        "ci_upper": min(1.0, upper),
    }


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
    coherence_method: str = "legacy_one_minus_p"  # or "bootstrap_ci_lower"


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
    co_occurrence_indicator: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    bootstrap_ci: float = 0.95,
    bootstrap_seed: int | None = None,
    weights: dict | None = None,
    obs_scale: float = 50.0,
    persistence_scale: float = 30.0,
    survival_scale: float = 1.0,
    threshold: float = MIN_CONFIDENCE_THRESHOLD,
) -> DomainConfidence:
    """Compute weighted domain confidence and derive active/candidate
    status.

    co_occurrence_indicator (S56.4 fix, preferred path): 1D array of
    0/1 per episode/window indicating whether the joint-domain pattern
    held. When supplied, coherence_score is the lower bound of a
    bootstrap percentile CI on that rate -- a real effect-size
    statement, not a p-value inversion. fisher_p_value is still
    accepted/stored for logging even in this path but no longer drives
    coherence_score.

    fisher_p_value (legacy path, used only when co_occurrence_indicator
    is omitted): the Bonferroni-corrected p from
    domain_alignment.align_domains. Falls back to `1 - fisher_p_value`
    and emits a RuntimeWarning, since that formula misrepresents a
    p-value as a probability the relationship is true -- kept only for
    backward compatibility with callers that don't yet have per-episode
    indicators wired through.

    survival_scale=1.0 means even 1 survived transition already gives
    strong (0.5) survival credit, consistent with doctrine treating ANY
    cross-phase survival as a strong signal, not requiring many."""
    w = weights or DEFAULT_WEIGHTS
    if not abs(sum(w.values()) - 1.0) < 1e-9:
        raise ValueError(f"weights must sum to 1.0, got {sum(w.values())}")

    obs_score = _saturating(observation_count, obs_scale)
    persistence_score = _saturating(persistence_duration, persistence_scale)
    survival_score = _saturating(n_phase_transitions_survived, survival_scale)

    if co_occurrence_indicator is not None:
        effect_size = _bootstrap_coherence_effect_size(
            co_occurrence_indicator, n_bootstrap=n_bootstrap,
            ci=bootstrap_ci, seed=bootstrap_seed,
        )
        coherence_score = effect_size["ci_lower"]
        coherence_method = "bootstrap_ci_lower"
    else:
        warnings.warn(
            "compute_domain_confidence: co_occurrence_indicator not "
            "supplied -- falling back to legacy `1 - fisher_p_value` "
            "coherence formula, which is NOT a valid effect-size "
            "estimate (see S56.4 HONESTY FLAG in domain_confidence.py). "
            "Pass per-episode co_occurrence_indicator when available.",
            RuntimeWarning, stacklevel=2,
        )
        coherence_score = max(0.0, min(1.0, 1.0 - fisher_p_value))
        coherence_method = "legacy_one_minus_p"

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
        coherence_method=coherence_method,
    )


def bootstrap_domain_stability(
    co_occurrence_indicator: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> float:
    """Original S56.4 diagnostic: fraction of bootstrap resamples where
    the pattern's resampled rate stays above 0 (a coarse stability
    proxy). Kept for backward compatibility / comparison purposes.
    `compute_domain_confidence`'s actual coherence_score now uses
    `_bootstrap_coherence_effect_size`'s CI-lower-bound instead (a
    stricter effect-size estimate than this simple >0 fraction)."""
    x = np.asarray(co_occurrence_indicator)
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return 0.0
    hits = 0
    for _ in range(n_bootstrap):
        resample = x[rng.integers(0, n, size=n)]
        if resample.mean() > 0:
            hits += 1
    return hits / n_bootstrap


def compare_confidence_formulations(
    fisher_p_value: float,
    co_occurrence_indicator: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> dict:
    """Reports `1 - fisher_p_value` (legacy) side by side with the
    bootstrap-stability-fraction estimate and the CI-lower-bound
    estimate now actually used by compute_domain_confidence, so all
    three are comparable on the same underlying evidence."""
    naive = max(0.0, min(1.0, 1.0 - fisher_p_value))
    bootstrap = bootstrap_domain_stability(co_occurrence_indicator, n_bootstrap, seed)
    effect_size = _bootstrap_coherence_effect_size(
        co_occurrence_indicator, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "naive_one_minus_p": naive,
        "bootstrap_stability": bootstrap,
        "bootstrap_ci_lower": effect_size["ci_lower"],
        "divergence": abs(naive - bootstrap),
    }