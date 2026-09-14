"""
Day 18 -- Domain confidence scoring (Bible Part 5.8).

R2-S56.5 FIX (item 5 of Sprint 5-6 changes pack).

Confidence computed from 4 factors:
  - observation count (more episodes -> more confidence, saturating)
  - persistence duration (longer-lived domain -> more confidence, saturating)
  - cross-phase survival (domains that survive a phase transition get the
    HIGHEST confidence weight -- doctrine calls this "the strongest signal
    of true stability")
  - behavioral-narrative coherence: a real dependence-aware effect-size
    estimate (phi coefficient + bootstrap CI), NOT raw co-occurrence rate
    and NOT an inverted p-value.

The old design conflated two different statistical questions under one
"coherence" number:
  (a) is the behavioral<->narrative association statistically significant
      (a null-hypothesis test question -- answered upstream by
      domain_alignment.align_domains' corrected p-value, R2-S56.2), and
  (b) how strong/stable is that association (an effect-size question).

Neither of those was ever "the probability the domain relationship is
true" -- `1 - fisher_p_value` treated a p-value as if it were that
probability, which is not a valid interpretation of a p-value, and the
raw co-occurrence RATE isn't a dependence measure either (two totally
independent streams that are each individually common will co-occur
often by chance alone).

This module now keeps those two questions as two separate, explicitly
named objects:
  - DomainSignificance: the corrected p-value/method from upstream
    (domain_alignment), plus a plain significant/not-significant call
    at a declared alpha. Never converted into a magnitude or a score.
  - DomainMagnitude: a genuine dependence effect size -- the phi
    coefficient (the binary-variable analogue of a correlation
    coefficient) between the per-episode behavioral and narrative
    indicator arrays, with a bootstrap percentile CI. This is what
    actually answers "how coherent is this pairing", correcting for
    each stream's own base rate.

`compute_domain_confidence` REQUIRES both a corrected significance
result and both per-episode indicator arrays -- there is no fallback
path in this module. Callers that only have a raw/uncorrected p-value
and no per-episode indicators cannot call this function; see
`domain_emergence.legacy_diagnostic` for the quarantined old formula,
which is diagnostic-only and must never be imported by production code
(enforced by the repo-wide grep CI check, item 8/XCUT-2).

Domains below MIN_CONFIDENCE_THRESHOLD are "candidate" status only -- not
usable as input to the claims engine (out of scope here, Sprint 6 doesn't
build the claims engine, but the status label is what a future integration
would gate on).
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

MIN_CONFIDENCE_THRESHOLD = 0.5

# Corrected p-value provenance this module will accept as a genuine
# "significance" input. Anything else is refused -- an uncorrected raw
# p-value is not an acceptable substitute (see domain_alignment.py
# R2-S56.2 / R2-S56.6 for what produces these).
_ACCEPTED_SIGNIFICANCE_METHODS = frozenset({
    "per_episode_fisher_exact",
    "subject_level_within_subject_permutation",
})


class MissingCorrectedEvidenceError(TypeError):
    """Raised when compute_domain_confidence is called without the full
    corrected-evidence bundle it now requires: a significance result
    from domain_alignment's corrected p-value pipeline AND both
    per-episode behavioral/narrative indicator arrays for the real
    dependence effect size. No silent fallback to an uncorrected or
    invented number -- this is a typed failure, not a bare `assert`
    (PH0.2 pattern)."""


class UnrecognizedSignificanceMethodError(ValueError):
    """Raised when significance_method isn't one of the corrected
    pipeline's known output labels -- refuses to treat an arbitrary or
    uncorrected p-value as if it were the corrected result."""


@dataclass
class DomainSignificance:
    """Answers ONE question: is the association statistically
    significant at the declared alpha, per the corrected null test
    upstream. Never a magnitude, never a "probability the domain is
    true"."""
    p_value: float
    method: str
    alpha: float
    is_significant: bool


@dataclass
class DomainMagnitude:
    """Answers a different question: how strong/stable is the
    association, independent of whether it cleared a significance
    threshold. `phi` is the observed phi coefficient (in [-1, 1]) between
    the per-episode behavioral and narrative indicator arrays; `ci_lower`
    /`ci_upper` bound a bootstrap percentile CI on phi. This corrects for
    each stream's own base rate, unlike a raw co-occurrence rate."""
    phi: float
    ci_lower: float
    ci_upper: float
    n_bootstrap: int


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
    significance: DomainSignificance
    magnitude: DomainMagnitude
    coherence_method: str = "phi_bootstrap_ci_lower"


def _saturating(x: float, scale: float) -> float:
    """Maps [0, inf) -> [0, 1), saturating -- more observations/duration
    always help but with diminishing returns, never fully caps at exactly 1."""
    x = max(x, 0.0)
    return x / (x + scale)


def _phi_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Matthews/phi correlation coefficient between two binary arrays --
    a real dependence measure that accounts for each variable's own
    base rate (unlike a raw joint co-occurrence rate). 0 = independent,
    +1 = perfectly co-occurring, -1 = perfectly mutually exclusive."""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    n11 = int(np.sum(a & b))
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n00 = int(np.sum(~a & ~b))
    denom = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if denom == 0.0:
        return 0.0
    return (n11 * n00 - n10 * n01) / denom


def _bootstrap_phi_ci(
    behavioral_indicator: np.ndarray,
    narrative_indicator: np.ndarray,
    n_bootstrap: int,
    ci: float,
    seed: int | None,
) -> tuple[float, float]:
    """Paired bootstrap (resample episode indices, keep the two streams
    paired per resample) percentile CI on the phi coefficient."""
    a = np.asarray(behavioral_indicator)
    b = np.asarray(narrative_indicator)
    n = len(a)
    if n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot_phi = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_phi[i] = _phi_coefficient(a[idx], b[idx])
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(boot_phi, alpha))
    upper = float(np.quantile(boot_phi, 1.0 - alpha))
    return max(-1.0, lower), min(1.0, upper)


def compute_domain_confidence(
    observation_count: int,
    persistence_duration: float,
    n_phase_transitions_survived: int,
    significance_pvalue: float,
    significance_method: str,
    behavioral_indicator: np.ndarray | None,
    narrative_indicator: np.ndarray | None,
    alpha: float = 0.05,
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

    significance_pvalue / significance_method (REQUIRED, R2-S56.5 fix):
    the corrected p-value and its provenance label from
    domain_alignment.align_domains' AlignmentResult (`pair_pvalues`,
    `pvalue_method`) -- must be one of the corrected pipeline's known
    method labels. Used ONLY to answer significance (is_significant at
    `alpha`), never inverted into a score or blended into `confidence`.

    behavioral_indicator / narrative_indicator (REQUIRED, R2-S56.5 fix):
    parallel 1D 0/1 arrays, one entry per episode, indicating whether
    the behavioral pattern / narrative pattern held in that episode.
    Used to compute the phi coefficient (a real dependence effect size)
    and its bootstrap CI -- this is what drives `coherence_score`, not
    a raw co-occurrence rate and not a p-value.

    Calling this without the full corrected-evidence bundle raises
    MissingCorrectedEvidenceError -- there is no silent fallback in this
    module (see domain_emergence.legacy_diagnostic for the quarantined
    old formula, diagnostic-only, never wired into this function).

    survival_scale=1.0 means even 1 survived transition already gives
    strong (0.5) survival credit, consistent with doctrine treating ANY
    cross-phase survival as a strong signal, not requiring many."""
    if behavioral_indicator is None or narrative_indicator is None:
        raise MissingCorrectedEvidenceError(
            "compute_domain_confidence requires both behavioral_indicator "
            "and narrative_indicator (per-episode 0/1 arrays) to compute "
            "a real dependence effect size -- there is no legacy "
            "raw-p-value fallback in this module. See "
            "domain_emergence.legacy_diagnostic if you are doing "
            "migration-era comparison work only (never for a production "
            "decision)."
        )
    if len(behavioral_indicator) != len(narrative_indicator):
        raise ValueError(
            "behavioral_indicator and narrative_indicator must be the "
            "same length (one entry per episode)."
        )
    if significance_method not in _ACCEPTED_SIGNIFICANCE_METHODS:
        raise UnrecognizedSignificanceMethodError(
            f"significance_method={significance_method!r} is not a "
            f"recognized corrected-pipeline output "
            f"({sorted(_ACCEPTED_SIGNIFICANCE_METHODS)}). Pass the "
            "pvalue_method from domain_alignment.AlignmentResult, not "
            "an arbitrary/uncorrected p-value."
        )

    w = weights or DEFAULT_WEIGHTS
    if not abs(sum(w.values()) - 1.0) < 1e-9:
        raise ValueError(f"weights must sum to 1.0, got {sum(w.values())}")

    obs_score = _saturating(observation_count, obs_scale)
    persistence_score = _saturating(persistence_duration, persistence_scale)
    survival_score = _saturating(n_phase_transitions_survived, survival_scale)

    significance = DomainSignificance(
        p_value=significance_pvalue,
        method=significance_method,
        alpha=alpha,
        is_significant=significance_pvalue < alpha,
    )

    phi = _phi_coefficient(behavioral_indicator, narrative_indicator)
    ci_lower, ci_upper = _bootstrap_phi_ci(
        behavioral_indicator, narrative_indicator,
        n_bootstrap=n_bootstrap, ci=bootstrap_ci, seed=bootstrap_seed,
    )
    magnitude = DomainMagnitude(
        phi=phi, ci_lower=ci_lower, ci_upper=ci_upper, n_bootstrap=n_bootstrap,
    )
    # Coherence score used in the confidence blend: the conservative
    # (CI-lower-bound) dependence effect size, clipped to [0, 1] --
    # negative/anti-correlated pairings contribute no positive
    # "coherence" credit rather than being folded in as a penalty here.
    coherence_score = max(0.0, ci_lower)

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
        significance=significance,
        magnitude=magnitude,
        coherence_method="phi_bootstrap_ci_lower",
    )