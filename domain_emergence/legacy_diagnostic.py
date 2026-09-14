"""
legacy_diagnostic -- QUARANTINE MODULE (R2-S56.5 / item 5 fix).

Everything in this file is diagnostic/migration-comparison tooling ONLY.
Nothing here may be imported by any production code path
(`domain_emergence/domain_confidence.py` or anything it feeds). This is
the repo-wide CI-enforced boundary from item 8/XCUT-2: production code
must never call `1 - p` and pretend it's a probability the domain
relationship is true.

Kept only so a reviewer can compare the old naive formula against the
current corrected approach (DomainSignificance + DomainMagnitude in
domain_confidence.py) on the same underlying data, e.g. while writing
up the migration / sign-off doc for R2-S56.2.
"""

from __future__ import annotations
import numpy as np

from domain_emergence.domain_confidence import _phi_coefficient, _bootstrap_phi_ci


def naive_one_minus_p_confidence(fisher_p_value: float) -> float:
    """THE FLAGGED LEGACY FORMULA. `1 - p` is NOT a valid probability
    that a relationship is true -- a p-value is the probability of
    seeing data this extreme (or more) under the null, not the
    probability the null (or its complement) is correct. Diagnostic-use
    only; never call this from a production decision path."""
    return max(0.0, min(1.0, 1.0 - fisher_p_value))


def bootstrap_domain_stability(
    co_occurrence_indicator: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> float:
    """Original S56.4 diagnostic: fraction of bootstrap resamples where
    the pattern's resampled rate stays above 0 (a coarse stability
    proxy, and -- like the raw co-occurrence rate it's built on -- not
    a dependence measure). Superseded in production by the phi-based
    effect size in domain_confidence.py. Kept for comparison only."""
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
    behavioral_indicator: np.ndarray,
    narrative_indicator: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> dict:
    """Diagnostic-only report: the legacy `1 - p` number side by side
    with the raw-rate bootstrap-stability fraction and the current
    production phi-based CI-lower-bound effect size, all on the same
    underlying per-episode data, so the divergence between them is
    visible for migration write-ups. No pass/fail gate; not used by
    any production decision."""
    joint_indicator = (
        np.asarray(behavioral_indicator).astype(bool)
        & np.asarray(narrative_indicator).astype(bool)
    ).astype(int)
    naive = naive_one_minus_p_confidence(fisher_p_value)
    raw_rate_stability = bootstrap_domain_stability(joint_indicator, n_bootstrap, seed)
    phi = _phi_coefficient(behavioral_indicator, narrative_indicator)
    ci_lower, ci_upper = _bootstrap_phi_ci(
        behavioral_indicator, narrative_indicator,
        n_bootstrap=n_bootstrap, ci=0.95, seed=seed,
    )
    return {
        "naive_one_minus_p": naive,
        "bootstrap_stability_raw_rate": raw_rate_stability,
        "phi_ci_lower": max(0.0, ci_lower),
        "divergence": abs(naive - max(0.0, ci_lower)),
    }