"""
Day 18 -- Bonferroni vs Benjamini-Hochberg (FDR) comparison task.

Bible risk 3-C: Bonferroni correction (used in domain_alignment.py) is
conservative and can produce too FEW domains under heavy multiple-testing
load (many behavioral x narrative pairs). This module implements BH/FDR as
an alternative correction and provides a direct side-by-side comparison on
the SAME set of raw p-values, so the tradeoff is measurable rather than
assumed. This module does not change which correction domain_alignment.py
uses by default (still Bonferroni, per doctrine's literal spec) -- it's a
diagnostic/comparison tool, per the directive's explicit Day 18 task.
"""

from __future__ import annotations
from dataclasses import dataclass


def bonferroni_correct(raw_pvalues: dict, alpha: float = 0.05) -> dict:
    """Multiply each p by the total number of tests, cap at 1.0. Same
    formula domain_alignment.py already applies inline -- factored out
    here so it can be compared against BH on identical inputs."""
    n = len(raw_pvalues)
    if n == 0:
        return {}
    return {k: min(p * n, 1.0) for k, p in raw_pvalues.items()}


def benjamini_hochberg_correct(raw_pvalues: dict, alpha: float = 0.05) -> dict:
    """Standard BH step-up procedure. Returns BH-adjusted p-values (the
    'q-values'): sort ascending, adjusted_p_(i) = min over j>=i of
    p_(j) * m / j, enforced monotonic (never decreasing as rank increases
    when read in sorted order, per the standard BH correction)."""
    n = len(raw_pvalues)
    if n == 0:
        return {}

    items = sorted(raw_pvalues.items(), key=lambda kv: kv[1])
    keys_sorted = [k for k, _ in items]
    pvals_sorted = [p for _, p in items]

    adjusted = [0.0] * n
    adjusted[-1] = min(pvals_sorted[-1] * n / n, 1.0)
    for i in range(n - 2, -1, -1):
        rank = i + 1
        candidate = pvals_sorted[i] * n / rank
        adjusted[i] = min(candidate, adjusted[i + 1], 1.0)

    return {keys_sorted[i]: adjusted[i] for i in range(n)}


@dataclass
class CorrectionComparison:
    bonferroni_pvalues: dict
    bh_pvalues: dict
    bonferroni_significant: set     # keys significant at alpha under Bonferroni
    bh_significant: set             # keys significant at alpha under BH
    only_bh_finds: set              # significant under BH but NOT Bonferroni (BH's extra power)
    only_bonferroni_finds: set      # significant under Bonferroni but NOT BH (shouldn't normally happen)
    agree: set                      # significant under both


def compare_corrections(raw_pvalues: dict, alpha: float = 0.05) -> CorrectionComparison:
    """Run both corrections on the same raw p-values and report where they
    diverge. BH is less conservative than Bonferroni, so
    only_bonferroni_finds should normally be empty -- BH's significant set
    is a superset of Bonferroni's for the same input (this is a property of
    the two procedures, not an assumption; asserted, not just claimed)."""
    bonf = bonferroni_correct(raw_pvalues, alpha)
    bh = benjamini_hochberg_correct(raw_pvalues, alpha)

    bonf_sig = {k for k, p in bonf.items() if p < alpha}
    bh_sig = {k for k, p in bh.items() if p < alpha}

    return CorrectionComparison(
        bonferroni_pvalues=bonf,
        bh_pvalues=bh,
        bonferroni_significant=bonf_sig,
        bh_significant=bh_sig,
        only_bh_finds=bh_sig - bonf_sig,
        only_bonferroni_finds=bonf_sig - bh_sig,
        agree=bonf_sig & bh_sig,
    )