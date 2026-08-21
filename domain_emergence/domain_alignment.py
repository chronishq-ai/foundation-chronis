"""
Day 17 -- Behavioral x Narrative domain alignment (Bible Part 5.8, stage 3).

For each (behavioral_cluster, narrative_topic) pair, tests co-occurrence
significance via Fisher's exact test on episodes, Bonferroni-corrected for
the number of pairs tested, threshold p < 0.05. Outcome per doctrine:

  - significant co-occurrence -> JOINT DOMAIN
  - behavioral candidate, no significant narrative partner -> HIGH IGNORANCE PRIOR
  - narrative candidate, no significant behavioral partner  -> ASPIRATIONAL OR HYPOTHETICAL

Noise labels (-1, from HDBSCAN and from silent/no-cluster narrative episodes)
are never treated as domain candidates on either side.

Input contract: behavioral_labels and narrative_labels must be the SAME
LENGTH and index-aligned to the same episode ordering (caller's
responsibility -- this module does not itself re-align mismatched episode
sets from context_clustering.py's kept_episodes vs narrative_topics.py's
partial_fit order).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.stats import fisher_exact

NOISE = -1


@dataclass
class AlignmentResult:
    pair_pvalues: dict              # {(behavioral_id, narrative_id): corrected_p} -- Bonferroni, as before
    raw_pvalues: dict               # {(behavioral_id, narrative_id): uncorrected_p} -- for multiple_comparisons.py
    joint_domains: list             # [(behavioral_id, narrative_id), ...] significant pairs
    high_ignorance_prior: list      # behavioral ids with no significant narrative partner
    aspirational_or_hypothetical: list  # narrative ids with no significant behavioral partner
    n_tests: int


def _contingency_table(behavioral_labels: np.ndarray, narrative_labels: np.ndarray,
                        b_id: int, n_id: int) -> list:
    both = int(np.sum((behavioral_labels == b_id) & (narrative_labels == n_id)))
    b_only = int(np.sum((behavioral_labels == b_id) & (narrative_labels != n_id)))
    n_only = int(np.sum((behavioral_labels != b_id) & (narrative_labels == n_id)))
    neither = int(np.sum((behavioral_labels != b_id) & (narrative_labels != n_id)))
    return [[both, b_only], [n_only, neither]]


def align_domains(
    behavioral_labels: np.ndarray,
    narrative_labels: np.ndarray,
    alpha: float = 0.05,
) -> AlignmentResult:
    """Run Fisher's exact test, Bonferroni-corrected, over every
    (behavioral_cluster, narrative_topic) pair. Noise (-1) excluded from
    candidate sets on both sides."""
    behavioral_labels = np.asarray(behavioral_labels)
    narrative_labels = np.asarray(narrative_labels)
    assert len(behavioral_labels) == len(narrative_labels), (
        "behavioral_labels and narrative_labels must be index-aligned, same length"
    )

    behavioral_ids = sorted(set(behavioral_labels.tolist()) - {NOISE})
    narrative_ids = sorted(set(narrative_labels.tolist()) - {NOISE})

    n_tests = len(behavioral_ids) * len(narrative_ids)
    pair_pvalues = {}
    raw_pvalues = {}
    joint_domains = []
    sig_behavioral = set()
    sig_narrative = set()

    if n_tests == 0:
        return AlignmentResult(
            pair_pvalues={},
            raw_pvalues={},
            joint_domains=[],
            high_ignorance_prior=behavioral_ids,
            aspirational_or_hypothetical=narrative_ids,
            n_tests=0,
        )

    for b_id in behavioral_ids:
        for n_id in narrative_ids:
            table = _contingency_table(behavioral_labels, narrative_labels, b_id, n_id)
            _, p_raw = fisher_exact(table, alternative="greater")
            p_corrected = min(p_raw * n_tests, 1.0)
            pair_pvalues[(b_id, n_id)] = p_corrected
            raw_pvalues[(b_id, n_id)] = p_raw

            if p_corrected < alpha:
                joint_domains.append((b_id, n_id))
                sig_behavioral.add(b_id)
                sig_narrative.add(n_id)

    high_ignorance_prior = [b for b in behavioral_ids if b not in sig_behavioral]
    aspirational_or_hypothetical = [n for n in narrative_ids if n not in sig_narrative]

    return AlignmentResult(
        pair_pvalues=pair_pvalues,
        raw_pvalues=raw_pvalues,
        joint_domains=joint_domains,
        high_ignorance_prior=high_ignorance_prior,
        aspirational_or_hypothetical=aspirational_or_hypothetical,
        n_tests=n_tests,
    )