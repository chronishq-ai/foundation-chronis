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
import logging
import numpy as np
from dataclasses import dataclass
from scipy.stats import fisher_exact

from domain_emergence.multiple_comparisons import compare_corrections

NOISE = -1

logger = logging.getLogger(__name__)


class AlignmentKeyMismatchError(ValueError):
    """Raised (S56.7) when behavioral_labels/narrative_labels/episode_ids
    are not a valid 1:1 index-aligned join. Explicit exception, not a
    bare `assert` (PH0.1/PH0.2 pattern -- must not be compiled out under
    python -O)."""


@dataclass
class AlignmentResult:
    pair_pvalues: dict              # {(behavioral_id, narrative_id): corrected_p} -- Bonferroni, as before
    raw_pvalues: dict               # {(behavioral_id, narrative_id): uncorrected_p} -- for multiple_comparisons.py
    joint_domains: list             # [(behavioral_id, narrative_id), ...] significant pairs
    high_ignorance_prior: list      # behavioral ids with no significant narrative partner
    aspirational_or_hypothetical: list  # narrative ids with no significant behavioral partner
    n_tests: int
    episode_ids: np.ndarray | None = None   # immutable join key, S56.7
    naive_pvalues: dict | None = None       # S56.5: per-episode Fisher's p, kept for comparison
                                             # when subject_ids swaps the decision path to
                                             # subject-level p-values (None when subject_ids
                                             # omitted -- naive_pvalues == raw_pvalues then)
    pvalue_method: str = "per_episode_fisher_exact"  # or "subject_level_cluster_bootstrap"


def _contingency_table(behavioral_labels: np.ndarray, narrative_labels: np.ndarray,
                        b_id: int, n_id: int) -> list:
    both = int(np.sum((behavioral_labels == b_id) & (narrative_labels == n_id)))
    b_only = int(np.sum((behavioral_labels == b_id) & (narrative_labels != n_id)))
    n_only = int(np.sum((behavioral_labels != b_id) & (narrative_labels == n_id)))
    neither = int(np.sum((behavioral_labels != b_id) & (narrative_labels != n_id)))
    return [[both, b_only], [n_only, neither]]


def _subject_level_pvalue(
    behavioral_labels: np.ndarray, narrative_labels: np.ndarray,
    subject_ids: np.ndarray, b_id: int, n_id: int,
    n_bootstrap: int = 1000, seed: int | None = None,
) -> float:
    """S56.5 FIX -- implemented per user request, ships WITHOUT the
    Mandatory-adjacent senior sign-off the pack calls for on the
    dependence-correction METHOD CHOICE (block permutation vs. cluster
    bootstrap vs. mixed-effects vs. subject-level aggregation -- this
    picks subject-level cluster bootstrap, one of the pack's four named
    options). DO NOT MERGE without that review.

    Cluster-bootstrap p-value for one (b_id, n_id) pair: resample whole
    SUBJECTS (with replacement), not individual episodes/windows, so
    repeated same-person windows are never treated as independent draws.
    p = P(bootstrap co-occurrence count >= observed count | resampling
    respects subject boundaries). This is the same logic
    subject_level_dependence_diagnostic already used for one pair --
    factored out here so align_domains can call it for every pair in
    the decision path, not just as a one-off diagnostic."""
    table = _contingency_table(behavioral_labels, narrative_labels, b_id, n_id)
    observed_stat = table[0][0]

    unique_subjects = np.unique(subject_ids)
    rng = np.random.default_rng(seed)
    boot_stats = []
    for _ in range(n_bootstrap):
        sampled_subjects = unique_subjects[rng.integers(0, len(unique_subjects), size=len(unique_subjects))]
        idx = np.concatenate([np.where(subject_ids == s)[0] for s in sampled_subjects])
        b_resampled = behavioral_labels[idx]
        n_resampled = narrative_labels[idx]
        co_occur = int(np.sum((b_resampled == b_id) & (n_resampled == n_id)))
        boot_stats.append(co_occur)

    boot_stats = np.asarray(boot_stats)
    return float(np.mean(boot_stats >= observed_stat))


def align_domains(
    behavioral_labels: np.ndarray,
    narrative_labels: np.ndarray,
    alpha: float = 0.05,
    episode_ids: np.ndarray | None = None,
    subject_ids: np.ndarray | None = None,
    subject_level_n_bootstrap: int = 1000,
    subject_level_seed: int | None = None,
    log_correction_comparison: bool = True,
) -> AlignmentResult:
    """Run Fisher's exact test, Bonferroni-corrected, over every
    (behavioral_cluster, narrative_topic) pair. Noise (-1) excluded from
    candidate sets on both sides.

    episode_ids (S56.7): optional immutable join key, same length as the
    label arrays. When supplied, enforces a real 1:1 alignment (each
    episode_id appears exactly once) instead of relying on caller
    discipline that the two label arrays happen to already be in the
    same order. Raises AlignmentKeyMismatchError (not a bare assert) on
    any mismatch. When omitted, behavior is unchanged (caller-discipline
    positional alignment, as before) -- this is intentionally backward
    compatible.

    subject_ids (S56.5 -- NOW DRIVES THE DECISION PATH, senior sign-off
    NOT yet obtained, see `_subject_level_pvalue` docstring): optional,
    same length as the label arrays. When supplied, `pair_pvalues` /
    `raw_pvalues` / `joint_domains` are computed from a SUBJECT-LEVEL
    CLUSTER-BOOTSTRAP p-value per pair (resampling whole subjects, not
    individual episodes) instead of the naive per-episode Fisher's
    exact test -- fixing the repeated-same-person-window independence
    violation the naive test has. The naive Fisher's p is still
    computed and returned separately in `naive_pvalues` for comparison/
    logging, but no longer drives which pairs become joint domains.
    `pvalue_method` on the result reports which path was used. When
    omitted (default), behavior is unchanged: naive per-episode
    Fisher's exact drives the decision, exactly as before -- backward
    compatible for callers without subject identifiers.

    log_correction_comparison (S56.10): when True (default), also runs
    Benjamini-Hochberg alongside Bonferroni on the same raw p-values and
    logs the comparison -- so both corrections are actually run and
    visible, not left as an unused alternative. Does not change which
    correction `pair_pvalues`/`joint_domains` uses (still Bonferroni,
    per doctrine's literal spec)."""
    behavioral_labels = np.asarray(behavioral_labels)
    narrative_labels = np.asarray(narrative_labels)
    if len(behavioral_labels) != len(narrative_labels):
        # Explicit exception, not `assert` -- must not vanish under
        # python -O / -OO (PH0.1/PH0.2 pattern).
        raise AssertionError(
            "behavioral_labels and narrative_labels must be index-aligned, same length"
        )

    if episode_ids is not None:
        episode_ids = np.asarray(episode_ids)
        if len(episode_ids) != len(behavioral_labels):
            raise AlignmentKeyMismatchError(
                "episode_ids must be the same length as behavioral_labels/narrative_labels"
            )
        unique_ids, counts = np.unique(episode_ids, return_counts=True)
        dupes = unique_ids[counts > 1]
        if len(dupes) > 0:
            raise AlignmentKeyMismatchError(
                f"episode_ids must be 1:1 (unique) -- duplicate episode_id(s): {dupes.tolist()}"
            )

    if subject_ids is not None:
        subject_ids = np.asarray(subject_ids)
        if len(subject_ids) != len(behavioral_labels):
            raise ValueError(
                "subject_ids must be the same length as behavioral_labels/narrative_labels"
            )

    behavioral_ids = sorted(set(behavioral_labels.tolist()) - {NOISE})
    narrative_ids = sorted(set(narrative_labels.tolist()) - {NOISE})

    n_tests = len(behavioral_ids) * len(narrative_ids)
    pair_pvalues = {}
    raw_pvalues = {}
    naive_pvalues = {}
    joint_domains = []
    sig_behavioral = set()
    sig_narrative = set()
    pvalue_method = (
        "subject_level_cluster_bootstrap" if subject_ids is not None
        else "per_episode_fisher_exact"
    )

    if n_tests == 0:
        return AlignmentResult(
            pair_pvalues={},
            raw_pvalues={},
            joint_domains=[],
            high_ignorance_prior=behavioral_ids,
            aspirational_or_hypothetical=narrative_ids,
            n_tests=0,
            episode_ids=episode_ids,
            naive_pvalues={},
            pvalue_method=pvalue_method,
        )

    for b_id in behavioral_ids:
        for n_id in narrative_ids:
            table = _contingency_table(behavioral_labels, narrative_labels, b_id, n_id)
            _, naive_p = fisher_exact(table, alternative="greater")
            naive_pvalues[(b_id, n_id)] = naive_p

            if subject_ids is not None:
                p_raw = _subject_level_pvalue(
                    behavioral_labels, narrative_labels, subject_ids, b_id, n_id,
                    n_bootstrap=subject_level_n_bootstrap, seed=subject_level_seed,
                )
            else:
                p_raw = naive_p

            p_corrected = min(p_raw * n_tests, 1.0)
            pair_pvalues[(b_id, n_id)] = p_corrected
            raw_pvalues[(b_id, n_id)] = p_raw

            if p_corrected < alpha:
                joint_domains.append((b_id, n_id))
                sig_behavioral.add(b_id)
                sig_narrative.add(n_id)

    high_ignorance_prior = [b for b in behavioral_ids if b not in sig_behavioral]
    aspirational_or_hypothetical = [n for n in narrative_ids if n not in sig_narrative]

    if log_correction_comparison and raw_pvalues:
        # S56.10: run BH alongside Bonferroni on the SAME raw p-values and
        # log it, so both are actually run/visible rather than left as an
        # unused, un-exercised alternative. Decision path (joint_domains
        # above) is untouched -- still Bonferroni, per doctrine. Uses
        # whichever raw_pvalues drove the decision (subject-level if
        # subject_ids was supplied, naive otherwise), so the BH/Bonferroni
        # comparison stays consistent with what actually gated joint_domains.
        comparison = compare_corrections(raw_pvalues, alpha=alpha)
        logger.info(
            "domain_alignment correction comparison (pvalue_method=%s): "
            "bonferroni_significant=%s bh_significant=%s only_bh_finds=%s "
            "only_bonferroni_finds=%s",
            pvalue_method,
            sorted(comparison.bonferroni_significant),
            sorted(comparison.bh_significant),
            sorted(comparison.only_bh_finds),
            sorted(comparison.only_bonferroni_finds),
        )

    return AlignmentResult(
        pair_pvalues=pair_pvalues,
        raw_pvalues=raw_pvalues,
        joint_domains=joint_domains,
        high_ignorance_prior=high_ignorance_prior,
        aspirational_or_hypothetical=aspirational_or_hypothetical,
        n_tests=n_tests,
        episode_ids=episode_ids,
        naive_pvalues=naive_pvalues,
        pvalue_method=pvalue_method,
    )


def subject_level_dependence_diagnostic(
    behavioral_labels: np.ndarray,
    narrative_labels: np.ndarray,
    subject_ids: np.ndarray,
    b_id: int,
    n_id: int,
    n_bootstrap: int = 1000,
    seed: int | None = None,
) -> dict:
    """Diagnostic wrapper around `_subject_level_pvalue`: compares the
    naive per-episode Fisher's-exact p-value for one (b_id, n_id) pair
    against the subject-level cluster-bootstrap p that `align_domains`
    now uses in its decision path when `subject_ids` is supplied (S56.5).
    Reports how anti-conservative (too-small) the naive p-value was."""
    behavioral_labels = np.asarray(behavioral_labels)
    narrative_labels = np.asarray(narrative_labels)
    subject_ids = np.asarray(subject_ids)

    table = _contingency_table(behavioral_labels, narrative_labels, b_id, n_id)
    _, naive_p = fisher_exact(table, alternative="greater")

    subject_level_p = _subject_level_pvalue(
        behavioral_labels, narrative_labels, subject_ids, b_id, n_id,
        n_bootstrap=n_bootstrap, seed=seed,
    )

    return {
        "naive_fisher_p": float(naive_p),
        "subject_level_bootstrap_p": subject_level_p,
        "naive_is_anti_conservative": bool(naive_p < subject_level_p),
        "n_subjects": len(np.unique(subject_ids)),
    }