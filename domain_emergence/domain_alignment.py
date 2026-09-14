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

Input contract, two supported alignment modes:

  1. Legacy positional mode (default, backward compatible): caller
     guarantees behavioral_labels/narrative_labels are already the SAME
     LENGTH and index-aligned to the same episode ordering. The optional
     `episode_ids` array in this mode is validated for length/uniqueness
     only -- it does NOT reorder anything and does NOT protect against a
     silent permutation of one side relative to the other (R2-S56.6:
     this was previously the ONLY join behavior even when episode_ids
     was supplied, which is why it was a false sense of safety).

  2. Real keyed-join mode (R2-S56.6 fix, use this whenever the two
     modalities may not already share one positional order -- e.g.
     context_clustering.py's kept_episodes vs narrative_topics.py's
     partial_fit order): supply BOTH `behavioral_episode_ids` and
     `narrative_episode_ids`. The module then joins the two label
     streams by episode_id itself, rather than trusting caller-supplied
     positional order. See `_join_by_episode_id` below.
"""

from __future__ import annotations
import logging
import numpy as np
from dataclasses import dataclass
from scipy.stats import fisher_exact

from domain_emergence.multiple_comparisons import compare_corrections

NOISE = -1
MISSING_MODALITY = -999  # R2-S56.6 outer-join follow-up: sentinel for "this
                          # episode exists on one side of the join but this
                          # modality has no data for it" -- distinct from
                          # NOISE (-1), which means data IS present but
                          # HDBSCAN/the topic model assigned no cluster/topic.
                          # Conflating the two would silently treat "we never
                          # measured this" the same as "we measured it and it
                          # doesn't fit any cluster," which is a real
                          # information loss the doc's outer-join ask is
                          # about preventing.

logger = logging.getLogger(__name__)


class AlignmentKeyMismatchError(ValueError):
    """Raised when behavioral_labels/narrative_labels/episode_ids are not
    a valid 1:1 index-aligned join (legacy mode) or not a valid 1:1 keyed
    join (R2-S56.6 keyed-join mode: duplicate id on either side, id
    present on one side but missing on the other when allow_outer_join
    is False). Explicit exception, not a bare `assert` (PH0.1/PH0.2
    pattern -- must not be compiled out under python -O)."""


@dataclass
class AlignmentResult:
    pair_pvalues: dict              # {(behavioral_id, narrative_id): corrected_p} -- Bonferroni, as before
    raw_pvalues: dict               # {(behavioral_id, narrative_id): uncorrected_p} -- for multiple_comparisons.py
    joint_domains: list             # [(behavioral_id, narrative_id), ...] significant pairs
    high_ignorance_prior: list      # behavioral ids with no significant narrative partner
    aspirational_or_hypothetical: list  # narrative ids with no significant behavioral partner
    n_tests: int
    episode_ids: np.ndarray | None = None   # immutable join key actually used to order the
                                             # data that produced this result (legacy positional
                                             # array if that mode was used, or the real joined-id
                                             # ordering if keyed-join mode was used -- R2-S56.6)
    join_mode: str = "legacy_positional"    # "legacy_positional" or "keyed_join" (R2-S56.6)
    naive_pvalues: dict | None = None       # S56.5: per-episode Fisher's p, kept for comparison
                                             # when subject_ids swaps the decision path to
                                             # subject-level p-values (None when subject_ids
                                             # omitted -- naive_pvalues == raw_pvalues then)
    pvalue_method: str = "per_episode_fisher_exact"  # or "subject_level_within_subject_permutation"
    outer_join_missing_counts: dict | None = None
    # R2-S56.6 outer-join follow-up. None unless full_outer_join=True was
    # used. When set: {"behavioral_only": n, "narrative_only": n} -- counts
    # of episode_ids present on only one side of the join, i.e. missing the
    # OTHER modality entirely. These episodes ARE included in `episode_ids`
    # (tagged MISSING_MODALITY on the missing side) for transparency/
    # downstream inspection, but are EXCLUDED from the significance/
    # decision pipeline below (pair_pvalues, joint_domains,
    # high_ignorance_prior, aspirational_or_hypothetical are all computed
    # on complete cases only -- an episode missing either modality cannot
    # be placed in any 2x2 cell for ANY pair without guessing). This is a
    # scope-limited fix: it surfaces missing-modality episodes rather than
    # silently dropping them, but does NOT implement a separate "missing
    # data" outcome bucket distinct from noise/no-signal at the Bible-
    # doctrine level -- that three-way-vs-four-way outcome taxonomy
    # question is a domain-modeling decision, tracked separately, not
    # assumed here.
    n_complete_cases: int | None = None
    # R2-S56.6 outer-join follow-up. None unless full_outer_join=True.
    # Number of episodes with BOTH modalities present that actually drove
    # pair_pvalues/joint_domains -- lets a caller sanity-check how much of
    # the outer-joined episode set the decision is actually based on.


def _contingency_table(behavioral_labels: np.ndarray, narrative_labels: np.ndarray,
                        b_id: int, n_id: int) -> list:
    both = int(np.sum((behavioral_labels == b_id) & (narrative_labels == n_id)))
    b_only = int(np.sum((behavioral_labels == b_id) & (narrative_labels != n_id)))
    n_only = int(np.sum((behavioral_labels != b_id) & (narrative_labels == n_id)))
    neither = int(np.sum((behavioral_labels != b_id) & (narrative_labels != n_id)))
    return [[both, b_only], [n_only, neither]]


def _join_by_episode_id(
    behavioral_labels: np.ndarray,
    narrative_labels: np.ndarray,
    behavioral_episode_ids: np.ndarray,
    narrative_episode_ids: np.ndarray,
    subject_ids: np.ndarray | None,
    allow_outer_join: bool,
    full_outer_join: bool = False,
) -> tuple:
    """R2-S56.6 FIX -- real keyed join on episode_id, replacing the prior
    behavior where episode_ids were validated for uniqueness but the two
    label arrays were still just assumed to already be positionally
    aligned (so a permutation of one side relative to the other produced
    a silently-wrong, valid-looking result).

    behavioral_episode_ids / narrative_episode_ids: 1:1 (unique) id
    arrays, same length as behavioral_labels / narrative_labels
    respectively. subject_ids, if supplied, must be positionally aligned
    to behavioral_episode_ids (i.e. subject_ids[i] is the subject for
    behavioral_episode_ids[i]) -- it is carried through the same join.

    Returns (joined_episode_ids, joined_behavioral_labels,
    joined_narrative_labels, joined_subject_ids_or_None,
    missing_counts), all ordered by sorted joined_episode_ids so the
    result is deterministic regardless of input order (permutation-
    invariant). missing_counts is
    {"behavioral_only": n, "narrative_only": n} -- always returned
    (zeros when the two id sets already match exactly).

    Raises AlignmentKeyMismatchError on:
      - duplicate id on either side
      - length mismatch between an id array and its label array
      - an id present on one side but not the other, when neither
        allow_outer_join nor full_outer_join is True (the default --
        strict inner join)

    Three join modes for a one-sided id mismatch:
      - allow_outer_join=True, full_outer_join=False (default when
        relaxing the strict check): INNER join on the intersection --
        non-overlapping ids are silently dropped (logged as a warning).
        This does NOT implement "missing modality recorded as NULL";
        it only relaxes "reject any mismatch" to "drop the mismatch."
      - full_outer_join=True (R2-S56.6 outer-join follow-up, new):
        FULL OUTER join on the union of ids -- every id from either
        side is kept. An id missing from one side gets MISSING_MODALITY
        on that side rather than being dropped. This is the "explicit
        NULL-for-missing-modality" semantics the audit asked for.
      - both True at once: contradictory (one says drop missing, the
        other says keep missing as NULL) -- raises ValueError.
    """
    if allow_outer_join and full_outer_join:
        raise ValueError(
            "allow_outer_join=True and full_outer_join=True are "
            "contradictory: allow_outer_join drops non-overlapping "
            "episode_ids (inner join), full_outer_join keeps them "
            "tagged MISSING_MODALITY (outer join). Pass at most one."
        )

    behavioral_labels = np.asarray(behavioral_labels)
    narrative_labels = np.asarray(narrative_labels)
    behavioral_episode_ids = np.asarray(behavioral_episode_ids)
    narrative_episode_ids = np.asarray(narrative_episode_ids)

    if len(behavioral_episode_ids) != len(behavioral_labels):
        raise AlignmentKeyMismatchError(
            "behavioral_episode_ids must be the same length as behavioral_labels "
            f"({len(behavioral_episode_ids)} != {len(behavioral_labels)})"
        )
    if len(narrative_episode_ids) != len(narrative_labels):
        raise AlignmentKeyMismatchError(
            "narrative_episode_ids must be the same length as narrative_labels "
            f"({len(narrative_episode_ids)} != {len(narrative_labels)})"
        )
    if subject_ids is not None and len(subject_ids) != len(behavioral_labels):
        raise AlignmentKeyMismatchError(
            "subject_ids must be positionally aligned to behavioral_episode_ids "
            f"({len(subject_ids)} != {len(behavioral_labels)})"
        )

    def _check_unique(ids: np.ndarray, side: str) -> None:
        unique_ids, counts = np.unique(ids, return_counts=True)
        dupes = unique_ids[counts > 1]
        if len(dupes) > 0:
            raise AlignmentKeyMismatchError(
                f"{side}_episode_ids must be 1:1 (unique) -- duplicate "
                f"episode_id(s): {dupes.tolist()}"
            )

    _check_unique(behavioral_episode_ids, "behavioral")
    _check_unique(narrative_episode_ids, "narrative")

    b_id_set = set(behavioral_episode_ids.tolist())
    n_id_set = set(narrative_episode_ids.tolist())
    common_ids = b_id_set & n_id_set
    behavioral_only = b_id_set - n_id_set
    narrative_only = n_id_set - b_id_set
    missing_counts = {
        "behavioral_only": len(behavioral_only),
        "narrative_only": len(narrative_only),
    }

    if (behavioral_only or narrative_only) and not (allow_outer_join or full_outer_join):
        raise AlignmentKeyMismatchError(
            "episode_id sets do not match between behavioral and narrative "
            f"streams -- {len(behavioral_only)} id(s) only on the behavioral "
            f"side, {len(narrative_only)} id(s) only on the narrative side. "
            "Pass allow_outer_join=True to inner-join on the intersection "
            "(dropping the non-overlapping ids), or full_outer_join=True to "
            "keep them tagged MISSING_MODALITY, or fix the upstream episode "
            "sets so they actually match."
        )

    behavioral_map = dict(zip(behavioral_episode_ids.tolist(), behavioral_labels.tolist()))
    narrative_map = dict(zip(narrative_episode_ids.tolist(), narrative_labels.tolist()))
    subject_map = (
        dict(zip(behavioral_episode_ids.tolist(), subject_ids.tolist()))
        if subject_ids is not None else None
    )

    if full_outer_join:
        joined_ids = np.array(sorted(b_id_set | n_id_set))
        joined_behavioral = np.array(
            [behavioral_map.get(i, MISSING_MODALITY) for i in joined_ids.tolist()]
        )
        joined_narrative = np.array(
            [narrative_map.get(i, MISSING_MODALITY) for i in joined_ids.tolist()]
        )
        joined_subject = (
            np.array([subject_map.get(i, MISSING_MODALITY) for i in joined_ids.tolist()])
            if subject_map is not None else None
        )
        return joined_ids, joined_behavioral, joined_narrative, joined_subject, missing_counts

    if behavioral_only or narrative_only:
        logger.warning(
            "align_domains keyed join: dropping %d behavioral-only and %d "
            "narrative-only episode_id(s) not present on both sides "
            "(allow_outer_join=True inner-join mode)",
            len(behavioral_only), len(narrative_only),
        )

    joined_ids = np.array(sorted(common_ids))

    joined_behavioral = np.array([behavioral_map[i] for i in joined_ids.tolist()])
    joined_narrative = np.array([narrative_map[i] for i in joined_ids.tolist()])
    joined_subject = (
        np.array([subject_map[i] for i in joined_ids.tolist()])
        if subject_map is not None else None
    )

    return joined_ids, joined_behavioral, joined_narrative, joined_subject, missing_counts


def _subject_level_pvalue(
    behavioral_labels: np.ndarray, narrative_labels: np.ndarray,
    subject_ids: np.ndarray, b_id: int, n_id: int,
    n_permutations: int = 1000, seed: int | None = None,
) -> float:
    """R2-S56.2 FIX -- restricted within-subject permutation test,
    replacing the prior subject-level cluster-BOOTSTRAP implementation.

    Why the prior implementation was invalid: it resampled whole subjects
    WITH REPLACEMENT and recomputed the SAME observed-association
    statistic on each resample, then reported the fraction of bootstrap
    counts >= the observed count. That is a bootstrap distribution of
    the observed association itself, not a null distribution of
    independence -- it never breaks the behavioral<->narrative pairing
    the null hypothesis is supposed to remove. On an intentionally
    extreme dataset where every subject shows the same behavioral/
    narrative pair (perfect association), that implementation returned
    p=1.0, which is the wrong direction entirely for a significance test
    meant to detect association.

    This implementation instead constructs a genuine null distribution:
    for each of `n_permutations` replicates, independently permute
    narrative_labels WITHIN each subject's own set of episodes (subject
    membership, each subject's episode count, and each subject's own
    narrative-label marginal distribution are all exactly preserved --
    only the within-subject pairing between a subject's behavioral label
    sequence and their narrative label sequence is destroyed). The same
    co-occurrence statistic used for the observed data is recomputed on
    each permuted replicate; p = P(permuted statistic >= observed
    statistic), with the standard +1/+1 correction (Davison & Hinkley)
    so a finite permutation count never reports an unjustified p=0.

    Exchangeability assumption: episodes within a subject are
    exchangeable under the null (no unmodelled temporal dependence
    within a subject that would itself explain co-occurrence). If that
    assumption is not defensible for a given deployment, a block/
    circular permutation variant should be substituted -- this is a
    Statistics/Research-lead-owned method decision (see module Ownership
    Model notes), not a parameter an intern should flip silently.
    """
    behavioral_labels = np.asarray(behavioral_labels)
    narrative_labels = np.asarray(narrative_labels)
    subject_ids = np.asarray(subject_ids)

    table = _contingency_table(behavioral_labels, narrative_labels, b_id, n_id)
    observed_stat = table[0][0]

    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(subject_ids)
    subject_idx = {s: np.where(subject_ids == s)[0] for s in unique_subjects}

    perm_stats = np.empty(n_permutations, dtype=np.int64)
    for p in range(n_permutations):
        permuted_narrative = narrative_labels.copy()
        for idx in subject_idx.values():
            if len(idx) > 1:
                permuted_narrative[idx] = narrative_labels[idx][rng.permutation(len(idx))]
        perm_stats[p] = int(
            np.sum((behavioral_labels == b_id) & (permuted_narrative == n_id))
        )

    # +1/+1 correction: never report p=0 from a finite Monte Carlo sample.
    return float((np.sum(perm_stats >= observed_stat) + 1) / (n_permutations + 1))


def align_domains(
    behavioral_labels: np.ndarray,
    narrative_labels: np.ndarray,
    alpha: float = 0.05,
    episode_ids: np.ndarray | None = None,
    behavioral_episode_ids: np.ndarray | None = None,
    narrative_episode_ids: np.ndarray | None = None,
    allow_outer_join: bool = False,
    full_outer_join: bool = False,
    subject_ids: np.ndarray | None = None,
    subject_level_n_permutations: int = 1000,
    subject_level_seed: int | None = None,
    log_correction_comparison: bool = True,
) -> AlignmentResult:
    """Run Fisher's exact test, Bonferroni-corrected, over every
    (behavioral_cluster, narrative_topic) pair. Noise (-1) excluded from
    candidate sets on both sides.

    Two alignment modes (see module docstring):

    episode_ids (legacy positional mode, S56.7 original scope): optional
    immutable id array, same length as the (already positionally
    aligned) label arrays. Validated for length + uniqueness only --
    this does NOT re-derive alignment from the ids, it only checks the
    caller's claim that positions are already aligned is at least
    self-consistent (no duplicate/wrong-length ids). Use
    behavioral_episode_ids/narrative_episode_ids instead whenever the
    two modalities are not already guaranteed to share one positional
    order (R2-S56.6).

    behavioral_episode_ids / narrative_episode_ids (real keyed-join
    mode, R2-S56.6 fix): when BOTH are supplied, this function performs
    an actual id-keyed join instead of trusting positional order --
    labels are looked up by episode_id and reassembled in sorted-id
    order, so a permutation of one side relative to the other is
    corrected rather than silently accepted. Raises
    AlignmentKeyMismatchError on duplicate ids on either side, or ids
    present on one side but not the other (unless allow_outer_join=True,
    which inner-joins the intersection and logs the drop -- see
    `_join_by_episode_id` for the exact, currently-inner-join-only,
    semantics of that flag). When this mode is used, subject_ids (if
    supplied) must be positionally aligned to behavioral_episode_ids
    (pre-join) and is carried through the same join.

    Do not supply both `episode_ids` and
    `behavioral_episode_ids`/`narrative_episode_ids` in the same call --
    that's a contradictory request (one says "already aligned", the
    other says "join for me") and raises ValueError.

    subject_ids (S56.5 / R2-S56.2): optional. When supplied,
    `pair_pvalues` / `raw_pvalues` / `joint_domains` are computed from a
    within-subject-permutation p-value per pair (see
    `_subject_level_pvalue`) instead of the naive per-episode Fisher's
    exact test -- fixing the repeated-same-person-window independence
    violation the naive test has, via a statistically valid null
    distribution (not the invalid cluster-bootstrap this replaces). The
    naive Fisher's p is still computed and returned separately in
    `naive_pvalues` for comparison/logging, but no longer drives which
    pairs become joint domains. `pvalue_method` on the result reports
    which path was used. When omitted (default), behavior is unchanged:
    naive per-episode Fisher's exact drives the decision, exactly as
    before -- backward compatible for callers without subject
    identifiers.

    log_correction_comparison (S56.10): when True (default), also runs
    Benjamini-Hochberg alongside Bonferroni on the same raw p-values and
    logs the comparison -- so both corrections are actually run and
    visible, not left as an unused alternative. Does not change which
    correction `pair_pvalues`/`joint_domains` uses (still Bonferroni,
    per doctrine's literal spec).

    full_outer_join (R2-S56.6 outer-join follow-up): requires keyed-join
    mode (behavioral_episode_ids + narrative_episode_ids). When True,
    every episode_id from EITHER side is kept -- an id missing a
    modality gets `MISSING_MODALITY` on that side instead of being
    silently dropped (the `allow_outer_join` flag drops it; the two
    flags are mutually exclusive, see `_join_by_episode_id`). The
    returned `episode_ids`/joined label arrays include these
    missing-modality episodes for transparency. The significance/
    decision pipeline (pair_pvalues, joint_domains, high_ignorance_prior,
    aspirational_or_hypothetical) is computed on COMPLETE CASES ONLY --
    episodes missing either modality cannot be placed in any pair's 2x2
    table without guessing, so they're excluded from every test, not
    just the one nominally "missing." `outer_join_missing_counts` and
    `n_complete_cases` on the result report exactly how many/which
    episodes that affected. This does NOT implement a separate
    "missing data" outcome distinct from noise/no-signal in the
    high_ignorance_prior / aspirational_or_hypothetical taxonomy --
    that's a Bible-doctrine-level modeling question, out of scope here."""
    behavioral_labels = np.asarray(behavioral_labels)
    narrative_labels = np.asarray(narrative_labels)

    keyed_join_requested = (
        behavioral_episode_ids is not None or narrative_episode_ids is not None
    )
    if keyed_join_requested and episode_ids is not None:
        raise ValueError(
            "Pass either `episode_ids` (legacy positional mode) or "
            "`behavioral_episode_ids`+`narrative_episode_ids` (real keyed-"
            "join mode, R2-S56.6), not both -- they express contradictory "
            "claims about whether the caller has already aligned the data."
        )
    if keyed_join_requested and (
        behavioral_episode_ids is None or narrative_episode_ids is None
    ):
        raise ValueError(
            "Keyed-join mode requires BOTH behavioral_episode_ids and "
            "narrative_episode_ids -- supplying only one cannot join anything."
        )
    if full_outer_join and not keyed_join_requested:
        raise ValueError(
            "full_outer_join=True requires keyed-join mode -- supply both "
            "behavioral_episode_ids and narrative_episode_ids so there is "
            "an actual id to outer-join on."
        )

    join_mode = "legacy_positional"
    joined_episode_ids = None
    outer_join_missing_counts = None
    n_complete_cases = None

    if keyed_join_requested:
        join_mode = "keyed_join_outer" if full_outer_join else "keyed_join"
        joined_episode_ids, behavioral_labels, narrative_labels, subject_ids, outer_join_missing_counts = (
            _join_by_episode_id(
                behavioral_labels, narrative_labels,
                behavioral_episode_ids, narrative_episode_ids,
                subject_ids, allow_outer_join, full_outer_join,
            )
        )
        if full_outer_join:
            complete_mask = (
                (behavioral_labels != MISSING_MODALITY)
                & (narrative_labels != MISSING_MODALITY)
            )
            n_complete_cases = int(np.sum(complete_mask))
            behavioral_labels = behavioral_labels[complete_mask]
            narrative_labels = narrative_labels[complete_mask]
            if subject_ids is not None:
                subject_ids = subject_ids[complete_mask]
    else:
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
        joined_episode_ids = episode_ids

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
        "subject_level_within_subject_permutation" if subject_ids is not None
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
            episode_ids=joined_episode_ids,
            join_mode=join_mode,
            naive_pvalues={},
            pvalue_method=pvalue_method,
            outer_join_missing_counts=outer_join_missing_counts,
            n_complete_cases=n_complete_cases,
        )

    for b_id in behavioral_ids:
        for n_id in narrative_ids:
            table = _contingency_table(behavioral_labels, narrative_labels, b_id, n_id)
            _, naive_p = fisher_exact(table, alternative="greater")
            naive_pvalues[(b_id, n_id)] = naive_p

            if subject_ids is not None:
                p_raw = _subject_level_pvalue(
                    behavioral_labels, narrative_labels, subject_ids, b_id, n_id,
                    n_permutations=subject_level_n_permutations, seed=subject_level_seed,
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
        episode_ids=joined_episode_ids,
        join_mode=join_mode,
        naive_pvalues=naive_pvalues,
        pvalue_method=pvalue_method,
        outer_join_missing_counts=outer_join_missing_counts,
        n_complete_cases=n_complete_cases,
    )


def subject_level_dependence_diagnostic(
    behavioral_labels: np.ndarray,
    narrative_labels: np.ndarray,
    subject_ids: np.ndarray,
    b_id: int,
    n_id: int,
    n_permutations: int = 1000,
    seed: int | None = None,
) -> dict:
    """Diagnostic wrapper around `_subject_level_pvalue`: compares the
    naive per-episode Fisher's-exact p-value for one (b_id, n_id) pair
    against the within-subject-permutation p that `align_domains` now
    uses in its decision path when `subject_ids` is supplied (S56.5 /
    R2-S56.2). Reports how anti-conservative (too-small) the naive
    p-value was."""
    behavioral_labels = np.asarray(behavioral_labels)
    narrative_labels = np.asarray(narrative_labels)
    subject_ids = np.asarray(subject_ids)

    table = _contingency_table(behavioral_labels, narrative_labels, b_id, n_id)
    _, naive_p = fisher_exact(table, alternative="greater")

    subject_level_p = _subject_level_pvalue(
        behavioral_labels, narrative_labels, subject_ids, b_id, n_id,
        n_permutations=n_permutations, seed=seed,
    )

    return {
        "naive_fisher_p": float(naive_p),
        "subject_level_permutation_p": subject_level_p,
        "naive_is_anti_conservative": bool(naive_p < subject_level_p),
        "n_subjects": len(np.unique(subject_ids)),
    }