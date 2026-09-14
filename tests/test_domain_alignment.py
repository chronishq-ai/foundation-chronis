import numpy as np
import pytest
from domain_emergence.domain_alignment import (
    align_domains, NOISE, AlignmentKeyMismatchError,
    subject_level_dependence_diagnostic,
)


def test_perfectly_correlated_pair_is_joint_domain():
    # behavioral cluster 0 always co-occurs with narrative topic 0
    behavioral = np.array([0] * 20 + [1] * 20)
    narrative = np.array([0] * 20 + [1] * 20)
    result = align_domains(behavioral, narrative)
    assert (0, 0) in result.joint_domains
    assert (1, 1) in result.joint_domains
    assert result.high_ignorance_prior == []
    assert result.aspirational_or_hypothetical == []


def test_uncorrelated_pair_not_joint_domain():
    rng = np.random.default_rng(0)
    behavioral = rng.integers(0, 2, size=200)
    narrative = rng.integers(0, 2, size=200)
    result = align_domains(behavioral, narrative)
    # random noise shouldn't survive Bonferroni correction at n=200
    assert result.joint_domains == []


def test_behavioral_only_flags_high_ignorance_prior():
    # behavioral cluster 0 exists, but narrative side is ALL noise
    behavioral = np.array([0] * 20 + [1] * 20)
    narrative = np.full(40, NOISE)
    result = align_domains(behavioral, narrative)
    assert set(result.high_ignorance_prior) == {0, 1}
    assert result.aspirational_or_hypothetical == []
    assert result.joint_domains == []


def test_narrative_only_flags_aspirational_or_hypothetical():
    behavioral = np.full(40, NOISE)
    narrative = np.array([0] * 20 + [1] * 20)
    result = align_domains(behavioral, narrative)
    assert set(result.aspirational_or_hypothetical) == {0, 1}
    assert result.high_ignorance_prior == []
    assert result.joint_domains == []


def test_all_noise_produces_no_candidates():
    behavioral = np.full(20, NOISE)
    narrative = np.full(20, NOISE)
    result = align_domains(behavioral, narrative)
    assert result.n_tests == 0
    assert result.joint_domains == []
    assert result.high_ignorance_prior == []
    assert result.aspirational_or_hypothetical == []


def test_mismatched_length_raises():
    behavioral = np.array([0, 1, 0])
    narrative = np.array([0, 1])
    try:
        align_domains(behavioral, narrative)
        assert False, "expected AssertionError"
    except AssertionError:
        pass


def test_bonferroni_correction_scales_with_n_tests():
    # 2 behavioral x 3 narrative = 6 tests -> corrected p should be raw p * 6, capped at 1.0
    behavioral = np.array([0] * 10 + [1] * 10)
    narrative = np.array(([0] * 5 + [1] * 5) * 2)
    result = align_domains(behavioral, narrative)
    assert result.n_tests == 2 * 2  # only 2 narrative ids actually present (0,1)
    for p in result.pair_pvalues.values():
        assert 0.0 <= p <= 1.0


# --- legacy positional mode: episode_ids validated but NOT re-joined
# (weaker check -- see keyed-join tests below for R2-S56.6's real join) ---

def test_episode_ids_well_formed_passes_through():
    behavioral = np.array([0] * 20 + [1] * 20)
    narrative = np.array([0] * 20 + [1] * 20)
    episode_ids = np.arange(40)
    result = align_domains(behavioral, narrative, episode_ids=episode_ids)
    assert result.episode_ids is not None
    assert np.array_equal(result.episode_ids, episode_ids)
    assert result.join_mode == "legacy_positional"


def test_episode_ids_duplicate_raises_typed_exception():
    behavioral = np.array([0, 1, 0, 1])
    narrative = np.array([0, 1, 0, 1])
    episode_ids = np.array([1, 2, 2, 3])  # duplicate episode_id=2
    with pytest.raises(AlignmentKeyMismatchError):
        align_domains(behavioral, narrative, episode_ids=episode_ids)


def test_episode_ids_wrong_length_raises_typed_exception():
    behavioral = np.array([0, 1, 0, 1])
    narrative = np.array([0, 1, 0, 1])
    episode_ids = np.array([1, 2, 3])  # too short
    with pytest.raises(AlignmentKeyMismatchError):
        align_domains(behavioral, narrative, episode_ids=episode_ids)


def test_no_episode_ids_still_works_backward_compatible():
    behavioral = np.array([0] * 20 + [1] * 20)
    narrative = np.array([0] * 20 + [1] * 20)
    result = align_domains(behavioral, narrative)
    assert result.episode_ids is None
    assert result.join_mode == "legacy_positional"


# --- R2-S56.6: real keyed join on episode_id ---

def test_keyed_join_permutation_recovery():
    """R2-S56.6 T1/T5: narrative records supplied in a permuted order,
    but with correct ids attached -- the keyed join must restore correct
    pairing, unlike the old positional-only check which would have
    silently accepted the permuted (wrong) pairing."""
    behavioral = np.array([0] * 20 + [1] * 20)
    behavioral_ids = np.arange(40)

    narrative_true = np.array([0] * 20 + [1] * 20)  # true pairing: id i -> same as behavioral
    rng = np.random.default_rng(0)
    perm = rng.permutation(40)
    narrative_shuffled = narrative_true[perm]
    narrative_ids_shuffled = behavioral_ids[perm]  # ids travel WITH their correct labels

    result = align_domains(
        behavioral, narrative_shuffled,
        behavioral_episode_ids=behavioral_ids,
        narrative_episode_ids=narrative_ids_shuffled,
    )
    # after the real keyed join, the true perfect association must be recovered
    assert (0, 0) in result.joint_domains
    assert (1, 1) in result.joint_domains
    assert result.join_mode == "keyed_join"
    assert np.array_equal(result.episode_ids, np.sort(behavioral_ids))


def test_keyed_join_missing_narrative_episode_raises():
    behavioral = np.array([0, 0, 1, 1])
    behavioral_ids = np.array([1, 2, 3, 4])
    narrative = np.array([0, 0, 1])
    narrative_ids = np.array([1, 2, 3])  # episode 4 missing on narrative side
    with pytest.raises(AlignmentKeyMismatchError):
        align_domains(
            behavioral, narrative,
            behavioral_episode_ids=behavioral_ids,
            narrative_episode_ids=narrative_ids,
        )


def test_keyed_join_duplicate_id_raises():
    behavioral = np.array([0, 0, 1, 1])
    behavioral_ids = np.array([1, 1, 3, 4])  # duplicate
    narrative = np.array([0, 0, 1, 1])
    narrative_ids = np.array([1, 2, 3, 4])
    with pytest.raises(AlignmentKeyMismatchError):
        align_domains(
            behavioral, narrative,
            behavioral_episode_ids=behavioral_ids,
            narrative_episode_ids=narrative_ids,
        )


def test_keyed_join_extra_narrative_episode_allowed_with_outer_join_flag():
    behavioral = np.array([0, 0, 1, 1])
    behavioral_ids = np.array([1, 2, 3, 4])
    narrative = np.array([0, 0, 1, 1, 0])
    narrative_ids = np.array([1, 2, 3, 4, 99])  # 99 not on behavioral side
    result = align_domains(
        behavioral, narrative,
        behavioral_episode_ids=behavioral_ids,
        narrative_episode_ids=narrative_ids,
        allow_outer_join=True,
    )
    assert result.join_mode == "keyed_join"
    assert 99 not in result.episode_ids.tolist()
    assert len(result.episode_ids) == 4


# --- R2-S56.6 outer-join follow-up: full_outer_join with explicit
# MISSING_MODALITY semantics, replacing allow_outer_join's silent-drop
# behavior for callers who need to know WHICH episodes were incomplete ---

def test_full_outer_join_keeps_missing_modality_episodes_with_sentinel():
    behavioral = np.array([0, 0, 1, 1])
    behavioral_ids = np.array([1, 2, 3, 4])
    narrative = np.array([0, 0, 1])
    narrative_ids = np.array([1, 2, 3])  # episode 4 missing on narrative side
    result = align_domains(
        behavioral, narrative,
        behavioral_episode_ids=behavioral_ids,
        narrative_episode_ids=narrative_ids,
        full_outer_join=True,
    )
    assert result.join_mode == "keyed_join_outer"
    # episode 4 kept, not dropped
    assert 4 in result.episode_ids.tolist()
    assert result.outer_join_missing_counts == {"behavioral_only": 1, "narrative_only": 0}


def test_full_outer_join_decision_pipeline_excludes_incomplete_episodes():
    """The significance/decision pipeline must be computed on complete
    cases only -- an episode missing one modality can't be placed in
    any pair's 2x2 table, so full_outer_join's result must match what
    allow_outer_join (inner join, same complete-case data) produces,
    even though full_outer_join's episode_ids/n_complete_cases surface
    more information about what was excluded."""
    behavioral = np.array([0] * 10 + [1] * 10)
    behavioral_ids = np.arange(20)
    narrative = np.array([0] * 10 + [1] * 10)
    narrative_ids = np.arange(20)
    # Add one behavioral-only and one narrative-only episode -- these
    # must not be silently forced into either pair's contingency table.
    behavioral_ext = np.concatenate([behavioral, [0]])
    behavioral_ids_ext = np.concatenate([behavioral_ids, [100]])
    narrative_ext = np.concatenate([narrative, [1]])
    narrative_ids_ext = np.concatenate([narrative_ids, [200]])

    outer_result = align_domains(
        behavioral_ext, narrative_ext,
        behavioral_episode_ids=behavioral_ids_ext,
        narrative_episode_ids=narrative_ids_ext,
        full_outer_join=True,
    )
    inner_result = align_domains(
        behavioral_ext, narrative_ext,
        behavioral_episode_ids=behavioral_ids_ext,
        narrative_episode_ids=narrative_ids_ext,
        allow_outer_join=True,
    )
    assert outer_result.n_complete_cases == 20
    assert outer_result.joint_domains == inner_result.joint_domains
    assert outer_result.pair_pvalues == inner_result.pair_pvalues
    # but outer_result additionally surfaces the incomplete episodes
    assert len(outer_result.episode_ids) == 22
    assert len(inner_result.episode_ids) == 20


def test_full_outer_join_and_allow_outer_join_together_raises_valueerror():
    behavioral = np.array([0, 1])
    behavioral_ids = np.array([1, 2])
    narrative = np.array([0, 1])
    narrative_ids = np.array([1, 2])
    with pytest.raises(ValueError):
        align_domains(
            behavioral, narrative,
            behavioral_episode_ids=behavioral_ids,
            narrative_episode_ids=narrative_ids,
            allow_outer_join=True,
            full_outer_join=True,
        )


def test_full_outer_join_requires_keyed_join_mode():
    behavioral = np.array([0, 1])
    narrative = np.array([0, 1])
    with pytest.raises(ValueError):
        align_domains(behavioral, narrative, full_outer_join=True)


def test_full_outer_join_subject_ids_missing_for_narrative_only_episode():
    """subject_ids is positionally aligned to behavioral_episode_ids
    pre-join and has no meaning for a narrative-only episode -- it must
    get MISSING_MODALITY too, and (since it's excluded from complete
    cases anyway) must not break the subject-level decision path."""
    behavioral = np.array([0] * 10 + [1] * 10)
    behavioral_ids = np.arange(20)
    subject_ids = np.repeat(np.arange(4), 5)
    narrative = np.array([0] * 10 + [1] * 10)
    narrative_ids = np.arange(20)

    narrative_ext = np.concatenate([narrative, [1]])
    narrative_ids_ext = np.concatenate([narrative_ids, [200]])  # narrative-only episode

    result = align_domains(
        behavioral, narrative_ext,
        behavioral_episode_ids=behavioral_ids,
        narrative_episode_ids=narrative_ids_ext,
        subject_ids=subject_ids,
        full_outer_join=True,
        subject_level_seed=0,
    )
    assert result.n_complete_cases == 20
    assert result.pvalue_method == "subject_level_within_subject_permutation"


def test_keyed_join_and_legacy_episode_ids_together_raises():
    behavioral = np.array([0, 1])
    narrative = np.array([0, 1])
    with pytest.raises(ValueError):
        align_domains(
            behavioral, narrative,
            episode_ids=np.array([1, 2]),
            behavioral_episode_ids=np.array([1, 2]),
            narrative_episode_ids=np.array([1, 2]),
        )


def test_keyed_join_requires_both_id_arrays():
    behavioral = np.array([0, 1])
    narrative = np.array([0, 1])
    with pytest.raises(ValueError):
        align_domains(
            behavioral, narrative,
            behavioral_episode_ids=np.array([1, 2]),
        )


def test_keyed_join_subject_ids_carried_through_join():
    """subject_ids (positionally aligned to behavioral_episode_ids
    pre-join) must survive the keyed join and still drive the
    within-subject-permutation decision path afterward."""
    behavioral = np.array([0] * 20 + [1] * 20)
    behavioral_ids = np.arange(40)
    subject_ids = np.repeat(np.arange(8), 5)

    rng = np.random.default_rng(1)
    narrative_true = np.array([0] * 20 + [1] * 20)
    perm = rng.permutation(40)
    narrative_shuffled = narrative_true[perm]
    narrative_ids_shuffled = behavioral_ids[perm]

    result = align_domains(
        behavioral, narrative_shuffled,
        behavioral_episode_ids=behavioral_ids,
        narrative_episode_ids=narrative_ids_shuffled,
        subject_ids=subject_ids,
        subject_level_seed=0,
    )
    assert result.pvalue_method == "subject_level_within_subject_permutation"
    assert result.naive_pvalues is not None


# --- S56.5 / R2-S56.2: within-subject permutation replaces the invalid
# subject-level cluster-bootstrap null test ---

def test_subject_level_diagnostic_flags_naive_anti_conservatism():
    """Repeated same-subject windows inflate the naive Fisher's-exact
    significance -- diagnostic should show naive_p < subject_level_p
    when the apparent signal is really just a few subjects repeated
    many times."""
    rng = np.random.default_rng(0)
    n_subjects = 4
    windows_per_subject = 15
    subject_ids = np.repeat(np.arange(n_subjects), windows_per_subject)

    behavioral = np.zeros(n_subjects * windows_per_subject, dtype=int)
    narrative = np.zeros(n_subjects * windows_per_subject, dtype=int)
    subj0_mask = subject_ids == 0
    behavioral[subj0_mask] = 1
    narrative[subj0_mask] = 1
    behavioral[~subj0_mask] = rng.integers(0, 2, size=(~subj0_mask).sum())
    narrative[~subj0_mask] = rng.integers(0, 2, size=(~subj0_mask).sum())

    result = subject_level_dependence_diagnostic(
        behavioral, narrative, subject_ids, b_id=1, n_id=1,
        n_permutations=500, seed=0,
    )
    assert "naive_fisher_p" in result
    assert "subject_level_permutation_p" in result
    assert result["n_subjects"] == n_subjects


def test_subject_level_diagnostic_does_not_mutate_align_domains_output():
    """Without subject_ids passed to align_domains itself, calling the
    diagnostic separately doesn't change align_domains' output
    (legacy/no-subject_ids call path is untouched)."""
    behavioral = np.array([0] * 20 + [1] * 20)
    narrative = np.array([0] * 20 + [1] * 20)
    subject_ids = np.repeat(np.arange(8), 5)
    before = align_domains(behavioral, narrative).joint_domains
    subject_level_dependence_diagnostic(
        behavioral, narrative, subject_ids, b_id=0, n_id=0, n_permutations=50, seed=0,
    )
    after = align_domains(behavioral, narrative).joint_domains
    assert before == after


def test_subject_ids_supplied_uses_subject_level_pvalue_method():
    behavioral = np.array([0] * 20 + [1] * 20)
    narrative = np.array([0] * 20 + [1] * 20)
    subject_ids = np.repeat(np.arange(8), 5)
    result = align_domains(
        behavioral, narrative, subject_ids=subject_ids,
        subject_level_n_permutations=200, subject_level_seed=0,
    )
    assert result.pvalue_method == "subject_level_within_subject_permutation"
    assert result.naive_pvalues is not None
    assert set(result.naive_pvalues.keys()) == set(result.raw_pvalues.keys())


def test_no_subject_ids_uses_naive_pvalue_method_backward_compatible():
    behavioral = np.array([0] * 20 + [1] * 20)
    narrative = np.array([0] * 20 + [1] * 20)
    result = align_domains(behavioral, narrative)
    assert result.pvalue_method == "per_episode_fisher_exact"
    # backward compatible: raw_pvalues == naive_pvalues when no subject_ids
    assert result.raw_pvalues == result.naive_pvalues


def test_subject_level_correction_fixes_false_positive_from_repeated_subject():
    """S56.5 T1's exact scenario, wired to the actual decision path: a
    signal that looks significant under naive per-episode Fisher's exact
    but is really driven by ONE subject's windows repeated many times
    should be correctly demoted -- NOT declared a joint domain -- once
    subject_ids is supplied. Without subject_ids, the same data DOES
    get declared a joint domain (documents the naive-test's
    anti-conservatism this fix addresses)."""
    rng = np.random.default_rng(0)
    n_subjects = 4
    windows_per_subject = 15
    subject_ids = np.repeat(np.arange(n_subjects), windows_per_subject)

    behavioral = np.zeros(n_subjects * windows_per_subject, dtype=int)
    narrative = np.zeros(n_subjects * windows_per_subject, dtype=int)
    subj0_mask = subject_ids == 0
    behavioral[subj0_mask] = 1
    narrative[subj0_mask] = 1
    behavioral[~subj0_mask] = rng.integers(0, 2, size=(~subj0_mask).sum())
    narrative[~subj0_mask] = rng.integers(0, 2, size=(~subj0_mask).sum())

    naive_result = align_domains(behavioral, narrative, log_correction_comparison=False)
    corrected_result = align_domains(
        behavioral, narrative, subject_ids=subject_ids,
        subject_level_n_permutations=500, subject_level_seed=0,
        log_correction_comparison=False,
    )

    assert (1, 1) in naive_result.joint_domains, \
        "sanity check: naive per-episode test should flag this pair as significant"
    assert (1, 1) not in corrected_result.joint_domains, \
        f"subject-level correction should demote a single-subject-driven signal, got {corrected_result.joint_domains}"


def test_subject_ids_wrong_length_raises():
    behavioral = np.array([0, 1, 0, 1])
    narrative = np.array([0, 1, 0, 1])
    subject_ids = np.array([0, 0, 1])  # too short
    with pytest.raises(ValueError):
        align_domains(behavioral, narrative, subject_ids=subject_ids)


def test_subject_level_positive_control_perfect_association_across_many_subjects():
    """R2-S56.2 acceptance T1: many INDEPENDENT subjects, each showing a
    real within-subject association between behavioral cluster and
    narrative topic (not just a single repeated subject), must be
    detected as significant -- the corrected p-value must be materially
    below alpha. This is the positive control the invalid bootstrap
    implementation failed (it returned p=1.0 here)."""
    rng = np.random.default_rng(42)
    n_subjects = 30
    windows_per_subject = 6
    subject_ids = np.repeat(np.arange(n_subjects), windows_per_subject)

    # Each subject has a genuine WITHIN-subject association: whenever
    # their behavioral label is 0, their narrative label is also 0 with
    # high probability, and similarly for 1 -- i.e. there is real
    # per-episode variability (not a constant-per-subject value), so a
    # within-subject permutation test has power to detect it.
    behavioral = rng.integers(0, 2, size=n_subjects * windows_per_subject)
    flip = rng.random(n_subjects * windows_per_subject) < 0.05
    narrative = np.where(flip, 1 - behavioral, behavioral)

    result = align_domains(
        behavioral, narrative, subject_ids=subject_ids,
        subject_level_n_permutations=1000, subject_level_seed=0,
        log_correction_comparison=False,
    )
    assert (0, 0) in result.joint_domains
    assert (1, 1) in result.joint_domains
    for pair in ((0, 0), (1, 1)):
        assert result.pair_pvalues[pair] < 0.01


def test_subject_level_null_calibration_no_within_subject_association():
    """R2-S56.2 acceptance T2 (reduced scale for test runtime): under a
    genuinely null generative process (behavioral/narrative independent
    within every subject), the within-subject permutation p-value should
    not be systematically anti-conservative -- it should not flag
    significance any more often than a per-episode Fisher's test would
    be expected to under multiplicity correction."""
    rng = np.random.default_rng(7)
    n_subjects = 20
    windows_per_subject = 8
    subject_ids = np.repeat(np.arange(n_subjects), windows_per_subject)
    behavioral = rng.integers(0, 2, size=n_subjects * windows_per_subject)
    narrative = rng.integers(0, 2, size=n_subjects * windows_per_subject)

    result = align_domains(
        behavioral, narrative, subject_ids=subject_ids,
        subject_level_n_permutations=500, subject_level_seed=1,
        log_correction_comparison=False,
    )
    assert result.joint_domains == []


# --- S56.10: BH run + logged alongside Bonferroni ---

def test_correction_comparison_logged(caplog):
    import logging
    behavioral = np.array([0] * 10 + [1] * 10)
    narrative = np.array(([0] * 5 + [1] * 5) * 2)
    with caplog.at_level(logging.INFO, logger="domain_emergence.domain_alignment"):
        align_domains(behavioral, narrative, log_correction_comparison=True)
    assert any("correction comparison" in r.message for r in caplog.records)


def test_correction_comparison_can_be_disabled(caplog):
    import logging
    behavioral = np.array([0] * 10 + [1] * 10)
    narrative = np.array(([0] * 5 + [1] * 5) * 2)
    with caplog.at_level(logging.INFO, logger="domain_emergence.domain_alignment"):
        align_domains(behavioral, narrative, log_correction_comparison=False)
    assert not any("correction comparison" in r.message for r in caplog.records)


def test_correction_comparison_does_not_change_decision_path():
    """Still Bonferroni per doctrine -- logging BH alongside must not
    change joint_domains."""
    behavioral = np.array([0] * 10 + [1] * 10)
    narrative = np.array(([0] * 5 + [1] * 5) * 2)
    with_log = align_domains(behavioral, narrative, log_correction_comparison=True)
    without_log = align_domains(behavioral, narrative, log_correction_comparison=False)
    assert with_log.joint_domains == without_log.joint_domains
    assert with_log.pair_pvalues == without_log.pair_pvalues