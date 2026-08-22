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


# --- S56.7: immutable episode_id join key ---

def test_episode_ids_well_formed_passes_through():
    behavioral = np.array([0] * 20 + [1] * 20)
    narrative = np.array([0] * 20 + [1] * 20)
    episode_ids = np.arange(40)
    result = align_domains(behavioral, narrative, episode_ids=episode_ids)
    assert result.episode_ids is not None
    assert np.array_equal(result.episode_ids, episode_ids)


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


# --- S56.5: subject-level cluster-bootstrap now drives the decision
# path when subject_ids is supplied (senior sign-off on method choice
# NOT yet obtained -- see _subject_level_pvalue docstring) ---

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
    # only subject 0 ever co-occurs (b=1,n=1), but does so on ALL of
    # their windows -- looks like a strong per-episode signal but is
    # really just n=1 independent subject repeated.
    subj0_mask = subject_ids == 0
    behavioral[subj0_mask] = 1
    narrative[subj0_mask] = 1
    behavioral[~subj0_mask] = rng.integers(0, 2, size=(~subj0_mask).sum())
    narrative[~subj0_mask] = rng.integers(0, 2, size=(~subj0_mask).sum())

    result = subject_level_dependence_diagnostic(
        behavioral, narrative, subject_ids, b_id=1, n_id=1,
        n_bootstrap=500, seed=0,
    )
    assert "naive_fisher_p" in result
    assert "subject_level_bootstrap_p" in result
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
        behavioral, narrative, subject_ids, b_id=0, n_id=0, n_bootstrap=50, seed=0,
    )
    after = align_domains(behavioral, narrative).joint_domains
    assert before == after


def test_subject_ids_supplied_uses_subject_level_pvalue_method():
    behavioral = np.array([0] * 20 + [1] * 20)
    narrative = np.array([0] * 20 + [1] * 20)
    subject_ids = np.repeat(np.arange(8), 5)
    result = align_domains(
        behavioral, narrative, subject_ids=subject_ids,
        subject_level_n_bootstrap=200, subject_level_seed=0,
    )
    assert result.pvalue_method == "subject_level_cluster_bootstrap"
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
        subject_level_n_bootstrap=500, subject_level_seed=0,
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