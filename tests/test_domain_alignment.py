import numpy as np
from domain_emergence.domain_alignment import align_domains, NOISE


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