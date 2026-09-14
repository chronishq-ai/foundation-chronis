import numpy as np
import pytest
from domain_emergence.domain_confidence import (
    compute_domain_confidence, DomainConfidence, DomainSignificance,
    DomainMagnitude, MIN_CONFIDENCE_THRESHOLD,
    MissingCorrectedEvidenceError, UnrecognizedSignificanceMethodError,
)

SIG_METHOD = "subject_level_within_subject_permutation"


def _indicators(n, joint_rate, seed=0):
    """n episodes, behavioral==narrative==1 for a joint_rate fraction of
    them (perfectly dependent core) -- a simple strong-signal fixture."""
    rng = np.random.default_rng(seed)
    joint = (rng.random(n) < joint_rate).astype(int)
    return joint.copy(), joint.copy()


def test_high_everything_gives_active_status():
    behavioral, narrative = _indicators(500, 0.9, seed=1)
    result = compute_domain_confidence(
        observation_count=500, persistence_duration=200,
        n_phase_transitions_survived=3,
        significance_pvalue=0.001, significance_method=SIG_METHOD,
        behavioral_indicator=behavioral, narrative_indicator=narrative,
        bootstrap_seed=0,
    )
    assert result.status == "active"
    assert result.confidence >= MIN_CONFIDENCE_THRESHOLD


def test_low_everything_gives_candidate_status():
    behavioral = np.zeros(30, dtype=int)
    narrative = np.zeros(30, dtype=int)
    result = compute_domain_confidence(
        observation_count=1, persistence_duration=1,
        n_phase_transitions_survived=0,
        significance_pvalue=0.99, significance_method=SIG_METHOD,
        behavioral_indicator=behavioral, narrative_indicator=narrative,
        bootstrap_seed=0,
    )
    assert result.status == "candidate"
    assert result.confidence < MIN_CONFIDENCE_THRESHOLD


def test_survival_weighted_highest():
    """Doctrine: cross-phase survival is the strongest stability signal --
    a domain with only survival high (everything else low) should score
    meaningfully higher than a domain with only obs count high (everything
    else low), since survival's weight (0.4) exceeds observation's (0.2)."""
    flat = np.zeros(10, dtype=int)
    only_survival = compute_domain_confidence(
        observation_count=0, persistence_duration=0,
        n_phase_transitions_survived=5,
        significance_pvalue=1.0, significance_method=SIG_METHOD,
        behavioral_indicator=flat, narrative_indicator=flat,
    )
    only_observation = compute_domain_confidence(
        observation_count=500, persistence_duration=0,
        n_phase_transitions_survived=0,
        significance_pvalue=1.0, significance_method=SIG_METHOD,
        behavioral_indicator=flat, narrative_indicator=flat,
    )
    assert only_observation.observation_score > only_survival.survival_score
    assert only_survival.confidence > only_observation.confidence


def test_missing_indicators_raises_typed_error_no_fallback():
    """R2-S56.5: calling the production API without the corrected
    per-episode evidence must fail loudly, not silently fall back to
    any p-value-inversion formula."""
    with pytest.raises(MissingCorrectedEvidenceError):
        compute_domain_confidence(
            observation_count=10, persistence_duration=10,
            n_phase_transitions_survived=0,
            significance_pvalue=0.001, significance_method=SIG_METHOD,
            behavioral_indicator=None, narrative_indicator=None,
        )


def test_unrecognized_significance_method_rejected():
    """An uncorrected/arbitrary p-value label must not be accepted as if
    it were the corrected pipeline's output."""
    flat = np.ones(10, dtype=int)
    with pytest.raises(UnrecognizedSignificanceMethodError):
        compute_domain_confidence(
            observation_count=10, persistence_duration=10,
            n_phase_transitions_survived=0,
            significance_pvalue=0.001, significance_method="raw_uncorrected_p",
            behavioral_indicator=flat, narrative_indicator=flat,
        )


def test_mismatched_indicator_lengths_rejected():
    with pytest.raises(ValueError):
        compute_domain_confidence(
            observation_count=10, persistence_duration=10,
            n_phase_transitions_survived=0,
            significance_pvalue=0.001, significance_method=SIG_METHOD,
            behavioral_indicator=np.ones(10), narrative_indicator=np.ones(9),
        )


def test_independent_high_prevalence_signals_give_low_coherence():
    """Two streams that are each individually common (~80% prevalence)
    but statistically INDEPENDENT of each other will co-occur often by
    chance alone (~64% joint rate) -- a raw-co-occurrence-rate measure
    would misread that as strong 'coherence'. The phi-based effect size
    must not be fooled by shared high base rates: independent streams
    should give a coherence score near zero, not near their joint rate."""
    rng = np.random.default_rng(42)
    n = 2000
    behavioral = (rng.random(n) < 0.8).astype(int)
    narrative = (rng.random(n) < 0.8).astype(int)  # independent draw
    result = compute_domain_confidence(
        observation_count=n, persistence_duration=50,
        n_phase_transitions_survived=0,
        significance_pvalue=0.5, significance_method=SIG_METHOD,
        behavioral_indicator=behavioral, narrative_indicator=narrative,
        bootstrap_seed=0,
    )
    joint_rate = np.mean((behavioral == 1) & (narrative == 1))
    assert joint_rate > 0.5, "sanity check: chance co-occurrence is high here"
    assert abs(result.magnitude.phi) < 0.1
    assert result.coherence_score < 0.1, (
        f"independent high-base-rate streams should not read as coherent, "
        f"got coherence_score={result.coherence_score}"
    )


def test_known_dependent_control_moves_both_metrics():
    """A genuinely dependent pairing should show both HIGH significance
    (very small corrected p) and HIGH magnitude (phi near 1, CI-lower
    comfortably above 0) -- the two objects should agree here, even
    though they answer different questions."""
    joint, _ = _indicators(300, 0.7, seed=3)
    result = compute_domain_confidence(
        observation_count=300, persistence_duration=100,
        n_phase_transitions_survived=1,
        significance_pvalue=0.0001, significance_method=SIG_METHOD,
        behavioral_indicator=joint, narrative_indicator=joint,
        bootstrap_seed=0,
    )
    assert result.significance.is_significant
    assert result.magnitude.phi == pytest.approx(1.0)
    assert result.magnitude.ci_lower > 0.9
    assert result.coherence_score > 0.9


def test_significance_object_never_used_as_magnitude():
    """A non-significant p-value must not, by itself, drag coherence to
    zero -- significance and magnitude are independent objects; a small
    dataset can be non-significant yet still show a strong phi if it
    happens to be perfectly aligned (the CI will be wide, reflecting low
    n, but the point estimate is still a real dependence measure, not a
    p-value)."""
    joint, _ = _indicators(20, 0.5, seed=5)
    result = compute_domain_confidence(
        observation_count=20, persistence_duration=5,
        n_phase_transitions_survived=0,
        significance_pvalue=0.6, significance_method=SIG_METHOD,
        behavioral_indicator=joint, narrative_indicator=joint,
        bootstrap_seed=0,
    )
    assert result.significance.is_significant is False
    assert result.magnitude.phi == pytest.approx(1.0)


def test_zero_observation_count_gives_zero_obs_score():
    flat = np.zeros(5, dtype=int)
    result = compute_domain_confidence(
        observation_count=0, persistence_duration=0,
        n_phase_transitions_survived=0,
        significance_pvalue=1.0, significance_method=SIG_METHOD,
        behavioral_indicator=flat, narrative_indicator=flat,
    )
    assert result.observation_score == 0.0
    assert result.persistence_score == 0.0
    assert result.confidence == 0.0


def test_weights_must_sum_to_one():
    flat = np.ones(10, dtype=int)
    with pytest.raises(ValueError):
        compute_domain_confidence(
            observation_count=10, persistence_duration=10,
            n_phase_transitions_survived=1,
            significance_pvalue=0.1, significance_method=SIG_METHOD,
            behavioral_indicator=flat, narrative_indicator=flat,
            weights={"observation": 0.5, "persistence": 0.5, "survival": 0.5, "coherence": 0.5},
        )


def test_confidence_always_in_zero_one_range():
    rng = np.random.default_rng(7)
    for obs in [0, 1, 1000]:
        for dur in [0, 1, 1000]:
            for surv in [0, 1, 100]:
                for p in [0.0, 0.5, 1.0]:
                    ind = (rng.random(20) < 0.5).astype(int)
                    result = compute_domain_confidence(
                        obs, dur, surv, p, SIG_METHOD, ind, ind, bootstrap_seed=0,
                    )
                    assert 0.0 <= result.confidence <= 1.0


def test_empty_indicator_gives_zero_coherence():
    empty = np.array([], dtype=int)
    result = compute_domain_confidence(
        observation_count=10, persistence_duration=10,
        n_phase_transitions_survived=0,
        significance_pvalue=0.5, significance_method=SIG_METHOD,
        behavioral_indicator=empty, narrative_indicator=empty,
    )
    assert result.coherence_score == 0.0


def _code_only(path):
    """Strip STRING and COMMENT tokens (docstrings/comments) so a grep
    check looks only at actual executable code, not documentation that
    legitimately discusses the pattern it forbids."""
    import io
    import tokenize
    with open(path, "rb") as f:
        tokens = tokenize.tokenize(f.readline)
        pieces = [
            tok.string for tok in tokens
            if tok.type not in (tokenize.STRING, tokenize.COMMENT, tokenize.ENCODING)
        ]
    return " ".join(pieces)


def test_no_one_minus_p_pattern_in_production_module():
    """R2-S56.5 / XCUT-2 CI check: production domain_confidence.py must
    contain zero occurrences of the legacy p-value-inversion pattern in
    actual code (docstrings are allowed to discuss it by name)."""
    code = _code_only("domain_emergence/domain_confidence.py")
    assert "1 - fisher_p_value" not in code
    assert "1.0 - fisher_p_value" not in code
    assert "one_minus_p" not in code