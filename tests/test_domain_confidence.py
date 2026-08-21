import pytest
from domain_emergence.domain_confidence import (
    compute_domain_confidence, DomainConfidence, MIN_CONFIDENCE_THRESHOLD,
)


def test_high_everything_gives_active_status():
    result = compute_domain_confidence(
        observation_count=500, persistence_duration=200,
        n_phase_transitions_survived=3, fisher_p_value=0.001,
    )
    assert result.status == "active"
    assert result.confidence >= MIN_CONFIDENCE_THRESHOLD


def test_low_everything_gives_candidate_status():
    result = compute_domain_confidence(
        observation_count=1, persistence_duration=1,
        n_phase_transitions_survived=0, fisher_p_value=0.99,
    )
    assert result.status == "candidate"
    assert result.confidence < MIN_CONFIDENCE_THRESHOLD


def test_survival_weighted_highest():
    """Doctrine: cross-phase survival is the strongest stability signal --
    a domain with only survival high (everything else low) should score
    meaningfully higher than a domain with only obs count high (everything
    else low), since survival's weight (0.4) exceeds observation's (0.2)."""
    only_survival = compute_domain_confidence(
        observation_count=0, persistence_duration=0,
        n_phase_transitions_survived=5, fisher_p_value=1.0,
    )
    only_observation = compute_domain_confidence(
        observation_count=500, persistence_duration=0,
        n_phase_transitions_survived=0, fisher_p_value=1.0,
    )
    # the raw saturating scores aren't directly comparable (different scales),
    # but survival's higher WEIGHT (0.4 vs 0.2) should win out in final confidence
    # even though observation_score here is numerically higher than survival_score
    assert only_observation.observation_score > only_survival.survival_score
    assert only_survival.confidence > only_observation.confidence


def test_coherence_score_inverts_pvalue():
    result_sig = compute_domain_confidence(
        observation_count=10, persistence_duration=10,
        n_phase_transitions_survived=0, fisher_p_value=0.0,
    )
    result_nonsig = compute_domain_confidence(
        observation_count=10, persistence_duration=10,
        n_phase_transitions_survived=0, fisher_p_value=1.0,
    )
    assert result_sig.coherence_score == 1.0
    assert result_nonsig.coherence_score == 0.0
    assert result_sig.confidence > result_nonsig.confidence


def test_zero_observation_count_gives_zero_obs_score():
    result = compute_domain_confidence(
        observation_count=0, persistence_duration=0,
        n_phase_transitions_survived=0, fisher_p_value=1.0,
    )
    assert result.observation_score == 0.0
    assert result.persistence_score == 0.0
    assert result.confidence == 0.0


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        compute_domain_confidence(
            observation_count=10, persistence_duration=10,
            n_phase_transitions_survived=1, fisher_p_value=0.1,
            weights={"observation": 0.5, "persistence": 0.5, "survival": 0.5, "coherence": 0.5},
        )


def test_confidence_always_in_zero_one_range():
    for obs in [0, 1, 1000]:
        for dur in [0, 1, 1000]:
            for surv in [0, 1, 100]:
                for p in [0.0, 0.5, 1.0]:
                    result = compute_domain_confidence(obs, dur, surv, p)
                    assert 0.0 <= result.confidence <= 1.0