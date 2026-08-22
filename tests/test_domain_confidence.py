import numpy as np
import pytest
import warnings
from domain_emergence.domain_confidence import (
    compute_domain_confidence, DomainConfidence, MIN_CONFIDENCE_THRESHOLD,
    bootstrap_domain_stability, compare_confidence_formulations,
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


def test_bootstrap_domain_stability_all_present_is_stable():
    indicator = np.ones(30)
    result = bootstrap_domain_stability(indicator, n_bootstrap=200, seed=0)
    assert result == 1.0


def test_bootstrap_domain_stability_all_absent_is_zero():
    indicator = np.zeros(30)
    result = bootstrap_domain_stability(indicator, n_bootstrap=200, seed=0)
    assert result == 0.0


def test_compare_confidence_formulations_reports_divergence():
    """S56.4 T1: diagnostic report, no pass/fail gate -- just quantifies
    where 1-p and the bootstrap-stability estimate diverge."""
    indicator = np.array([1] * 3 + [0] * 27)  # rare co-occurrence
    result = compare_confidence_formulations(
        fisher_p_value=0.001, co_occurrence_indicator=indicator,
        n_bootstrap=200, seed=0,
    )
    assert "naive_one_minus_p" in result
    assert "bootstrap_stability" in result
    assert "bootstrap_ci_lower" in result
    assert "divergence" in result
    assert result["naive_one_minus_p"] == pytest.approx(0.999)


def test_confidence_always_in_zero_one_range():
    for obs in [0, 1, 1000]:
        for dur in [0, 1, 1000]:
            for surv in [0, 1, 100]:
                for p in [0.0, 0.5, 1.0]:
                    result = compute_domain_confidence(obs, dur, surv, p)
                    assert 0.0 <= result.confidence <= 1.0


def test_legacy_path_warns_and_uses_one_minus_p():
    """S56.4 fix: omitting co_occurrence_indicator still works (backward
    compatible) but now explicitly warns that it's the flagged legacy
    formula, not silently treated as equivalent to the real fix."""
    with pytest.warns(RuntimeWarning, match="legacy"):
        result = compute_domain_confidence(
            observation_count=10, persistence_duration=10,
            n_phase_transitions_survived=0, fisher_p_value=0.0,
        )
    assert result.coherence_method == "legacy_one_minus_p"
    assert result.coherence_score == 1.0


def test_bootstrap_path_no_warning_and_uses_ci_lower():
    """S56.4 fix: supplying co_occurrence_indicator takes the real
    effect-size path -- no warning, coherence_method reports it, and
    coherence_score is a genuine CI-lower-bound rate, not 1-p."""
    strong_indicator = np.ones(50)  # co-occurs every single episode
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # fail the test if any warning fires
        result = compute_domain_confidence(
            observation_count=10, persistence_duration=10,
            n_phase_transitions_survived=0, fisher_p_value=0.5,
            co_occurrence_indicator=strong_indicator,
            n_bootstrap=200, bootstrap_seed=0,
        )
    assert result.coherence_method == "bootstrap_ci_lower"
    # co-occurs every episode -> bootstrap CI lower bound should be 1.0
    assert result.coherence_score == pytest.approx(1.0)


def test_bootstrap_path_weak_signal_gives_low_coherence():
    """Rare co-occurrence (3/30 episodes) should give a LOW CI-lower-
    bound coherence score, even though the naive 1-p formula could
    still report a small (seemingly 'significant') p-value for the
    same weak underlying signal -- this is exactly the divergence
    S56.4 exists to surface."""
    weak_indicator = np.array([1] * 3 + [0] * 27)
    result = compute_domain_confidence(
        observation_count=10, persistence_duration=10,
        n_phase_transitions_survived=0, fisher_p_value=0.001,
        co_occurrence_indicator=weak_indicator,
        n_bootstrap=500, bootstrap_seed=0,
    )
    assert result.coherence_method == "bootstrap_ci_lower"
    assert result.coherence_score < 0.3, \
        f"weak/rare co-occurrence should yield low coherence, got {result.coherence_score}"


def test_bootstrap_path_empty_indicator_gives_zero_coherence():
    result = compute_domain_confidence(
        observation_count=10, persistence_duration=10,
        n_phase_transitions_survived=0, fisher_p_value=0.5,
        co_occurrence_indicator=np.array([]),
    )
    assert result.coherence_score == 0.0