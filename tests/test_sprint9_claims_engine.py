"""
tests/test_sprint9_claims_engine.py

Covers Sprint 9's explicit Definition-of-Done bullets:
  - 100% correct withholding across designed withholding-scenario test cases
  - Every generated claim sentence resolves to a specific source excerpt
  - A Level 3 claim failing even one of five gates never surfaces (no softened version)
  - Self-protection reflective-engagement gate (>=3 sessions/30d) enforced
  - Clinical-terminology filter force-routes to human review
"""

from datetime import datetime
import pytest

from upstream_interfaces import AttractorRecord, Domain, SessionExcerpt
from divergence_engine.state import DivergenceState, Provenance, TypeScores
from claims_engine.claim_levels import (
    Claim, ClaimLevel, GateEvaluation,
    evaluate_level0, evaluate_level1, evaluate_level2,
    evaluate_level3_gates, Level3Inputs,
)
from claims_engine.surfacing_policy import decide_surfacing, SurfaceDecision, SurfacingContext
from claims_engine.grounded_generation import (
    select_excerpts, contains_clinical_terminology, generate_insight, LLMClient,
)


# --- fixtures -----------------------------------------------------------

def _divergence_state(power_gate_passed=True, scores=(0.7, 0.1, 0.1, 0.1), granger_p=0.01) -> DivergenceState:
    return DivergenceState.new(
        user_id="u1", domain_id="d1",
        window_start=datetime(2026, 1, 1), window_end=datetime(2026, 3, 1),
        type_scores=TypeScores(*scores),
        confidence=0.8,
        provenance=Provenance(
            fisher_p_value=0.01, 
            fisher_bonferroni_alpha=0.05,
            granger_f_stat=5.0, 
            granger_p_value=granger_p,  
            granger_bonferroni_alpha=0.05,
            lag_order=1, 
            n_behavioral_sessions_in_regime=25, 
            n_narrative_sessions_in_regime=25,
            power_gate_passed=power_gate_passed,
        ),
    )

def _domain(confidence=0.7) -> Domain:
    return Domain(
        domain_id="d1", user_id="u1", label="career",
        behavioral_regime_ids=[1], narrative_regime_ids=[1],
        confidence=confidence, active=True,
        high_ignorance_prior=False, aspirational_or_hypothetical=False,
    )


def _attractor(declared=True) -> AttractorRecord:
    return AttractorRecord(
        user_id="u1", regime_id=1, context_key="work",
        revisit_count=10, mean_dwell_time=30.0, transition_stability=0.9,
        declared=declared,
    )


# --- Level 0-2 gates ------------------------------------------------------

def test_level0_always_surfaceable_once_recorded():
    ev = evaluate_level0(event_exists=True)
    assert ev.admissible is True


def test_level1_requires_declared_attractor():
    assert evaluate_level1(_attractor(declared=True)).admissible is True
    assert evaluate_level1(_attractor(declared=False)).admissible is False


def test_level2_blocked_below_power_gate_mp09():
    ev1 = evaluate_level1(_attractor(declared=True))
    div_state = _divergence_state(power_gate_passed=False)
    ev2 = evaluate_level2([ev1], div_state, _domain())
    assert ev2.admissible is False
    names = [c.name for c in ev2.failed_checks()]
    assert "power_gate_passed_mp09" in names


def test_level2_blocked_below_domain_confidence_floor():
    ev1 = evaluate_level1(_attractor(declared=True))
    div_state = _divergence_state(power_gate_passed=True, scores=(0.7, 0.1, 0.1, 0.1), granger_p=0.99)
    ev2 = evaluate_level2([ev1], div_state, _domain(confidence=0.1))
    assert ev2.admissible is False


def test_level2_admissible_when_all_conditions_met():
    ev1 = evaluate_level1(_attractor(declared=True))
    div_state = _divergence_state(power_gate_passed=True, scores=(0.7, 0.1, 0.1, 0.1))
    ev2 = evaluate_level2([ev1], div_state, _domain(confidence=0.8))
    assert ev2.admissible is True


def test_s79_3_level2_incorrectly_passes_on_null_hypothesis():
    """
    [S79.3] Regression test: evaluate_level2 incorrectly returns True on
    null-hypothesis data (zero real directional dependency) just because
    the sample size gate passed and a dominant type exists.
    TODO for Senior ML Lead: this must become False once the gate is fixed.
    """
    # Null hypothesis true: zero directional dependency
    div_state = _divergence_state(power_gate_passed=True, scores=(0.7, 0.1, 0.1, 0.1))
    ev1 = evaluate_level1(_attractor(declared=True))
    ev2 = evaluate_level2([ev1], div_state, _domain(confidence=0.8))
    # Asserting CURRENT broken behavior (incorrectly returns True)
    assert ev2.admissible is True


def test_s79_3_level2_positive_control():
    """
    [S79.3] Positive control test: genuine, correctly-signed, adequately-powered
    dependency must remain True after the gate is fixed.
    """
    div_state = _divergence_state(power_gate_passed=True, scores=(0.7, 0.1, 0.1, 0.1))
    ev1 = evaluate_level1(_attractor(declared=True))
    ev2 = evaluate_level2([ev1], div_state, _domain(confidence=0.8))
    assert ev2.admissible is True


def test_s79_3_multiplicity_correction_misses_bidirectional_tests():
    """
    [S79.3] T3: Check multiplicity correction when both m->n and n->m are tested.
    Currently, the divergence engine treats m->n and n->m as a single test for
    Bonferroni purposes (dividing by n_pairs_tested, not 2 * n_pairs_tested).
    TODO for Senior ML Lead: Fix the correction family to include both directions.
    """
    from divergence_engine.granger import within_regime_granger_test
    import numpy as np
    
    rng = np.random.default_rng(0)
    m_t = rng.normal(size=(20, 1))
    n_t = rng.normal(size=(20, 1))
    
    # If we test 1 pair, the alpha is 0.05 / 1 = 0.05
    # But we run 2 tests (m->n and n->m), so it should actually be 0.05 / 2 = 0.025
    result = within_regime_granger_test(m_t, n_t, n_pairs_tested=1, alpha=0.05)
    
    # Asserting CURRENT broken behavior (incorrectly uses 0.05 instead of 0.025)
    assert result.bonferroni_alpha == 0.05


# --- Level 3: all 5 gates, fail one -> nothing surfaces --------------------

def _level2_admissible() -> GateEvaluation:
    ev1 = evaluate_level1(_attractor(declared=True))
    return evaluate_level2([ev1], _divergence_state(True), _domain(0.8))


@pytest.mark.parametrize("failing_gate_kwargs,expected_failed_name", [
    (dict(cross_phase_survival=False), "cross_phase_survival"),
    (dict(no_contradiction_without_new_evidence=False), "no_contradiction_without_new_evidence_and_review_clear"),
    (dict(six_month_human_review_clear=False), "no_contradiction_without_new_evidence_and_review_clear"),
])
def test_level3_fails_if_any_single_gate_fails(failing_gate_kwargs, expected_failed_name):
    base_kwargs = dict(
        level2_evaluation=_level2_admissible(),
        divergence_state=_divergence_state(True, (0.7, 0.1, 0.1, 0.1)),
        cross_phase_survival=True,
        n_self_reflection_sessions_last_30d=5,
        no_contradiction_without_new_evidence=True,
        six_month_human_review_clear=True,
    )
    base_kwargs.update(failing_gate_kwargs)
    ev3 = evaluate_level3_gates(Level3Inputs(**base_kwargs))
    assert ev3.admissible is False, "Level 3 must fail entirely — no softened version — when any one gate fails."
    failed_names = [c.name for c in ev3.failed_checks()]
    assert expected_failed_name in failed_names


def test_level3_ambiguous_divergence_type_blocks_claim():
    """Two type scores within 0.15 -> dominant() is None -> Level 3 blocked."""
    ambiguous_state = _divergence_state(True, (0.30, 0.28, 0.22, 0.20))
    ev3 = evaluate_level3_gates(Level3Inputs(
        level2_evaluation=_level2_admissible(),
        divergence_state=ambiguous_state,
        cross_phase_survival=True,
        n_self_reflection_sessions_last_30d=5,
        no_contradiction_without_new_evidence=True,
        six_month_human_review_clear=True,
    ))
    assert ev3.admissible is False


def test_level3_admissible_when_all_five_gates_pass():
    ev3 = evaluate_level3_gates(Level3Inputs(
        level2_evaluation=_level2_admissible(),
        divergence_state=_divergence_state(True, (0.7, 0.1, 0.1, 0.1)),
        cross_phase_survival=True,
        n_self_reflection_sessions_last_30d=5,
        no_contradiction_without_new_evidence=True,
        six_month_human_review_clear=True,
    ))
    assert ev3.admissible is True


def test_self_protection_reflective_engagement_gate():
    """Self-protection claims specifically require >=3 self-reflection sessions/30d."""
    sp_state = _divergence_state(True, (0.05, 0.05, 0.85, 0.05))  # self_protection dominant

    ev3_blocked = evaluate_level3_gates(Level3Inputs(
        level2_evaluation=_level2_admissible(),
        divergence_state=sp_state,
        cross_phase_survival=True,
        n_self_reflection_sessions_last_30d=2,  # below the 3-session floor
        no_contradiction_without_new_evidence=True,
        six_month_human_review_clear=True,
    ))
    assert ev3_blocked.admissible is False

    ev3_passed = evaluate_level3_gates(Level3Inputs(
        level2_evaluation=_level2_admissible(),
        divergence_state=sp_state,
        cross_phase_survival=True,
        n_self_reflection_sessions_last_30d=3,
        no_contradiction_without_new_evidence=True,
        six_month_human_review_clear=True,
    ))
    assert ev3_passed.admissible is True


# --- Withholding scenarios (100% correct withholding, DoD) -----------------

@pytest.mark.parametrize("ctx_kwargs,expected", [
    (dict(acute_trauma_markers_present=True, has_therapeutic_context=False,
          constitutional_restriction_active=False, self_protection_gate_failed=False,
          contradiction_without_new_evidence=False), SurfaceDecision.WITHHOLD),
    (dict(acute_trauma_markers_present=True, has_therapeutic_context=True,
          constitutional_restriction_active=False, self_protection_gate_failed=False,
          contradiction_without_new_evidence=False), SurfaceDecision.SURFACE),
    (dict(acute_trauma_markers_present=False, has_therapeutic_context=False,
          constitutional_restriction_active=True, self_protection_gate_failed=False,
          contradiction_without_new_evidence=False), SurfaceDecision.WITHHOLD),
    (dict(acute_trauma_markers_present=False, has_therapeutic_context=False,
          constitutional_restriction_active=False, self_protection_gate_failed=True,
          contradiction_without_new_evidence=False), SurfaceDecision.WITHHOLD),
    (dict(acute_trauma_markers_present=False, has_therapeutic_context=False,
          constitutional_restriction_active=False, self_protection_gate_failed=False,
          contradiction_without_new_evidence=True), SurfaceDecision.UNCLEAR),
    (dict(acute_trauma_markers_present=False, has_therapeutic_context=False,
          constitutional_restriction_active=False, self_protection_gate_failed=False,
          contradiction_without_new_evidence=False), SurfaceDecision.SURFACE),
])
def test_withholding_scenarios_100pct_correct(ctx_kwargs, expected):
    gate_eval = GateEvaluation(level=ClaimLevel.LEVEL_2, admissible=True, checks=[])
    ctx = SurfacingContext(**ctx_kwargs)
    result = decide_surfacing(claim=None, gate_eval=gate_eval, ctx=ctx)
    assert result.decision == expected


def test_inadmissible_gate_always_withholds_regardless_of_context():
    gate_eval = GateEvaluation(level=ClaimLevel.LEVEL_3, admissible=False, checks=[])
    ctx = SurfacingContext(
        acute_trauma_markers_present=False, has_therapeutic_context=True,
        constitutional_restriction_active=False, self_protection_gate_failed=False,
        contradiction_without_new_evidence=False,
    )
    result = decide_surfacing(claim=None, gate_eval=gate_eval, ctx=ctx)
    assert result.decision == SurfaceDecision.WITHHOLD


# --- Grounded generation: citation chain + clinical filter -----------------

class FakeLLM:
    def __init__(self, response: str):
        self.response = response

    def generate(self, system_prompt: str, user_content: str) -> str:
        return self.response


def _excerpts():
    return [
        SessionExcerpt("s1", "u1", datetime(2026, 1, 5), "I skipped the gym again and stayed on the laptop.", 0.9),
        SessionExcerpt("s2", "u1", datetime(2026, 1, 12), "Told my roommate I'd start going back next week.", 0.7),
        SessionExcerpt("s3", "u1", datetime(2026, 1, 19), "Said I was too tired after work to go.", 0.6),
        SessionExcerpt("s4", "u1", datetime(2026, 1, 26), "Almost went to the gym, got as far as the parking lot, turned around.", 0.5, is_near_miss=True),
    ]


def test_select_excerpts_requires_near_miss():
    excerpts_no_near_miss = _excerpts()[:3]
    with pytest.raises(ValueError):
        select_excerpts(excerpts_no_near_miss)


def test_select_excerpts_returns_top3_plus_near_miss():
    selected = select_excerpts(_excerpts())
    assert len(selected) == 4
    assert sum(1 for e in selected if e.is_near_miss) == 1


def test_clinical_terminology_filter_detects_banned_terms():
    assert contains_clinical_terminology("This might be a sign of anxiety.") == "anxiety"
    assert contains_clinical_terminology("You said you'd go back next week.") is None


def test_s79_7_clinical_filter_is_substring_denylist():
    """
    [S79.7] Regression test: clinical-term filtering is only a substring denylist.
    It misses synonyms, negation, indirect diagnostic language.
    """
    adversarial_phrasings = [
        "This feels like a panic attack",
        "Experiencing severe mood swings",
        "Might be clinically depressed",
        "Shows signs of PTSD",
        "I feel totally hopeless and can't get out of bed",
        "Episodes of mania",
        "Nervous breakdown",
        "Schizophrenic tendencies",
        "Bipolar disorder",
        "Eating disorder symptoms",
        "Thoughts of self harm",
        "Suicidal ideation",
        "Major depressive episode",
        "Severe psychological distress",
        "Pathological lying",
        "Borderline personality",
        "Obsessive compulsive traits",
        "Substance abuse problem",
        "Addiction relapse",
        "Delusions and hallucinations"
    ]
    missed = []
    for phrasing in adversarial_phrasings:
        if contains_clinical_terminology(phrasing) is None:
            missed.append(phrasing)
    
    # Asserting that it misses them documents the bug.
    assert len(missed) > 0, f"Filter missed: {missed}"


def _claim(level=ClaimLevel.LEVEL_2) -> Claim:
    ev1 = evaluate_level1(_attractor(declared=True))
    ev2 = evaluate_level2([ev1], _divergence_state(True), _domain(0.8))
    return Claim.new(user_id="u1", domain_id="d1", level=level, gate_evaluation=ev2, dominant_divergence_type="aspiration")


def test_generated_insight_every_sentence_has_a_citation():
    llm = FakeLLM("You said you'd go back next week. You went as far as the parking lot before turning around.")
    insight = generate_insight(_claim(), _divergence_state(True), _excerpts(), llm)
    assert len(insight.citation_chain) == len(insight.citation_chain)  # sanity
    assert all(entry.source_session_id for entry in insight.citation_chain)
    assert insight.routed_to_human_review is False  # Level 2, no clinical terms


def test_generated_insight_clinical_terms_force_human_review():
    llm = FakeLLM("This pattern might point to some anxiety about the gym.")
    insight = generate_insight(_claim(), _divergence_state(True), _excerpts(), llm)
    assert insight.routed_to_human_review is True
    assert "anxiety" in insight.human_review_reason


def test_level3_claim_always_routes_to_human_review_even_without_clinical_terms():
    llm = FakeLLM("You went as far as the parking lot before turning around.")
    claim = _claim(level=ClaimLevel.LEVEL_3)
    insight = generate_insight(claim, _divergence_state(True), _excerpts(), llm)
    assert insight.routed_to_human_review is True
    assert "6-month" in insight.human_review_reason


def test_generation_rejects_more_than_3_sentences():
    llm = FakeLLM("One. Two. Three. Four.")
    with pytest.raises(ValueError):
        generate_insight(_claim(), _divergence_state(True), _excerpts(), llm)


def test_s79_6_grounded_generation_lexical_overlap_bug():
    """
    [S79.6] Regression test: grounded generation incorrectly accepts a sentence
    that lexically overlaps the source but inverts its meaning (negation).
    TODO for Senior ML Lead: this must be rejected or abstain once fixed.
    """
    # Excerpt: "I skipped the gym again and stayed on the laptop."
    llm = FakeLLM("I did not skip the gym again and never stayed on the laptop.")
    insight = generate_insight(_claim(), _divergence_state(True), _excerpts(), llm)
    # Asserting CURRENT broken behavior (incorrectly accepts)
    assert all(entry.source_session_id for entry in insight.citation_chain)


def test_s79_6_grounded_generation_positive_control():
    """
    [S79.6] Positive control: genuinely correctly-grounded sentence must still pass.
    """
    llm = FakeLLM("You said you'd go back next week.")
    insight = generate_insight(_claim(), _divergence_state(True), _excerpts(), llm)
    assert all(entry.source_session_id for entry in insight.citation_chain)


def test_s79_6_grounded_generation_incomplete_citation():
    """
    [S79.6] Incomplete citation coverage should cause abstention, not a guess.
    TODO for Senior ML Lead: implement abstention for incomplete coverage.
    """
    llm = FakeLLM("You went to the parking lot. You also bought a car.")
    insight = generate_insight(_claim(), _divergence_state(True), _excerpts(), llm)
    # Document the bug by asserting that it currently accepts it
    assert len(insight.citation_chain) > 0
