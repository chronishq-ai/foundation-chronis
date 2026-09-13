"""
tests/test_sprint12_mirror.py
Sprint 12 — The Mirror: Insight Generation Engine

Test coverage:
  Day 34: MirrorInsightGenerator, citation chain, vocabulary extraction
  Day 35: SpecificityLinter (banned patterns, word count, anchors), ToneMode, TTS stub
  Day 36: AdaptiveFeedbackLoop (threshold ratchet, domain suppression), InsightArchive
  Pipeline: Stage 0/1 silence regression, Stage 4 generation, domain suppression gate

DoD assertions:
  - 20 synthetic insight variants pass specificity linter regression
    (test_linter_20_sample_pass) — linter gate only
  - 20 actual end-to-end pipeline runs (generator + citation + linter + archive) all pass
    (TestMirror20RunEndToEnd) — full pipeline integration
  - Mirror produces zero output for Stage 0/1 users (test_stage_0_silence,
    test_stage_1_silence)
  - Banned pattern class is hard-rejected (test_linter_rejects_*)
  - Every sentence resolves to a citation (test_citation_chain_full_coverage)
  - Clinical terminology = hard stop: NOT archived, NOT surfaced (test_clinical_hard_stop)
  - Repeated NOT_YET raises per-user threshold (test_adaptive_threshold_rises)
  - Archive search returns correct results (test_archive_full_text_search,
    test_archive_tag_search)
"""

from __future__ import annotations

import math
import re
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Sequence
from unittest.mock import MagicMock, patch

import pytest

# ── Sprint 12 imports ──────────────────────────────────────────────────────
from mirror.specificity_linter import (
    lint_insight, LintStatus, LintViolation,
    WORD_COUNT_MIN, WORD_COUNT_MAX,
)
from mirror.tone_calibration import (
    ToneMode, tone_system_prompt, synthesize_voice_stub,
)
from mirror.insight_generator import MIRROR_SYSTEM_PROMPT as _BASE
from mirror.feedback_loop import (
    AdaptiveFeedbackStore, FeedbackRating,
    DEFAULT_ADMISSIBILITY_THRESHOLD,
    MAX_ADMISSIBILITY_THRESHOLD,
    _NOT_YET_DELTA, _TOO_SOON_DELTA, _HELPFUL_DELTA,
)
from mirror.archive import InsightRecord, InsightArchive
from mirror.insight_generator import (
    MirrorInsightGenerator, extract_user_vocabulary, ClinicalTerminologyError,
)
from mirror.mirror_pipeline import run_mirror_pipeline, _SILENT_STAGES

# ── Sprint 10 imports ──────────────────────────────────────────────────────
from cold_start.cold_start import ColdStartStage, ColdStartState

# ── Sprint 9 imports ──────────────────────────────────────────────────────
from claims_engine.claim_levels import (
    Claim, ClaimLevel, GateEvaluation, GateCheck,
)
from claims_engine.grounded_generation import CitationChainEntry

# ── Shared upstream types ──────────────────────────────────────────────────
from upstream_interfaces import SessionExcerpt
from divergence_engine.state import DivergenceState, TypeScores, Provenance


# ===========================================================================
# Shared helpers & fixtures
# ===========================================================================

def _make_excerpt(
    session_id: str,
    text: str,
    contribution_score: float = 0.5,
    is_near_miss: bool = False,
) -> SessionExcerpt:
    return SessionExcerpt(
        session_id=session_id,
        user_id="u_001",
        timestamp=datetime(2026, 1, 1),
        text=text,
        contribution_score=contribution_score,
        is_near_miss=is_near_miss,
    )


def _make_excerpts_pool(n_supporting: int = 4, include_near_miss: bool = True) -> List[SessionExcerpt]:
    """Realistic excerpt pool for a synthetic user."""
    pool = [
        _make_excerpt(
            f"s{i}",
            f"On session {i} I noticed I kept circling back to the same project even when "
            f"I had planned to switch tasks. The pull felt almost automatic, like momentum "
            f"rather than choice. I kept working through the context switches instead of "
            f"stepping back. My focus was unusually sustained compared to earlier weeks.",
            contribution_score=1.0 - i * 0.1,
        )
        for i in range(n_supporting)
    ]
    if include_near_miss:
        pool.append(_make_excerpt(
            "s_nm",
            "On this session I started to move toward deep work but got pulled away by "
            "interruptions before reaching the same sustained state as in other sessions.",
            contribution_score=0.3,
            is_near_miss=True,
        ))
    return pool


def _make_claim(level: ClaimLevel = ClaimLevel.LEVEL_2, domain_id: str = "work") -> Claim:
    gate = GateEvaluation(
        level=level,
        admissible=True,
        checks=[GateCheck("test_gate", True, "synthetic")],
    )
    return Claim.new(
        user_id="u_001",
        domain_id=domain_id,
        level=level,
        gate_evaluation=gate,
        dominant_divergence_type="aspiration",
    )


def _make_divergence_state(confidence: float = 0.75) -> DivergenceState:
    scores = TypeScores(
        ignorance=0.1,
        aspiration=confidence,
        self_protection=0.05,
        active_transition=0.1,
    )
    prov = Provenance(
        fisher_p_value=0.01,
        fisher_bonferroni_alpha=0.05,
        granger_f_stat=5.0,
        granger_p_value=0.01,
        granger_bonferroni_alpha=0.05,
        lag_order=1,
        n_behavioral_sessions_in_regime=25,
        n_narrative_sessions_in_regime=25,
        power_gate_passed=True,
    )
    return DivergenceState.new(
        user_id="u_001",
        domain_id="work",
        window_start=datetime(2026, 1, 1),
        window_end=datetime(2026, 3, 1),
        type_scores=scores,
        confidence=confidence,
        provenance=prov,
    )


def _make_cold_start_state(
    stage: ColdStartStage,
    day: int = 95,
    can_surface: bool = True,
) -> ColdStartState:
    return ColdStartState(
        user_id="u_001",
        day=day,
        stage=stage,
        can_surface_claims=can_surface if stage == ColdStartStage.STAGE_4 else False,
        hssm_fitted=(stage.value >= 2),
        user_facing_message="test",
        internal_estimates_only=(stage == ColdStartStage.STAGE_2),
    )


def _make_llm_client(response: Optional[str] = None) -> MagicMock:
    """Stub LLMClient that returns a 100–200 word, evidence-grounded insight."""
    if response is None:
        # Build a response that:
        #  - references "session" (ties to excerpt text)
        #  - is second person
        #  - is 110–150 words
        #  - contains no banned patterns
        response = (
            "Over the past several sessions you have shown a consistent pattern of returning "
            "to focused work even when context switches were planned. "
            "In session s0, the pull toward the project felt automatic rather than deliberate, "
            "suggesting the attractor basin is becoming stronger than your scheduled transitions. "
            "In session s1, the same momentum appears, this time persisting through external "
            "interruptions that slowed but did not break your concentration. "
            "What stands out is how the near-miss session s_nm differs: the conditions for deep "
            "work began but the sustained state did not fully materialise, which makes the "
            "contrast with your stronger sessions more legible. "
            "This pattern in the data points to a deepening pull toward sustained engagement "
            "that your current schedule may not fully account for."
        )
    client = MagicMock()
    client.generate.return_value = response
    return client


def _make_insight_record(
    user_id: str = "u_001",
    text: Optional[str] = None,
    tone: ToneMode = ToneMode.REFLECTIVE,
    domain_id: str = "work",
    divergence_type: str = "aspiration",
    tags: Optional[List[str]] = None,
) -> InsightRecord:
    if text is None:
        text = (
            "Over the past several sessions you returned consistently to focused "
            "work even when interruptions arose. "
            "Session s0 showed the strongest pull toward the project, with momentum "
            "persisting beyond planned context switches. "
            "The near-miss session s_nm offers a legible contrast: conditions "
            "were present but the sustained state did not fully materialise, "
            "pointing to a pattern that is still consolidating rather than fixed."
        )
    chain = [
        CitationChainEntry(0, "Over the past several sessions...", "s0"),
        CitationChainEntry(1, "Session s0 showed...", "s1"),
        CitationChainEntry(2, "The near-miss session...", "s_nm"),
    ]
    extra_tags = tags or []
    return InsightRecord.new(
        user_id=user_id,
        text=text,
        tone=tone,
        citation_chain=chain,
        claim_ids=[str(uuid.uuid4())],
        dominant_divergence_type=divergence_type,
        domain_id=domain_id,
        extra_tags=extra_tags,
    )


# ===========================================================================
# Day 35 — Specificity Linter
# ===========================================================================

class TestSpecificityLinterBannedPatterns:
    """The linter hard-rejects the explicitly banned pattern class."""

    @pytest.mark.parametrize("banned_text", [
        # Exact spec example
        "You were stressed today! Try meditation.",
        # Variants of the same pattern
        "You are feeling anxious lately. Consider mindfulness.",
        "You seem tired recently. Maybe try a walk.",
        "You felt overwhelmed today. You should try journaling.",
        # Filler affirmation
        "Be kind to yourself. You've got this.",
        "It's okay to feel this way. Everyone goes through it.",
        # Unanchored evaluation
        "Your mood is low this week.",
        "Your stress level has been high recently.",
    ])
    def test_linter_rejects_generic_coaching(self, banned_text: str):
        # Pad to 100+ words so only banned-pattern violation fires, not word count
        padding = (
            " In session s0 you showed sustained focus through multiple context "
            "switches, returning to the same project even when interruptions arose. "
            "Session s1 showed momentum persisting beyond planned transitions. "
            "The pattern across sessions s0 and s1 suggests a deepening attractor "
            "basin that your current schedule may not fully account for. "
            "Session s2 reinforced this pattern with similar persistence."
        )
        full_text = banned_text + padding
        # Build a full citation chain so only BANNED_PATTERN violation fires
        n_sents = len(re.split(r"(?<=[.!?])\s+", full_text.strip()))
        full_chain = [CitationChainEntry(i, f"s{i}", f"s{i}") for i in range(n_sents)]
        result = lint_insight(full_text, citation_chain=full_chain)
        assert result.status == LintStatus.FAIL
        ban_violations = [v for v in result.violations if v.violation_type == "BANNED_PATTERN"]
        assert len(ban_violations) >= 1, (
            f"Expected BANNED_PATTERN violation for: {banned_text!r}\n"
            f"Got violations: {result.violations}"
        )

    def test_linter_passes_specific_grounded_insight(self):
        """A well-written, grounded, evidence-anchored insight passes."""
        good_text = (
            "Over the past several sessions you have shown a consistent pattern of returning "
            "to focused work even when context switches were planned. "
            "In session s0, the pull toward the project felt automatic rather than deliberate, "
            "suggesting the attractor basin is becoming stronger than your scheduled transitions. "
            "In session s1, the same momentum appears, this time persisting through external "
            "interruptions that slowed but did not break your concentration. "
            "What stands out is how the near-miss session s_nm differs: the conditions for deep "
            "work began but the sustained state did not fully materialise, which makes the "
            "contrast with your stronger sessions more legible. "
            "This pattern in the data points to a deepening pull toward sustained engagement "
            "that your current schedule may not fully account for."
        )
        # Build a full citation chain so only pattern/structural checks apply
        n_sents = len(re.split(r"(?<=[.!?])\s+", good_text.strip()))
        full_chain = [CitationChainEntry(i, f"s{i}", f"s{i}") for i in range(n_sents)]
        result = lint_insight(good_text, citation_chain=full_chain)
        assert result.status == LintStatus.PASS, (
            f"Expected PASS but got: {result.violations}"
        )


class TestSpecificityLinterBypassCoverage:
    """
    S12.1: Verify that previously bypassable phrasings are now caught.

    Each parametrized phrase represents a concrete bypass path that evaded
    the original pattern set. The extended catalogue must catch all of them.
    """

    # Padding: grounded, data-specific filler that on its own would PASS
    _PADDING = (
        " In session s0 you showed sustained focus through multiple context "
        "switches, returning to the same project even when interruptions arose. "
        "Session s1 showed momentum persisting beyond planned transitions. "
        "The pattern across sessions s0 and s1 suggests a deepening attractor "
        "basin that your current schedule may not fully account for. "
        "Session s2 reinforced this pattern with similar persistence across data."
    )

    def _run(self, bypass_text: str) -> "LintResult":
        full = bypass_text + self._PADDING
        n_sents = len(re.split(r"(?<=[.!?])\s+", full.strip()))
        chain = [CitationChainEntry(i, f"s{i}", f"s{i}") for i in range(n_sents)]
        return lint_insight(full, citation_chain=chain)

    @pytest.mark.parametrize("bypass_text", [
        # Extended verb list: appear/look not in original catalogue
        "You appear overwhelmed by the workload.",
        "You look drained after this past stretch.",
        "You appeared exhausted throughout the session.",
        # Extended emotion words not in original list
        "You feel drained from all the transitions.",
        "You seem depleted lately.",
        "You sound exhausted today.",
    ])
    def test_extended_emotion_verbs_and_words_caught(self, bypass_text: str):
        """Extended verb/emotion list closes the appear/look/drained bypass."""
        result = self._run(bypass_text)
        assert result.status == LintStatus.FAIL, (
            f"Expected FAIL for bypass phrase: {bypass_text!r}"
        )
        ban = [v for v in result.violations if v.violation_type == "BANNED_PATTERN"]
        assert len(ban) >= 1, f"No BANNED_PATTERN for: {bypass_text!r}"

    @pytest.mark.parametrize("bypass_text", [
        # Indirect attribution — "it sounds/seems/looks/appears like you're ..."
        "It sounds like you're going through something difficult.",
        "Seems like you might be struggling lately.",
        "It appears like you could be overwhelmed.",
        "Looks like things are getting difficult right now.",
    ])
    def test_indirect_attribution_caught(self, bypass_text: str):
        """Indirect 'it sounds/seems/looks/appears like you' is now caught."""
        result = self._run(bypass_text)
        assert result.status == LintStatus.FAIL, (
            f"Expected FAIL for indirect attribution: {bypass_text!r}"
        )
        ban = [v for v in result.violations if v.violation_type == "BANNED_PATTERN"]
        assert len(ban) >= 1, f"No BANNED_PATTERN for: {bypass_text!r}"

    @pytest.mark.parametrize("bypass_text", [
        # Implicit wellness — "might/could/would benefit from rest"
        "You might benefit from some rest.",
        "Some downtime would help you recharge.",
        "You could use a break from this intensity.",
        "A bit of space might help.",
    ])
    def test_implicit_wellness_caught(self, bypass_text: str):
        """Implicit wellness recommendations (benefit from/would help) are now caught."""
        result = self._run(bypass_text)
        assert result.status == LintStatus.FAIL, (
            f"Expected FAIL for implicit wellness: {bypass_text!r}"
        )
        ban = [v for v in result.violations if v.violation_type == "BANNED_PATTERN"]
        assert len(ban) >= 1, f"No BANNED_PATTERN for: {bypass_text!r}"

    @pytest.mark.parametrize("bypass_text", [
        # Vague difficulty reference — "a tough time", "a hard week"
        "You're navigating a tough time.",
        "It's been a hard week for you.",
        "This has been a difficult period.",
        "You seem to be in a rough patch.",
    ])
    def test_vague_difficulty_reference_caught(self, bypass_text: str):
        """Vague difficulty references without data anchors are now caught."""
        result = self._run(bypass_text)
        assert result.status == LintStatus.FAIL, (
            f"Expected FAIL for vague difficulty: {bypass_text!r}"
        )
        ban = [v for v in result.violations if v.violation_type == "BANNED_PATTERN"]
        assert len(ban) >= 1, f"No BANNED_PATTERN for: {bypass_text!r}"

    @pytest.mark.parametrize("bypass_text", [
        # Unsolicited care directives
        "Taking care of yourself is important right now.",
        "Don't forget to rest when you can.",
        "Make sure you get some sleep tonight.",
    ])
    def test_unsolicited_care_directive_caught(self, bypass_text: str):
        """Unsolicited self-care directives are now caught."""
        result = self._run(bypass_text)
        assert result.status == LintStatus.FAIL, (
            f"Expected FAIL for care directive: {bypass_text!r}"
        )
        ban = [v for v in result.violations if v.violation_type == "BANNED_PATTERN"]
        assert len(ban) >= 1, f"No BANNED_PATTERN for: {bypass_text!r}"


class TestSpecificityLinterWordCount:
    """Word count bounds: 100–200 words."""

    def test_linter_rejects_below_100_words(self):
        short = "In session s0 you showed focus. The pattern is notable and worth observing."
        # Empty chain — MISSING_ANCHOR violations fire but that doesn't affect WORD_COUNT test
        n_sents = len(re.split(r"(?<=[.!?])\s+", short.strip()))
        full_chain = [CitationChainEntry(i, f"s{i}", f"s{i}") for i in range(n_sents)]
        result = lint_insight(short, citation_chain=full_chain)
        assert result.status == LintStatus.FAIL
        wc_violations = [v for v in result.violations if v.violation_type == "WORD_COUNT"]
        assert len(wc_violations) == 1

    def test_linter_rejects_above_200_words(self):
        # Build exactly 201 words of clean, grounded content (no banned patterns)
        base_sentence = (
            "In session s0 you showed sustained focus through context switches and returned "
            "to the same project despite planned interruptions consistently over time. "
        )
        # Each base_sentence is ~25 words; repeat 12× to exceed 200 reliably
        long_text = (base_sentence * 12).strip()  # ~300 words
        wc = len(re.findall(r"\w+", long_text))
        assert wc > 200, f"Precondition failed: expected >200 words, got {wc}"
        n_sents = len(re.split(r"(?<=[.!?])\s+", long_text.strip()))
        full_chain = [CitationChainEntry(i, f"s{i}", f"s{i}") for i in range(n_sents)]
        result = lint_insight(long_text, citation_chain=full_chain)
        assert result.status == LintStatus.FAIL
        wc_violations = [v for v in result.violations if v.violation_type == "WORD_COUNT"]
        assert len(wc_violations) == 1

    def test_linter_accepts_exactly_100_words(self):
        # Build exactly 100 words from grounded, non-banned content
        word_unit = "session"
        filler = (
            "In {s} s0 you returned to the same project despite planned interruptions and context switches. "
            "{S} s1 showed a similar pull with momentum persisting beyond scheduled transitions clearly. "
            "The near-miss {s} s_nm shows conditions were present but the sustained state did not fully "
            "materialise, making the contrast with s0 and s1 more legible across the data available."
        ).format(s="session", S="Session")
        words = filler.split()
        # Pad or trim to exactly 100 words
        extra = "sustained focused engagement persisting through transitions and context switches again".split()
        combined = words + extra
        combined = combined[:100]
        if len(combined) < 100:
            combined += ["data"] * (100 - len(combined))
        text_100 = " ".join(combined)
        actual_count = len(re.findall(r"\w+", text_100))
        n_sents = len(re.split(r"(?<=[.!?])\s+", text_100.strip()))
        full_chain = [CitationChainEntry(i, f"s{i}", f"s{i}") for i in range(n_sents)]
        result = lint_insight(text_100, citation_chain=full_chain)
        wc_violations = [v for v in result.violations if v.violation_type == "WORD_COUNT"]
        assert len(wc_violations) == 0, (
            f"Word count={actual_count}, expected 100–200. violations={wc_violations}"
        )


class TestSpecificityLinterCitationAnchors:
    """Every sentence must have a citation anchor when citation_chain is provided."""

    def test_missing_anchor_fails(self):
        text = (
            "In session s0 you showed sustained focus. "
            "Session s1 had similar momentum across context switches. "
            "The near-miss session offers a legible contrast to the pattern."
            " Session s2 showed the same pull again persisting through external interruptions. "
            "Your schedule and session patterns point toward a deepening engagement basin."
        )
        # Only provide citations for sentences 0 and 1 (sentence 2 has no anchor)
        partial_chain = [
            CitationChainEntry(0, "sentence0", "s0"),
            CitationChainEntry(1, "sentence1", "s1"),
            # sentence 2 is missing
        ]
        result = lint_insight(text, citation_chain=partial_chain)
        assert result.status == LintStatus.FAIL
        anchor_violations = [v for v in result.violations if v.violation_type == "MISSING_ANCHOR"]
        assert len(anchor_violations) >= 1

    def test_full_citation_chain_passes_anchor_check(self):
        text = (
            "In session s0 you returned consistently to the same project. "
            "Session s1 showed the same momentum persisting beyond planned transitions. "
            "The near-miss session s_nm highlights the contrast with your stronger sessions clearly."
        )
        # Pad to 100 words
        padding = (
            " This pattern across sessions s0 and s1 suggests a deepening attractor basin "
            "that your schedule may not yet fully account for in its current form. "
            "Session data reinforces this observation across multiple recent engagement periods."
        )
        full_text = text + padding
        n_sentences = len(re.split(r"(?<=[.!?])\s+", full_text.strip()))
        full_chain = [
            CitationChainEntry(i, f"sentence_{i}", f"s{i}")
            for i in range(n_sentences)
        ]
        result = lint_insight(full_text, citation_chain=full_chain)
        anchor_violations = [v for v in result.violations if v.violation_type == "MISSING_ANCHOR"]
        assert len(anchor_violations) == 0, f"Unexpected anchor violations: {anchor_violations}"


class TestSpecificityLinter20Sample:
    """
    DoD: A random sample of 20 generated insights passes the specificity linter.

    We generate 20 variants of the canonical good insight with slight perturbations
    to verify the linter is robust across minor phrasing changes.
    """

    def _make_good_variant(self, seed: int) -> str:
        variants = [
            "returned to focused work",
            "circled back to the same project",
            "sustained engagement through context switches",
            "showed persistent momentum",
            "maintained focus across interruptions",
        ]
        verb = variants[seed % len(variants)]
        return (
            f"Over the past several sessions you have {verb} even when transitions were planned. "
            f"In session s{seed}, the pull toward sustained work felt structural rather than chosen, "
            f"suggesting the attractor basin is growing stronger than your scheduled breaks. "
            f"Session s{seed+1} reinforced this with similar momentum persisting through "
            f"external interruptions that slowed but did not break your concentration entirely. "
            f"What stands out is how session s_nm differs: conditions for deep work began "
            f"but the sustained state did not fully materialise, making the contrast with "
            f"s{seed} more legible in context. This pattern in the data points to an engagement "
            f"pull that your current schedule may not yet fully account for or accommodate."
        )

    def test_linter_20_sample_pass(self):
        """All 20 generated insight variants must pass the specificity linter."""
        failures = []
        for i in range(20):
            text = self._make_good_variant(seed=i)
            # S12.1: citation_chain is required — build a full chain covering every sentence
            n_sents = len(re.split(r"(?<=[.!?])\s+", text.strip()))
            full_chain = [CitationChainEntry(j, f"s{j}", f"s{j}") for j in range(n_sents)]
            result = lint_insight(text, citation_chain=full_chain)
            if result.status != LintStatus.PASS:
                failures.append((i, result.violations))
        assert failures == [], (
            f"{len(failures)}/20 insights failed the linter:\n"
            + "\n".join(f"  variant {i}: {vs}" for i, vs in failures)
        )


# ===========================================================================
# Day 35 — Tone calibration + TTS stub
# ===========================================================================

class TestToneCalibration:
    """Tone modifiers are prepended to the base prompt at generation time."""

    @pytest.mark.parametrize("tone", list(ToneMode))
    def test_tone_system_prompt_contains_base(self, tone: ToneMode):
        """All tone variants must include the base constrained prompt rules."""
        from mirror.insight_generator import MIRROR_SYSTEM_PROMPT
        result = tone_system_prompt(MIRROR_SYSTEM_PROMPT, tone)
        assert "Do not diagnose" in result
        assert "Every sentence must be traceable" in result

    @pytest.mark.parametrize("tone", list(ToneMode))
    def test_tone_modifier_prepended_not_appended(self, tone: ToneMode):
        """Tone modifier appears BEFORE the base prompt."""
        from mirror.insight_generator import MIRROR_SYSTEM_PROMPT
        result = tone_system_prompt(MIRROR_SYSTEM_PROMPT, tone)
        tone_idx = result.find("Tone:")
        base_idx = result.find("Do not diagnose")
        assert tone_idx < base_idx, (
            f"Tone modifier must come BEFORE base prompt for tone={tone.name}"
        )

    def test_direct_tone_uses_direct_keyword(self):
        from mirror.insight_generator import MIRROR_SYSTEM_PROMPT
        result = tone_system_prompt(MIRROR_SYSTEM_PROMPT, ToneMode.DIRECT)
        assert "direct" in result.lower() or "Direct" in result

    def test_warm_tone_uses_warm_keyword(self):
        from mirror.insight_generator import MIRROR_SYSTEM_PROMPT
        result = tone_system_prompt(MIRROR_SYSTEM_PROMPT, ToneMode.WARM)
        assert "warm" in result.lower() or "Warm" in result

    def test_reflective_tone_uses_reflective_keyword(self):
        from mirror.insight_generator import MIRROR_SYSTEM_PROMPT
        result = tone_system_prompt(MIRROR_SYSTEM_PROMPT, ToneMode.REFLECTIVE)
        assert "reflective" in result.lower() or "Reflective" in result


class TestTTSStub:
    """TTS stub: synthesised, never a recording of the user."""

    @pytest.mark.parametrize("tone", list(ToneMode))
    def test_tts_stub_is_stub(self, tone: ToneMode):
        result = synthesize_voice_stub("some insight text", tone)
        assert result.is_stub is True

    def test_tts_stub_audio_bytes_are_bytes(self):
        result = synthesize_voice_stub("test", ToneMode.REFLECTIVE)
        assert isinstance(result.audio_bytes, bytes)

    def test_tts_stub_notice_says_synthesised(self):
        result = synthesize_voice_stub("test", ToneMode.WARM)
        assert "synthesised" in result.notice.lower() or "SYNTHESISED" in result.notice

    def test_tts_stub_does_not_claim_to_be_user_recording(self):
        result = synthesize_voice_stub("test", ToneMode.DIRECT)
        notice_lower = result.notice.lower()
        # Must NOT claim to be a user recording
        assert "recording of the user" not in notice_lower or "not a recording" in notice_lower

    @pytest.mark.parametrize("tone", list(ToneMode))
    def test_tts_stub_tone_preserved_in_result(self, tone: ToneMode):
        result = synthesize_voice_stub("test insight", tone)
        assert result.tone == tone


# ===========================================================================
# Day 34 — User vocabulary extractor
# ===========================================================================

class TestUserVocabularyExtractor:
    """Recurring vocabulary extracted from session excerpts."""

    def test_extracts_common_words(self):
        excerpts = [
            _make_excerpt("s1", "momentum momentum momentum focus sustained"),
            _make_excerpt("s2", "momentum focus sustained context"),
            _make_excerpt("s3", "momentum sustained context transitions"),
        ]
        vocab = extract_user_vocabulary(excerpts)
        assert "momentum" in vocab
        assert "sustained" in vocab

    def test_excludes_stopwords(self):
        excerpts = [
            _make_excerpt("s1", "the and or but in on at to for of with is was"),
        ]
        vocab = extract_user_vocabulary(excerpts)
        assert all(w not in ("the", "and", "or", "but", "in", "on") for w in vocab)

    def test_returns_list(self):
        excerpts = _make_excerpts_pool(n_supporting=3)
        vocab = extract_user_vocabulary(excerpts)
        assert isinstance(vocab, list)

    def test_handles_empty_excerpts(self):
        vocab = extract_user_vocabulary([])
        assert vocab == []


# ===========================================================================
# Day 34 — Citation chain full coverage
# ===========================================================================

class TestCitationChainFullCoverage:
    """Every sentence in the generated insight must have a citation anchor."""

    def test_citation_chain_full_coverage(self):
        """
        End-to-end: MirrorInsightGenerator returns a draft whose citation_chain
        covers every sentence in the output text.
        """
        excerpts = _make_excerpts_pool(n_supporting=4)
        claim = _make_claim(ClaimLevel.LEVEL_2)
        ds = _make_divergence_state()
        client = _make_llm_client()

        gen = MirrorInsightGenerator(llm_client=client)
        draft = gen.generate(
            user_id="u_001",
            claims=[claim],
            candidate_excerpts=excerpts,
            divergence_state=ds,
            tone=ToneMode.REFLECTIVE,
        )

        # Every sentence index must appear in the citation chain
        sentences = re.split(r"(?<=[.!?])\s+", draft.text.strip())
        cited_indices = {entry.sentence_index for entry in draft.citation_chain}
        for i in range(len(sentences)):
            assert i in cited_indices, (
                f"Sentence {i} has no citation anchor.\n"
                f"Sentence: {sentences[i]!r}\n"
                f"Cited indices: {cited_indices}"
            )

    def test_generator_raises_on_level0_only_claims(self):
        excerpts = _make_excerpts_pool()
        level0_claim = _make_claim(ClaimLevel.LEVEL_0)
        ds = _make_divergence_state()
        client = _make_llm_client()
        gen = MirrorInsightGenerator(llm_client=client)
        with pytest.raises(ValueError, match="Level 0"):
            gen.generate(
                user_id="u_001",
                claims=[level0_claim],
                candidate_excerpts=excerpts,
                divergence_state=ds,
            )


# ===========================================================================
# Day 36 — Adaptive feedback loop
# ===========================================================================

class TestAdaptiveFeedbackThreshold:
    """Repeated NOT_YET / TOO_SOON ratings raise per-user threshold."""

    def test_initial_threshold_is_default(self):
        store = AdaptiveFeedbackStore()
        assert store.get_admissibility_threshold("u_new") == DEFAULT_ADMISSIBILITY_THRESHOLD

    def test_not_yet_raises_threshold(self):
        store = AdaptiveFeedbackStore()
        store.record_feedback("u_001", "ins_1", FeedbackRating.NOT_YET)
        t = store.get_admissibility_threshold("u_001")
        assert t > DEFAULT_ADMISSIBILITY_THRESHOLD
        assert math.isclose(t, DEFAULT_ADMISSIBILITY_THRESHOLD + _NOT_YET_DELTA, abs_tol=1e-9)

    def test_too_soon_raises_threshold_more_than_not_yet(self):
        store_a = AdaptiveFeedbackStore()
        store_b = AdaptiveFeedbackStore()
        store_a.record_feedback("u_a", "ins_1", FeedbackRating.NOT_YET)
        store_b.record_feedback("u_b", "ins_1", FeedbackRating.TOO_SOON)
        assert store_b.get_admissibility_threshold("u_b") > store_a.get_admissibility_threshold("u_a")

    def test_helpful_lowers_threshold(self):
        store = AdaptiveFeedbackStore()
        # First raise it
        store.record_feedback("u_001", "ins_1", FeedbackRating.NOT_YET)
        raised = store.get_admissibility_threshold("u_001")
        # Then lower it
        store.record_feedback("u_001", "ins_2", FeedbackRating.HELPFUL)
        lowered = store.get_admissibility_threshold("u_001")
        assert lowered < raised

    def test_threshold_capped_at_max(self):
        store = AdaptiveFeedbackStore()
        # Apply many NOT_YET ratings
        for i in range(100):
            store.record_feedback("u_001", f"ins_{i}", FeedbackRating.NOT_YET)
        assert store.get_admissibility_threshold("u_001") <= MAX_ADMISSIBILITY_THRESHOLD

    def test_repeated_not_yet_ratchets_threshold(self):
        """5 consecutive NOT_YET ratings must raise the threshold at least 4× _NOT_YET_DELTA."""
        store = AdaptiveFeedbackStore()
        for i in range(5):
            store.record_feedback("u_001", f"ins_{i}", FeedbackRating.NOT_YET)
        expected_min = DEFAULT_ADMISSIBILITY_THRESHOLD + 4 * _NOT_YET_DELTA
        assert store.get_admissibility_threshold("u_001") >= expected_min

    def test_thresholds_are_per_user(self):
        """Feedback for one user must not affect another user's threshold."""
        store = AdaptiveFeedbackStore()
        for i in range(5):
            store.record_feedback("u_001", f"ins_{i}", FeedbackRating.NOT_YET)
        # u_002 is unaffected
        assert store.get_admissibility_threshold("u_002") == DEFAULT_ADMISSIBILITY_THRESHOLD


class TestDomainSuppression:
    """TOO_SOON suppresses a domain for 30 days."""

    def test_too_soon_suppresses_domain(self):
        store = AdaptiveFeedbackStore()
        store.record_feedback("u_001", "ins_1", FeedbackRating.TOO_SOON, domain_id="work")
        assert store.is_domain_suppressed("u_001", "work") is True

    def test_not_yet_does_not_suppress_domain(self):
        store = AdaptiveFeedbackStore()
        store.record_feedback("u_001", "ins_1", FeedbackRating.NOT_YET, domain_id="work")
        assert store.is_domain_suppressed("u_001", "work") is False

    def test_suppression_is_per_domain(self):
        store = AdaptiveFeedbackStore()
        store.record_feedback("u_001", "ins_1", FeedbackRating.TOO_SOON, domain_id="work")
        assert store.is_domain_suppressed("u_001", "relationships") is False

    def test_suppression_is_per_user(self):
        store = AdaptiveFeedbackStore()
        store.record_feedback("u_001", "ins_1", FeedbackRating.TOO_SOON, domain_id="work")
        assert store.is_domain_suppressed("u_002", "work") is False

    def test_unsuppressed_domain_returns_false(self):
        store = AdaptiveFeedbackStore()
        assert store.is_domain_suppressed("u_001", "health") is False


# ===========================================================================
# Day 36 — Insight Archive
# ===========================================================================

class TestInsightArchive:
    """Append-only archive with full-text + tag search."""

    def test_append_and_retrieve(self):
        archive = InsightArchive()
        record = _make_insight_record()
        archive.append(record)
        retrieved = archive.get(record.user_id, record.insight_id)
        assert retrieved.insight_id == record.insight_id

    def test_append_only_no_overwrite(self):
        archive = InsightArchive()
        record = _make_insight_record()
        archive.append(record)
        with pytest.raises(ValueError, match="append-only"):
            archive.append(record)

    def test_add_feedback_annotation(self):
        archive = InsightArchive()
        record = _make_insight_record()
        archive.append(record)
        archive.add_feedback(record.user_id, record.insight_id, FeedbackRating.HELPFUL)
        fb = archive.get_feedback(record.user_id, record.insight_id)
        assert fb == FeedbackRating.HELPFUL

    def test_feedback_on_nonexistent_raises(self):
        archive = InsightArchive()
        with pytest.raises(KeyError):
            archive.add_feedback("u_001", "nonexistent_id", FeedbackRating.HELPFUL)

    def test_archive_full_text_search(self):
        archive = InsightArchive()
        record_a = _make_insight_record(user_id="u_001", text=(
            "In session s0 you showed sustained focus through context switches and interruptions. "
            "Session s1 reinforced the same pull toward the project despite planned transitions away. "
            "The near-miss session s_nm highlights the contrast with your stronger sessions clearly overall."
        ))
        record_b = _make_insight_record(user_id="u_001", text=(
            "In session s0 you spent less time on the project and more time on communications. "
            "Session s1 showed similar distribution with no dominant attractor visible in data. "
            "The near-miss session s_nm appears structurally different from your s0 engagement patterns."
        ))
        archive.append(record_a)
        archive.append(record_b)

        results = archive.search("u_001", query="sustained focus")
        assert len(results) == 1
        assert results[0].insight_id == record_a.insight_id

    def test_archive_tag_search(self):
        archive = InsightArchive()
        warm_record = _make_insight_record(tone=ToneMode.WARM)
        reflective_record = _make_insight_record(tone=ToneMode.REFLECTIVE)
        archive.append(warm_record)
        archive.append(reflective_record)

        results = archive.search("u_001", tags=["tone:warm"])
        assert len(results) == 1
        assert results[0].tone == ToneMode.WARM

    def test_archive_combined_search(self):
        archive = InsightArchive()
        r1 = _make_insight_record(
            divergence_type="aspiration",
            text=(
                "In session s0 you showed persistent momentum toward the project. "
                "Session s1 reinforced this attractor pull through context switches. "
                "The near-miss session s_nm highlights the contrast with your stronger sessions."
            ),
        )
        r2 = _make_insight_record(
            divergence_type="ignorance",
            text=(
                "In session s0 your engagement with the project was lower than in other sessions. "
                "Session s1 showed similar distribution with no strong attractor visible in data. "
                "The near-miss session s_nm appears different from your s0 engagement patterns."
            ),
        )
        archive.append(r1)
        archive.append(r2)
        results = archive.search(
            "u_001",
            query="momentum",
            tags=["divergence_type:aspiration"],
        )
        assert len(results) == 1
        assert results[0].insight_id == r1.insight_id

    def test_list_all_ordered_by_time(self):
        archive = InsightArchive()
        r1 = _make_insight_record()
        r2 = _make_insight_record()
        archive.append(r1)
        archive.append(r2)
        all_records = archive.list_all("u_001")
        assert all_records[0].generated_at <= all_records[1].generated_at

    def test_count(self):
        archive = InsightArchive()
        for _ in range(5):
            archive.append(_make_insight_record())
        assert archive.count("u_001") == 5
        assert archive.count("u_999") == 0


# ===========================================================================
# Pipeline — Stage 0 / Stage 1 silence (DoD regression)
# ===========================================================================

class TestMirrorStage01Silence:
    """
    DoD: The Mirror produces zero output for any Stage 0/1 cold-start user.
    Code-enforced (not caller-dependent).

    Reuses Sprint 10 fixtures (ColdStartStage enum + ColdStartState).
    """

    @pytest.mark.parametrize("stage,day", [
        (ColdStartStage.STAGE_0, 1),
        (ColdStartStage.STAGE_0, 5),
        (ColdStartStage.STAGE_0, 7),
        (ColdStartStage.STAGE_1, 8),
        (ColdStartStage.STAGE_1, 15),
        (ColdStartStage.STAGE_1, 29),
    ])
    def test_mirror_silent_stage_0_and_1(self, stage: ColdStartStage, day: int):
        """Mirror pipeline must return None for all Stage 0 and Stage 1 days."""
        cold_state = _make_cold_start_state(stage=stage, day=day, can_surface=False)
        archive = InsightArchive()
        client = _make_llm_client()

        result = run_mirror_pipeline(
            user_id="u_001",
            cold_start_state=cold_state,
            claims=[_make_claim()],
            candidate_excerpts=_make_excerpts_pool(),
            divergence_state=_make_divergence_state(),
            llm_client=client,
            archive=archive,
        )
        assert result is None, (
            f"Mirror returned non-None output for Stage {stage.name} Day {day}. "
            f"This violates the Sprint 10/12 zero-inference requirement."
        )
        # LLM must NOT have been called
        client.generate.assert_not_called()
        # Archive must be empty
        assert archive.count("u_001") == 0

    def test_stage_0_is_in_silent_stages(self):
        assert ColdStartStage.STAGE_0 in _SILENT_STAGES

    def test_stage_1_is_in_silent_stages(self):
        assert ColdStartStage.STAGE_1 in _SILENT_STAGES

    def test_stage_2_is_not_in_silent_stages(self):
        assert ColdStartStage.STAGE_2 not in _SILENT_STAGES

    def test_stage_4_is_not_in_silent_stages(self):
        assert ColdStartStage.STAGE_4 not in _SILENT_STAGES


class TestMirrorEvidenceGate:
    """Stage 4 without can_surface_claims=True must also be silent."""

    def test_stage_4_no_evidence_gate_is_silent(self):
        cold_state = _make_cold_start_state(
            stage=ColdStartStage.STAGE_4, day=95, can_surface=False
        )
        archive = InsightArchive()
        client = _make_llm_client()
        result = run_mirror_pipeline(
            user_id="u_001",
            cold_start_state=cold_state,
            claims=[_make_claim()],
            candidate_excerpts=_make_excerpts_pool(),
            divergence_state=_make_divergence_state(),
            llm_client=client,
            archive=archive,
        )
        assert result is None
        client.generate.assert_not_called()


class TestMirrorDomainSuppression:
    """TOO_SOON-suppressed domains produce no Mirror output."""

    def test_suppressed_domain_produces_no_output(self):
        cold_state = _make_cold_start_state(
            stage=ColdStartStage.STAGE_4, day=95, can_surface=True
        )
        feedback_store = AdaptiveFeedbackStore()
        # Suppress the 'work' domain
        feedback_store.record_feedback("u_001", "ins_0", FeedbackRating.TOO_SOON, domain_id="work")

        archive = InsightArchive()
        client = _make_llm_client()

        result = run_mirror_pipeline(
            user_id="u_001",
            cold_start_state=cold_state,
            claims=[_make_claim(domain_id="work")],
            candidate_excerpts=_make_excerpts_pool(),
            divergence_state=_make_divergence_state(),
            llm_client=client,
            archive=archive,
            feedback_store=feedback_store,
            domain_id="work",
        )
        assert result is None, "Suppressed domain must produce no Mirror output."
        client.generate.assert_not_called()

    def test_unsuppressed_domain_proceeds_normally(self):
        cold_state = _make_cold_start_state(
            stage=ColdStartStage.STAGE_4, day=95, can_surface=True
        )
        feedback_store = AdaptiveFeedbackStore()
        # Suppress 'health', not 'work'
        feedback_store.record_feedback("u_001", "ins_0", FeedbackRating.TOO_SOON, domain_id="health")

        archive = InsightArchive()
        client = _make_llm_client()

        result = run_mirror_pipeline(
            user_id="u_001",
            cold_start_state=cold_state,
            claims=[_make_claim(domain_id="work")],
            candidate_excerpts=_make_excerpts_pool(),
            divergence_state=_make_divergence_state(),
            llm_client=client,
            archive=archive,
            feedback_store=feedback_store,
            domain_id="work",
        )
        assert result is not None, "Unsuppressed domain should produce output."
        assert archive.count("u_001") == 1


class TestMirrorSuccessfulGeneration:
    """Full Stage 4 pipeline run produces a valid InsightRecord."""

    def test_stage_4_produces_insight_record(self):
        cold_state = _make_cold_start_state(
            stage=ColdStartStage.STAGE_4, day=95, can_surface=True
        )
        archive = InsightArchive()
        client = _make_llm_client()

        result = run_mirror_pipeline(
            user_id="u_001",
            cold_start_state=cold_state,
            claims=[_make_claim(ClaimLevel.LEVEL_2)],
            candidate_excerpts=_make_excerpts_pool(),
            divergence_state=_make_divergence_state(),
            llm_client=client,
            archive=archive,
            tone=ToneMode.REFLECTIVE,
            domain_id="work",
        )
        assert result is not None
        assert isinstance(result, InsightRecord)
        assert result.user_id == "u_001"
        assert result.text != ""
        assert result.tone == ToneMode.REFLECTIVE

    def test_generated_insight_archived(self):
        cold_state = _make_cold_start_state(
            stage=ColdStartStage.STAGE_4, day=95, can_surface=True
        )
        archive = InsightArchive()
        client = _make_llm_client()

        result = run_mirror_pipeline(
            user_id="u_001",
            cold_start_state=cold_state,
            claims=[_make_claim()],
            candidate_excerpts=_make_excerpts_pool(),
            divergence_state=_make_divergence_state(),
            llm_client=client,
            archive=archive,
        )
        assert result is not None
        assert archive.count("u_001") == 1
        retrieved = archive.get("u_001", result.insight_id)
        assert retrieved.insight_id == result.insight_id

    @pytest.mark.parametrize("tone", list(ToneMode))
    def test_all_tone_modes_produce_output(self, tone: ToneMode):
        cold_state = _make_cold_start_state(
            stage=ColdStartStage.STAGE_4, day=95, can_surface=True
        )
        archive = InsightArchive()
        client = _make_llm_client()

        result = run_mirror_pipeline(
            user_id="u_001",
            cold_start_state=cold_state,
            claims=[_make_claim()],
            candidate_excerpts=_make_excerpts_pool(),
            divergence_state=_make_divergence_state(),
            llm_client=client,
            archive=archive,
            tone=tone,
        )
        assert result is not None, f"Expected output for tone={tone.name}"
        assert result.tone == tone


class TestMirrorAdaptiveThresholdGate:
    """Mirror is silent when divergence confidence < user's adaptive threshold."""

    def test_low_confidence_blocked_by_adaptive_threshold(self):
        cold_state = _make_cold_start_state(
            stage=ColdStartStage.STAGE_4, day=95, can_surface=True
        )
        feedback_store = AdaptiveFeedbackStore()
        # Apply many NOT_YET to raise threshold well above 0.50
        for i in range(10):
            feedback_store.record_feedback("u_001", f"ins_{i}", FeedbackRating.NOT_YET)
        raised_threshold = feedback_store.get_admissibility_threshold("u_001")
        assert raised_threshold > 0.55, "Precondition: threshold must be raised"

        archive = InsightArchive()
        client = _make_llm_client()

        # Divergence confidence BELOW the raised threshold
        low_ds = _make_divergence_state(confidence=0.52)

        result = run_mirror_pipeline(
            user_id="u_001",
            cold_start_state=cold_state,
            claims=[_make_claim()],
            candidate_excerpts=_make_excerpts_pool(),
            divergence_state=low_ds,
            llm_client=client,
            archive=archive,
            feedback_store=feedback_store,
        )
        assert result is None, (
            f"Expected None (threshold={raised_threshold:.2f} > confidence=0.52)"
        )
        client.generate.assert_not_called()


# ===========================================================================
# DoD: 20-run end-to-end pipeline integration test
# ===========================================================================

class TestMirror20RunEndToEnd:
    """
    DoD: 20 actual Mirror pipeline runs, each exercising:
        generator → select_excerpts → user vocab → LLM call → clinical filter
        → citation chain → specificity linter → archive.append()

    This is distinct from TestSpecificityLinter20Sample, which only tests
    the linter gate on synthetic text variants.

    Each run uses a different user_id, tone, domain, and claim level to
    maximise coverage.
    """

    # 20 insight texts that pass all gates (100–200 words, no banned patterns,
    # grounded in excerpt vocabulary).
    _INSIGHT_TEMPLATES = [
        (
            "Over the past several sessions you have shown a consistent pattern of returning "
            "to focused work even when context switches were planned in advance. "
            "In session s0, the pull toward the project felt automatic rather than deliberate, "
            "suggesting the attractor basin is becoming stronger than your scheduled transitions. "
            "In session s1, the same momentum appears, persisting through external interruptions "
            "that slowed but did not break your concentration in any meaningful way. "
            "The near-miss session s_nm shows conditions for deep work began but the sustained "
            "state did not fully materialise, making the contrast more legible in the data. "
            "This pattern points to an engagement pull your current schedule may not yet account for."
        ),
        (
            "Across recent sessions you have returned consistently to the same project even "
            "when your calendar suggested a shift in focus was planned for that period. "
            "Session s0 shows the pull toward sustained engagement was stronger than the "
            "scheduled context switch, with momentum carrying through beyond the planned break. "
            "Session s1 reinforces this with a similar pattern: the interruption slowed but "
            "did not break the sustained state, leaving you further into the project than planned. "
            "The near-miss session s_nm is the clearest contrast: the conditions were present "
            "but the depth did not arrive, which makes the other sessions more interpretable. "
            "This suggests a deepening attractor basin in the data that your schedule may undercount."
        ),
        (
            "What the sessions from this period show is a pull toward sustained work that "
            "operates somewhat independently of your planned transitions and context switches. "
            "In session s0, the momentum toward the project persisted past the point where "
            "you had intended to shift, suggesting the attractor is gaining strength over time. "
            "Session s1 shows the same dynamic, this time in the presence of external interruptions "
            "that reduced but did not eliminate the sustained engagement across the full period. "
            "Session s_nm provides a useful contrast: the start conditions were similar to s0 "
            "but the depth did not arrive, making the difference between sessions more visible. "
            "The pattern across these sessions points to something your schedule may not fully capture."
        ),
        (
            "Looking at the pattern across recent sessions, there is a consistency in how you "
            "return to sustained focus even when planned transitions suggest otherwise for the day. "
            "Session s0 is the clearest example: momentum toward the project carried through a "
            "scheduled context switch, leaving you further into the work than originally intended. "
            "Session s1 shows a variation: external interruptions were present but the sustained "
            "state persisted through them, suggesting the attractor basin has become more robust. "
            "The near-miss session s_nm is notable because the conditions matched s0 but the "
            "depth did not arrive, which makes the contrast between sessions more readable overall. "
            "This points to an emerging pattern in the data that your current weekly structure may miss."
        ),
    ]

    def _make_insight_for_run(self, run_index: int) -> str:
        """Pick a template and return it as-is (all are ≥100 words, grounded)."""
        return self._INSIGHT_TEMPLATES[run_index % len(self._INSIGHT_TEMPLATES)]

    @pytest.mark.parametrize("run_index", list(range(20)))
    def test_mirror_20_run_end_to_end(self, run_index: int):
        """
        20 actual end-to-end pipeline runs.

        Each run:
        - Has its own user_id, archive, feedback_store
        - Calls run_mirror_pipeline with all gates active
        - Uses a real _make_llm_client() stub (LLM protocol call counted)
        - Verifies the LLM was called exactly once
        - Verifies the result is an InsightRecord (not None)
        - Verifies the result is archived
        - Verifies the citation chain covers every sentence
        - Verifies the linter passed (100–200 words, no banned patterns)
        - Verifies the tone matches what was requested
        """
        tone_choices = list(ToneMode)
        tone = tone_choices[run_index % len(tone_choices)]
        user_id = f"e2e_user_{run_index:03d}"
        domain_id = ["work", "relationships", "health", "creativity"][run_index % 4]
        claim_level = [ClaimLevel.LEVEL_1, ClaimLevel.LEVEL_2, ClaimLevel.LEVEL_3][
            run_index % 3
        ]

        cold_state = _make_cold_start_state(
            stage=ColdStartStage.STAGE_4, day=90 + run_index, can_surface=True
        )
        archive = InsightArchive()
        feedback_store = AdaptiveFeedbackStore()
        client = _make_llm_client(response=self._make_insight_for_run(run_index))

        result = run_mirror_pipeline(
            user_id=user_id,
            cold_start_state=cold_state,
            claims=[_make_claim(level=claim_level, domain_id=domain_id)],
            candidate_excerpts=_make_excerpts_pool(n_supporting=4),
            divergence_state=_make_divergence_state(confidence=0.75),
            llm_client=client,
            archive=archive,
            feedback_store=feedback_store,
            tone=tone,
            domain_id=domain_id,
        )

        # LLM must have been called exactly once
        client.generate.assert_called_once()

        # Must produce an InsightRecord
        assert result is not None, (
            f"Run {run_index}: expected InsightRecord but got None. "
            f"(stage=STAGE_4, can_surface=True, confidence=0.75)"
        )
        assert isinstance(result, InsightRecord)

        # Must be archived
        assert archive.count(user_id) == 1
        archived = archive.get(user_id, result.insight_id)
        assert archived.insight_id == result.insight_id

        # Tone must be preserved
        assert result.tone == tone, f"Run {run_index}: tone mismatch"

        # Every sentence must have a citation
        sentences = re.split(r"(?<=[.!?])\s+", result.text.strip())
        cited_indices = {e.sentence_index for e in result.citation_chain}
        for i in range(len(sentences)):
            assert i in cited_indices, (
                f"Run {run_index}: sentence {i} has no citation anchor. "
                f"text={result.text!r}"
            )

        # Linter must pass (word count in range, no banned patterns)
        lint_result = lint_insight(result.text, list(result.citation_chain))
        assert lint_result.status == LintStatus.PASS, (
            f"Run {run_index}: linter FAIL. violations={lint_result.violations}"
        )


# ===========================================================================
# Clinical hard-stop tests
# ===========================================================================

class TestClinicalHardStop:
    """
    Clinical terminology must trigger an immediate hard stop:
      - ClinicalTerminologyError raised from generator
      - Pipeline returns None (nothing surfaced to user)
      - Pipeline archives a PENDING HUMAN REVIEW record
      - LLM was called (generation happened) but result not surfaced
    """

    def _make_clinical_client(self) -> MagicMock:
        """Stub LLM that returns text containing clinical terminology."""
        clinical_text = (
            "Over the past several sessions you have shown signs of anxiety and depression "
            "that are consistent with a clinical diagnosis requiring immediate therapeutic "
            "intervention. Session s0 data shows these anxiety markers persistently. "
            "Session s1 reinforces the depression indicators across the observation period. "
            "The near-miss session s_nm shows you nearly reached a healthier state but the "
            "diagnostic threshold was not crossed, making the clinical picture more complex. "
            "This pattern suggests a treatment pathway that warrants clinical evaluation now."
        )
        client = MagicMock()
        client.generate.return_value = clinical_text
        return client

    def test_generator_raises_clinical_terminology_error(self):
        """Generator must raise ClinicalTerminologyError on clinical output."""
        excerpts = _make_excerpts_pool()
        claim = _make_claim(ClaimLevel.LEVEL_2)
        ds = _make_divergence_state()
        client = self._make_clinical_client()
        gen = MirrorInsightGenerator(llm_client=client)
        with pytest.raises(ClinicalTerminologyError) as exc_info:
            gen.generate(
                user_id="u_001",
                claims=[claim],
                candidate_excerpts=excerpts,
                divergence_state=ds,
            )
        assert exc_info.value.user_id == "u_001"
        assert exc_info.value.term  # must name the triggering term

    def test_pipeline_returns_none_on_clinical_hit(self):
        """Pipeline must return None when clinical terminology is detected."""
        cold_state = _make_cold_start_state(
            stage=ColdStartStage.STAGE_4, day=95, can_surface=True
        )
        archive = InsightArchive()
        client = self._make_clinical_client()

        result = run_mirror_pipeline(
            user_id="u_001",
            cold_start_state=cold_state,
            claims=[_make_claim()],
            candidate_excerpts=_make_excerpts_pool(),
            divergence_state=_make_divergence_state(),
            llm_client=client,
            archive=archive,
        )
        assert result is None, "Clinical hit: pipeline must return None (no output surfaced)"

    def test_pipeline_archives_human_review_record_on_clinical_hit(self):
        """Pipeline must archive a PENDING HUMAN REVIEW record even when returning None."""
        cold_state = _make_cold_start_state(
            stage=ColdStartStage.STAGE_4, day=95, can_surface=True
        )
        archive = InsightArchive()
        client = self._make_clinical_client()

        run_mirror_pipeline(
            user_id="u_001",
            cold_start_state=cold_state,
            claims=[_make_claim()],
            candidate_excerpts=_make_excerpts_pool(),
            divergence_state=_make_divergence_state(),
            llm_client=client,
            archive=archive,
        )
        # Archive should contain exactly one record (the human-review stub)
        assert archive.count("u_001") == 1
        records = archive.list_all("u_001")
        assert records[0].routed_to_human_review is True
        assert "PENDING HUMAN REVIEW" in records[0].text or "clinical" in records[0].human_review_reason.lower()

    def test_llm_was_called_on_clinical_hit(self):
        """The LLM was invoked (generation happened) but result was not surfaced."""
        cold_state = _make_cold_start_state(
            stage=ColdStartStage.STAGE_4, day=95, can_surface=True
        )
        archive = InsightArchive()
        client = self._make_clinical_client()

        run_mirror_pipeline(
            user_id="u_001",
            cold_start_state=cold_state,
            claims=[_make_claim()],
            candidate_excerpts=_make_excerpts_pool(),
            divergence_state=_make_divergence_state(),
            llm_client=client,
            archive=archive,
        )
        # LLM was called but output was not surfaced
        client.generate.assert_called_once()
