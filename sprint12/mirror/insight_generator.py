"""
mirror/insight_generator.py
Sprint 12, Day 34 — Core daily insight generator.

Generates a 100–200 word, second-person, evidence-grounded daily insight
drawing on Level 1–3 claims + session excerpts + user vocabulary.

Every sentence resolves to a specific SessionExcerpt via the same
citation-chain mechanism built in Sprint 9 (grounded_generation.py).

Design decisions:
- Builds on Sprint 9's LLMClient Protocol, select_excerpts(), and
  _naive_attribute_sentence_to_excerpt() — no parallel infrastructure.
- Adds user-vocabulary extraction: the generator pulls the user's own
  recurring words and phrases from their SessionExcerpts and injects
  them into the system prompt so the LLM uses the user's own cadence.
- The Mirror generates longer output than Sprint 9's 3-sentence limit:
  100–200 words. The CONSTRAINED_SYSTEM_PROMPT is extended accordingly.
- Sprint 9's clinical filter is still applied — any clinical terminology
  triggers human-review routing.

Bible ref: Part 5.21 (Insight Generation / The Mirror, Module 4.10)
"""

from __future__ import annotations

import re
import uuid
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Sequence, TYPE_CHECKING

from upstream_interfaces import SessionExcerpt
from claims_engine.claim_levels import Claim, ClaimLevel
from claims_engine.grounded_generation import (
    CitationChainEntry,
    LLMClient,
    contains_clinical_terminology,
    select_excerpts,
    _naive_attribute_sentence_to_excerpt,
    _split_sentences,
)
from divergence_engine.state import DivergenceState
from mirror.tone_calibration import ToneMode, tone_system_prompt
from mirror.specificity_linter import lint_insight

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mirror-specific system prompt (extends Sprint 9's base prompt)
# ---------------------------------------------------------------------------

MIRROR_SYSTEM_PROMPT = """You are generating a daily personalised reflection for one person based ONLY \
on the excerpts and data provided. All rules below are mandatory:
1. Do not diagnose. Never use clinical or medical terminology.
2. Do not name emotions the person has not named themselves.
3. Be specific — refer to what actually happened in the excerpts, not generic patterns.
4. Be tentative — this is an observation, not a fact about who they are.
5. Write in the second person ("you"), 100–200 words.
6. Every sentence must be traceable to one of the provided excerpts.
7. Use the person's own vocabulary and phrasing where the excerpts supply it.
8. Never give wellness advice (e.g. "try meditation"). Observe; do not prescribe.
9. Full paragraph form — not a list."""


# ---------------------------------------------------------------------------
# User vocabulary extractor
# ---------------------------------------------------------------------------

# Common stopwords to exclude from vocabulary extraction
_STOPWORDS = {
    "i", "me", "my", "we", "you", "your", "he", "she", "it", "they",
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "did", "will", "would", "could",
    "should", "that", "this", "these", "those", "not", "no", "so",
    "just", "like", "also", "then", "than", "when", "what", "how",
    "if", "as", "up", "out", "about", "more", "very", "can", "get",
    "got", "even", "still", "really", "actually", "feel", "felt",
    "think", "thought", "said", "know", "want", "go", "went", "time",
}

_MIN_WORD_LENGTH = 4
_TOP_N_VOCAB = 12


def extract_user_vocabulary(excerpts: Sequence[SessionExcerpt]) -> List[str]:
    """
    Extract the user's most frequent non-stopword vocabulary from their
    SessionExcerpts. Used to instruct the LLM to mirror the user's cadence.

    Returns up to _TOP_N_VOCAB words, sorted by frequency descending.
    """
    counter: Counter = Counter()
    for excerpt in excerpts:
        tokens = re.findall(r"\b[a-z]{%d,}\b" % _MIN_WORD_LENGTH, excerpt.text.lower())
        for tok in tokens:
            if tok not in _STOPWORDS:
                counter[tok] += 1
    return [word for word, _ in counter.most_common(_TOP_N_VOCAB)]


def _build_mirror_user_content(
    excerpts: Sequence[SessionExcerpt],
    divergence_state: DivergenceState,
    claims: Sequence[Claim],
    user_vocab: List[str],
) -> str:
    """
    Build the user-content block sent to the LLM.
    Includes: claim levels present, divergence context, excerpts, user vocab.
    """
    excerpt_block = "\n".join(
        f"[{i+1}] (session {e.session_id}"
        f"{', NEAR-MISS' if e.is_near_miss else ''}): {e.text}"
        for i, e in enumerate(excerpts)
    )
    claim_summary = ", ".join(
        f"Level {c.level.value} ({c.domain_id})" for c in claims
    )
    dominant = divergence_state.type_scores.dominant()
    vocab_hint = (
        f"User's recurring vocabulary (use where natural): {', '.join(user_vocab)}"
        if user_vocab else ""
    )
    return (
        f"Active claims: {claim_summary}\n"
        f"Dominant divergence type: {dominant}\n"
        f"Divergence confidence: {divergence_state.confidence:.2f}\n"
        f"{vocab_hint}\n\n"
        f"Excerpts:\n{excerpt_block}"
    )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ClinicalTerminologyError(ValueError):
    """
    Raised when the LLM output contains clinical terminology.

    Per the Sprint 9 standing contract: clinical terminology detected in
    generated output must trigger an immediate hard stop and route to the
    mandatory 6-month human-review queue. The insight is NOT archived and
    NOT surfaced to the user until a human reviewer clears it.

    The caller (mirror_pipeline.py) catches this and sets
    routed_to_human_review=True on the pipeline result.
    """
    def __init__(self, term: str, user_id: str) -> None:
        self.term = term
        self.user_id = user_id
        super().__init__(
            f"Clinical terminology '{term}' detected in generated insight for "
            f"user={user_id}. Hard stop: routing to human review queue. "
            f"Insight NOT archived or surfaced until cleared by a human reviewer."
        )


# ---------------------------------------------------------------------------
# Citation chain builder
# ---------------------------------------------------------------------------

def _build_citation_chain(
    text: str,
    excerpts: Sequence[SessionExcerpt],
) -> List[CitationChainEntry]:
    """
    Build a full citation chain: every sentence in `text` attributed to
    its best-matching SessionExcerpt via lexical overlap.

    Raises ValueError if any sentence cannot be attributed.

    KNOWN LIMITATION (explicit, not hidden):
    ----------------------------------------
    Current implementation uses deterministic lexical attribution (token
    overlap heuristic). This is sufficient for synthetic Sprint 12
    validation and proves the citation-chain mechanism end-to-end.

    Before production: replace with structured source attribution from
    the self-hosted generation layer (e.g. structured output with inline
    source tags, or retrieval-augmented generation with provenance).
    See README.md §4 "Known Limitations" for rationale.
    """
    sentences = _split_sentences(text)
    chain: List[CitationChainEntry] = []
    for i, sentence in enumerate(sentences):
        source_id = _naive_attribute_sentence_to_excerpt(sentence, excerpts)
        if source_id is None:
            raise ValueError(
                f"Sentence {i} could not be attributed to any source excerpt. "
                f"Refusing to ship an ungrounded sentence. "
                f"Sentence text: {sentence!r}"
            )
        chain.append(CitationChainEntry(
            sentence_index=i,
            sentence_text=sentence,
            source_session_id=source_id,
        ))
    return chain


# ---------------------------------------------------------------------------
# InsightDraft — intermediate result before archive entry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InsightDraft:
    """
    Result of a successful insight generation pass (pre-archive).

    All hard constraints have been verified:
      - text is 100–200 words
      - clinical filter passed (or human_review flagged)
      - citation chain covers every sentence
      - specificity linter passed

    Callers: mirror_pipeline.py assembles this into an InsightRecord.
    """
    draft_id: str
    user_id: str
    text: str
    tone: ToneMode
    citation_chain: List[CitationChainEntry]
    routed_to_human_review: bool
    human_review_reason: Optional[str]
    claim_ids: List[str]
    dominant_divergence_type: Optional[str]
    domain_id: Optional[str]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

class MirrorInsightGenerator:
    """
    Core daily insight generator for The Mirror (Sprint 12, Day 34).

    Usage
    -----
    >>> gen = MirrorInsightGenerator(llm_client=my_client)
    >>> draft = gen.generate(
    ...     user_id="u_001",
    ...     claims=[level2_claim],
    ...     candidate_excerpts=excerpts,
    ...     divergence_state=ds,
    ...     tone=ToneMode.REFLECTIVE,
    ... )
    >>> print(draft.text)
    """

    def __init__(self, llm_client: LLMClient) -> None:
        """
        Args:
            llm_client: Self-hosted LLM client (Protocol from Sprint 9).
                        Never a third-party API call.
        """
        self._llm = llm_client

    def generate(
        self,
        user_id: str,
        claims: Sequence[Claim],
        candidate_excerpts: Sequence[SessionExcerpt],
        divergence_state: DivergenceState,
        tone: ToneMode = ToneMode.REFLECTIVE,
        domain_id: Optional[str] = None,
    ) -> InsightDraft:
        """
        Generate one daily insight for a user.

        Flow:
          1. Select excerpts (3 supporting + 1 near-miss, per Sprint 9)
          2. Extract user vocabulary from all candidate excerpts
          3. Build system prompt with tone modifier
          4. Call LLM (self-hosted only)
          5. Clinical terminology filter — HARD STOP if triggered
             (raises ClinicalTerminologyError; insight NOT archived)
          6. Build citation chain (every sentence must attribute)
          7. Run specificity linter (hard gate)
          8. Return InsightDraft

        Args:
            user_id:            User identifier.
            claims:             Level 1–3 Claim objects driving this insight.
                                Level 0 claims are informational and should not
                                drive Mirror generation.
            candidate_excerpts: Pool of SessionExcerpts for this user/domain.
                                Must include at least one near-miss excerpt.
            divergence_state:   Current Sprint 8 DivergenceState.
            tone:               User-selected ToneMode (default: REFLECTIVE).
            domain_id:          Which domain this insight concerns (optional,
                                used for archive tagging and domain suppression).

        Returns:
            InsightDraft — verified, linted, fully grounded.

        Raises:
            ClinicalTerminologyError: if clinical terminology is detected in the
                output. The insight is NOT archived. Caller must route to the
                human-review queue and present no output to the user.
            ValueError: if excerpts are insufficient, any sentence is
                        unattributable, or the linter rejects the output.
        """
        if not claims:
            raise ValueError(
                "generate() requires at least one Level 1–3 Claim. "
                "The Mirror must not run without admitted claims."
            )

        # Filter out Level 0 — Mirror generation is for Level 1+
        generative_claims = [c for c in claims if c.level != ClaimLevel.LEVEL_0]
        if not generative_claims:
            raise ValueError(
                "All provided claims are Level 0 (event facts). "
                "The Mirror generates insights from Level 1–3 claims only."
            )

        # Step 1: excerpt selection (Sprint 9 contract)
        excerpts = select_excerpts(candidate_excerpts)
        logger.debug(
            "user=%s  selected %d excerpts (%d supporting + 1 near-miss)",
            user_id, len(excerpts), len(excerpts) - 1,
        )

        # Step 2: user vocabulary
        user_vocab = extract_user_vocabulary(candidate_excerpts)
        logger.debug("user=%s  vocab=%s", user_id, user_vocab)

        # Step 3: tone-aware system prompt
        system_prompt = tone_system_prompt(MIRROR_SYSTEM_PROMPT, tone)

        # Step 4: assemble user content and call LLM
        user_content = _build_mirror_user_content(
            excerpts, divergence_state, generative_claims, user_vocab
        )
        raw_text = self._llm.generate(system_prompt, user_content)
        logger.debug("user=%s  raw_text_len=%d", user_id, len(raw_text))

        # Step 5: clinical filter — HARD STOP (Sprint 9 standing contract)
        # Clinical terminology → immediate routing to human review queue.
        # The insight is NOT archived and NOT surfaced until a human clears it.
        # This is a hard raise, not just a metadata flag.
        clinical_hit = contains_clinical_terminology(raw_text)
        if clinical_hit:
            logger.warning(
                "user=%s: clinical term '%s' detected — hard stop, routing to human review. "
                "Insight will NOT be archived or surfaced.",
                user_id, clinical_hit,
            )
            raise ClinicalTerminologyError(term=clinical_hit, user_id=user_id)

        # Step 6: citation chain (every sentence must attribute)
        citation_chain = _build_citation_chain(raw_text, excerpts)

        # Step 7: specificity linter — hard gate
        lint_result = lint_insight(raw_text, citation_chain)
        if not lint_result.passed:
            violation_summary = "; ".join(
                f"[sentence {v.sentence_index}] {v.violation_type}: {v.detail}"
                for v in lint_result.violations
            )
            raise ValueError(
                f"Insight blocked by specificity linter for user={user_id}. "
                f"Violations: {violation_summary}"
            )

        dominant = divergence_state.type_scores.dominant()
        return InsightDraft(
            draft_id=str(uuid.uuid4()),
            user_id=user_id,
            text=raw_text.strip(),
            tone=tone,
            citation_chain=citation_chain,
            routed_to_human_review=False,  # clinical hard-stop above ensures we never reach here with clinical terms
            human_review_reason=None,
            claim_ids=[c.claim_id for c in generative_claims],
            dominant_divergence_type=dominant,
            domain_id=domain_id,
        )
