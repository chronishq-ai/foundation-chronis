"""
claims_engine/grounded_generation.py

Sprint 9, Day 27 — constrained retrieval-augmented generation pipeline.

Retrieves the 3 highest-contribution supporting sessions + 1 deliberate
"near-miss" counter-example, passes all 4 excerpts + divergence state + claim
level to a constrained system prompt, and logs a full citation chain so every
generated sentence is traceable to its source excerpt.

*** Model call is abstracted behind `LLMClient` Protocol. ***
Per the directive: any LLM used here must be self-hosted, never a third-party
API. Wire your actual self-hosted inference client into `LLMClient` — this
module never imports or calls a third-party API client itself.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Protocol, Sequence
import re
import uuid

from upstream_interfaces import SessionExcerpt
from .claim_levels import Claim, ClaimLevel
from divergence_engine.state import DivergenceState


CLINICAL_TERMS = [
    "depression", "depressed", "anxiety", "anxious", "trauma", "traumatic",
    "disorder", "diagnosis", "diagnose", "pathology", "pathological",
]

SIX_MONTH_REVIEW_WINDOW = timedelta(days=182)

CONSTRAINED_SYSTEM_PROMPT = """You are generating a short, grounded reflection for one person, based ONLY \
on the excerpts and divergence data provided. Rules, all mandatory:
1. Do not diagnose. Never use clinical/medical terminology.
2. Do not name emotions the person has not named themselves.
3. Be specific, not general — refer to what actually happened in the excerpts.
4. Be tentative, not certain — this is a pattern observation, not a fact about who they are.
5. Maximum 3 sentences.
6. Every sentence must be traceable to one of the provided excerpts."""


class LLMClient(Protocol):
    """Self-hosted inference client contract. Implement against your own model server."""
    def generate(self, system_prompt: str, user_content: str) -> str: ...


@dataclass(frozen=True)
class CitationChainEntry:
    sentence_index: int
    sentence_text: str
    source_session_id: str


@dataclass(frozen=True)
class GeneratedInsight:
    insight_id: str
    claim_id: str
    text: str
    citation_chain: Sequence[CitationChainEntry]
    routed_to_human_review: bool
    human_review_reason: Optional[str]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def select_excerpts(candidates: Sequence[SessionExcerpt], n_supporting: int = 3) -> List[SessionExcerpt]:
    """
    Retrieve the n highest-contribution supporting sessions PLUS exactly one
    deliberate near-miss counter-example (a session that approached the
    attractor basin but did not enter it).
    """
    supporting = sorted(
        [c for c in candidates if not c.is_near_miss],
        key=lambda c: c.contribution_score,
        reverse=True,
    )[:n_supporting]

    near_misses = [c for c in candidates if c.is_near_miss]
    if not near_misses:
        raise ValueError(
            "No near-miss counter-example session available. Per Sprint 9 Day 27, "
            "generation requires exactly one near-miss excerpt — do not proceed without it."
        )
    near_miss = max(near_misses, key=lambda c: c.contribution_score)

    return supporting + [near_miss]


def contains_clinical_terminology(text: str) -> Optional[str]:
    lowered = text.lower()
    for term in CLINICAL_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            return term
    return None


def _build_user_content(excerpts: Sequence[SessionExcerpt], divergence_state: DivergenceState, level: ClaimLevel) -> str:
    excerpt_block = "\n".join(
        f"[{i+1}] (session {e.session_id}{', NEAR-MISS' if e.is_near_miss else ''}): {e.text}"
        for i, e in enumerate(excerpts)
    )
    dominant = divergence_state.type_scores.dominant()
    return (
        f"Claim level: {level.name}\n"
        f"Dominant divergence type: {dominant}\n"
        f"Divergence confidence: {divergence_state.confidence:.2f}\n\n"
        f"Excerpts:\n{excerpt_block}"
    )


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _naive_attribute_sentence_to_excerpt(sentence: str, excerpts: Sequence[SessionExcerpt]) -> Optional[str]:
    """
    Placeholder attribution heuristic: attributes a generated sentence to the
    excerpt with the highest lexical token overlap. In production this should
    be replaced by whatever citation mechanism your self-hosted LLM client
    supports natively (e.g. structured output with inline source tags) —
    this naive version exists so citation-chain logging is testable end-to-end
    without a real model in the loop.
    """
    sentence_tokens = set(re.findall(r"\w+", sentence.lower()))
    best_id, best_overlap = None, -1
    for e in excerpts:
        excerpt_tokens = set(re.findall(r"\w+", e.text.lower()))
        overlap = len(sentence_tokens & excerpt_tokens)
        if overlap > best_overlap:
            best_overlap, best_id = overlap, e.session_id
    return best_id


def generate_insight(
    claim: Claim,
    divergence_state: DivergenceState,
    candidate_excerpts: Sequence[SessionExcerpt],
    llm_client: LLMClient,
) -> GeneratedInsight:
    if claim.level not in (ClaimLevel.LEVEL_2, ClaimLevel.LEVEL_3):
        raise ValueError("Grounded generation is defined for Level 2/3 claims in this pipeline.")

    excerpts = select_excerpts(candidate_excerpts)
    user_content = _build_user_content(excerpts, divergence_state, claim.level)

    raw_text = llm_client.generate(CONSTRAINED_SYSTEM_PROMPT, user_content)

    clinical_hit = contains_clinical_terminology(raw_text)
    routed_to_human_review = bool(clinical_hit) or claim.level == ClaimLevel.LEVEL_3
    review_reason = None
    if clinical_hit:
        review_reason = f"Clinical terminology filter triggered on term: '{clinical_hit}'"
    elif claim.level == ClaimLevel.LEVEL_3:
        review_reason = "Standing 6-month mandatory human-review requirement for all Level 3 text."

    sentences = _split_sentences(raw_text)
    if len(sentences) > 3:
        raise ValueError(f"Generated insight exceeds the 3-sentence maximum ({len(sentences)} sentences).")

    citation_chain: List[CitationChainEntry] = []
    for i, sentence in enumerate(sentences):
        source_id = _naive_attribute_sentence_to_excerpt(sentence, excerpts)
        if source_id is None:
            raise ValueError(f"Sentence {i} could not be attributed to any source excerpt — refusing to ship it.")
        citation_chain.append(CitationChainEntry(sentence_index=i, sentence_text=sentence, source_session_id=source_id))

    return GeneratedInsight(
        insight_id=str(uuid.uuid4()),
        claim_id=claim.claim_id,
        text=raw_text.strip(),
        citation_chain=citation_chain,
        routed_to_human_review=routed_to_human_review,
        human_review_reason=review_reason,
    )
