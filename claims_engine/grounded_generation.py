"""
claims_engine/grounded_generation.py

Sprint 9, Day 27 — constrained retrieval-augmented generation pipeline.

Retrieves the 3 highest-contribution supporting sessions + 1 deliberate
"near-miss" counter-example, passes all 4 excerpts + divergence state + claim
level to a constrained system prompt, and logs a full citation chain so every
generated sentence is traceable to its source excerpt.

S79.7 / XCUT-2 FIX (semantic safety): the previous 11-term substring denylist
(`contains_clinical_terminology`) is deleted. It could not catch indirect
diagnostic framing ("it sounds like a pattern consistent with how people
struggling with their mood often act"), clinical synonyms that dodge the
literal word list, or severe-distress/self-harm language that isn't a
"clinical term" at all. Replaced with `evaluate_clinical_safety`, an
LLM-based semantic classifier returning a structured SAFE /
HUMAN_REVIEW_REQUIRED / REJECT verdict. Exported cleanly for Sprint 11/12 to
import directly.

S79.6 FIX (semantic grounding): the previous `_naive_attribute_sentence_to_excerpt`
picked whichever excerpt had the highest raw token overlap with a generated
sentence — that is a similarity heuristic, not a grounding check, and cannot
detect negation inversions ("often stayed" attributed to an excerpt about
someone who "rarely stayed") or confident-sounding hallucinated specifics.
Replaced with `verify_claim_entailment`, an LLM-based entailment judge that
must find the source excerpt(s) STRICTLY entail the generated sentence,
explicitly rejects negation inversions and unsupported additions, and
returns an ABSTAIN verdict (not a guess) when citation coverage is
incomplete or ambiguous.

*** Model calls are abstracted behind `LLMClient`. ***
Per the standing directive, any LLM wired into `LLMClient` must be
self-hosted, never a third-party API — this module never imports or calls a
third-party API client itself. The Protocol below mirrors the common
`client.chat.completions.create(...)` calling convention purely for
interface familiarity with self-hosted inference gateways that expose an
OpenAI-compatible surface; it is not sanction to point this at a third-party
endpoint.

IMPLEMENTATION NOTE — read before trusting this in production:
Both `evaluate_clinical_safety` and `verify_claim_entailment` are only as
good as (a) the deployed judge model and (b) the prompts below. The prompts
here are a solid production starting point, not a substitute for evaluation
against a labeled test set (known-unsafe generations, known negation-inversion
cases, known-good citations) before this gates real output. Both functions
fail CLOSED on any parse error, malformed response, or unexpected field —
REJECT for safety, ABSTAIN for entailment — never fail open to SAFE/ENTAILED.
Don't loosen that without understanding what silently swallowing a parse
failure would mean for this pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence
import json
import re
import uuid

from upstream_interfaces import SessionExcerpt
from .claim_levels import Claim, ClaimLevel
from divergence_engine.state import DivergenceState


SIX_MONTH_REVIEW_WINDOW = timedelta(days=182)

CONSTRAINED_SYSTEM_PROMPT = """You are generating a short, grounded reflection for one person, based ONLY \
on the excerpts and divergence data provided. Rules, all mandatory:
1. Do not diagnose. Never use clinical/medical terminology.
2. Do not name emotions the person has not named themselves.
3. Be specific, not general — refer to what actually happened in the excerpts.
4. Be tentative, not certain — this is a pattern observation, not a fact about who they are.
5. Maximum 3 sentences.
6. Every sentence must be traceable to one of the provided excerpts."""

# Placeholder model identifiers — wire these to your actual deployed,
# self-hosted judge/classifier model IDs before shipping.
SAFETY_CLASSIFIER_MODEL = "self-hosted-clinical-safety-classifier-v1"
ENTAILMENT_JUDGE_MODEL = "self-hosted-entailment-judge-v1"
GENERATION_MODEL = "self-hosted-insight-generator-v1"


# ---------------------------------------------------------------------------
# Self-hosted LLM client contract
# ---------------------------------------------------------------------------

class _ChatCompletionMessage(Protocol):
    content: str


class _ChatCompletionChoice(Protocol):
    message: _ChatCompletionMessage


class _ChatCompletionResponse(Protocol):
    choices: List[_ChatCompletionChoice]


class _CompletionsNamespace(Protocol):
    def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> _ChatCompletionResponse: ...


class _ChatNamespace(Protocol):
    completions: _CompletionsNamespace


class LLMClient(Protocol):
    """Self-hosted inference client contract, `client.chat.completions.create(...)`-shaped."""
    chat: _ChatNamespace


def _call_llm_json(
    llm_client: LLMClient,
    model: str,
    system_prompt: str,
    user_content: str,
) -> Optional[Dict[str, Any]]:
    """
    Calls the LLM and parses its response as JSON. Returns None on ANY
    failure (malformed JSON, empty response, client exception, non-dict
    payload) so callers can fail closed rather than guessing at a default.
    """
    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
    except Exception:
        return None

    if not raw or not raw.strip():
        return None

    # Defensive strip in case the model wraps JSON in a code fence despite
    # response_format being requested — never trust format compliance blindly.
    cleaned = raw.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)

    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(parsed, dict):
        return None

    return parsed


# ---------------------------------------------------------------------------
# S79.7 / XCUT-2 — semantic clinical/safety classifier
# ---------------------------------------------------------------------------

class ClinicalSafetyVerdict(Enum):
    SAFE = "SAFE"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"
    REJECT = "REJECT"


@dataclass(frozen=True)
class SafetyClassification:
    verdict: ClinicalSafetyVerdict
    reasoning: str
    detected_concerns: Sequence[str]  # e.g. ["indirect_diagnostic_framing", "self_harm_language"]
    raw_model_response: Optional[Dict[str, Any]] = None


_SAFETY_CLASSIFIER_SYSTEM_PROMPT = """You are a clinical-safety classifier for short, AI-generated behavioral \
reflections that will be shown directly to the person they are about, with no clinician in the loop at \
generation time. You are the last automated check before that text could ship. Be conservative: when genuinely \
uncertain, prefer the stricter verdict.

Classify the CANDIDATE TEXT you are given against these categories:

1. DIRECT DIAGNOSIS: any explicit clinical/medical label, disorder name, or diagnostic term ("depression", \
"anxiety disorder", "trauma response", "ADHD", etc.), stated or implied to apply to this person.

2. INDIRECT DIAGNOSTIC FRAMING: language that does not use a clinical noun but still frames the person's \
behavior in clinical/pathologizing terms — e.g. "this is consistent with how people who struggle with their \
mental health often behave", "this pattern can be a sign of something deeper going on", "you may want to get \
this looked at". Reframing a diagnosis as a "pattern" or "tendency" does not make it non-diagnostic if the \
underlying claim is still a clinical one.

3. CLINICAL SYNONYMS / EUPHEMISM: words that route around an explicit denylist while carrying the same \
diagnostic meaning (e.g. "burnout" used as a stand-in for a clinical exhaustion/depressive diagnosis, "highly \
strung" as a stand-in for a clinical anxiety diagnosis) — judge by clinical MEANING, not by exact wording.

4. SEVERE DISTRESS OR SELF-HARM LANGUAGE: any language, direct or oblique, referencing self-harm, suicidal \
ideation, wanting to disappear/not exist, hopelessness framed as terminal, or similar. This category applies \
regardless of whether the language originated in the source excerpts or was introduced by the generation — its \
presence in candidate text bound for autonomous delivery is itself the concern, independent of provenance.

5. NON-CLINICAL, EMOTION-NAMING, OR OTHERWISE BENIGN: none of the above apply; the text stays at the level of \
observed behavior and does not attribute a clinical cause or self-harm risk.

Decide a verdict:
- "SAFE": none of categories 1-4 apply.
- "HUMAN_REVIEW_REQUIRED": category 2 or 3 applies (soft clinical framing/synonyms), but NOT category 1 or 4.
- "REJECT": category 1 (direct diagnosis) or category 4 (severe distress/self-harm language) applies. This text \
must not ship autonomously under any circumstances, including as a "softened" or "review-flagged" version — \
REJECT is a hard stop, and any case involving self-harm or suicide-adjacent language must be handled entirely \
outside this autonomous pipeline (routed to a human/crisis protocol out of band), never surfaced by this system.

Respond ONLY with a single JSON object, no prose outside it, of the exact shape:
{
  "verdict": "SAFE" | "HUMAN_REVIEW_REQUIRED" | "REJECT",
  "detected_concerns": [array of short machine-readable strings from: "direct_diagnosis", \
"indirect_diagnostic_framing", "clinical_synonym", "self_harm_or_severe_distress", or "none"],
  "reasoning": "one or two sentences explaining the verdict, quoting the specific phrase(s) that drove it if any"
}"""


def evaluate_clinical_safety(text: str, llm_client: LLMClient) -> SafetyClassification:
    """
    Semantic clinical/safety classifier for generated insight text. Exported
    for reuse by Sprints 11 and 12's pipelines — import this rather than
    reimplementing a local denylist.

    Fails CLOSED: any classifier-call failure or malformed response returns
    REJECT, never SAFE.
    """
    parsed = _call_llm_json(
        llm_client,
        model=SAFETY_CLASSIFIER_MODEL,
        system_prompt=_SAFETY_CLASSIFIER_SYSTEM_PROMPT,
        user_content=f"CANDIDATE TEXT:\n{text}",
    )

    if parsed is None:
        return SafetyClassification(
            verdict=ClinicalSafetyVerdict.REJECT,
            reasoning="Safety classifier call failed or returned unparseable output; failing closed to REJECT.",
            detected_concerns=["classifier_call_failed"],
            raw_model_response=None,
        )

    verdict_raw = parsed.get("verdict")
    try:
        verdict = ClinicalSafetyVerdict(verdict_raw)
    except (ValueError, TypeError):
        return SafetyClassification(
            verdict=ClinicalSafetyVerdict.REJECT,
            reasoning=f"Safety classifier returned an unrecognized verdict field ({verdict_raw!r}); failing closed to REJECT.",
            detected_concerns=["classifier_verdict_unparseable"],
            raw_model_response=parsed,
        )

    detected_concerns = parsed.get("detected_concerns", [])
    if not isinstance(detected_concerns, list):
        detected_concerns = [str(detected_concerns)]

    reasoning = parsed.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    return SafetyClassification(
        verdict=verdict,
        reasoning=reasoning,
        detected_concerns=[str(c) for c in detected_concerns],
        raw_model_response=parsed,
    )


# ---------------------------------------------------------------------------
# S79.6 — semantic entailment grounding check
# ---------------------------------------------------------------------------

class EntailmentVerdict(Enum):
    ENTAILED = "ENTAILED"
    NOT_ENTAILED = "NOT_ENTAILED"
    ABSTAIN = "ABSTAIN"  # citation coverage incomplete/ambiguous — do not guess


@dataclass(frozen=True)
class EntailmentResult:
    verdict: EntailmentVerdict
    source_session_id: Optional[str]  # only set when verdict == ENTAILED
    reasoning: str
    raw_model_response: Optional[Dict[str, Any]] = None


_ENTAILMENT_JUDGE_SYSTEM_PROMPT = """You are a strict entailment judge. You will be given a single CANDIDATE \
SENTENCE (one sentence from an AI-generated reflection) and a numbered list of SOURCE EXCERPTS. Your job is to \
determine whether the source excerpts, taken together, STRICTLY ENTAIL the candidate sentence — i.e. whether a \
careful reader of only the excerpts would agree the sentence is a faithful, non-speculative restatement or \
direct implication of what the excerpts say. This is a grounding check, not a similarity check — a sentence can \
share many words with an excerpt and still not be entailed by it, and can share few words and still be entailed.

Reject entailment (verdict "NOT_ENTAILED") for any of the following, even if the surface wording is close to an \
excerpt:
- NEGATION INVERSION: the candidate asserts something the excerpts assert the OPPOSITE of (e.g. candidate says \
"often" when the excerpt says "rarely" or "once"; candidate says a behavior occurred when the excerpt describes \
it being avoided or not happening).
- UNSUPPORTED SPECIFICS: the candidate adds a detail (a frequency, a cause, a comparison, a named emotion, a \
time period) that is not stated or directly implied by any excerpt, even if it sounds plausible.
- OVER-GENERALIZATION: the candidate generalizes from one excerpt's specific instance to a broader claim the \
excerpts don't support.
- WRONG ATTRIBUTION: the candidate's content is entailed by some excerpt, but not by the one your citation \
mechanism would naturally point to — always identify the CORRECT supporting excerpt by its bracketed index, \
never the most textually similar one if it isn't the one that actually supports the claim.

If the candidate is well-supported by a specific excerpt, return "ENTAILED" and the index of that excerpt. If \
support is spread thinly across multiple excerpts such that no single excerpt fully entails it, or if you are \
genuinely unsure whether the relationship is entailment vs. a plausible-but-unsupported extrapolation, return \
"ABSTAIN" — do not force a verdict. Do not reward confidence in the candidate's phrasing; judge only whether the \
excerpts support it.

Respond ONLY with a single JSON object, no prose outside it, of the exact shape:
{
  "verdict": "ENTAILED" | "NOT_ENTAILED" | "ABSTAIN",
  "supporting_excerpt_index": integer (1-based index into the numbered excerpt list) or null,
  "reasoning": "one or two sentences explaining the verdict, citing the specific excerpt content or the specific unsupported detail"
}"""


def verify_claim_entailment(
    sentence: str,
    excerpts: Sequence[SessionExcerpt],
    llm_client: LLMClient,
) -> EntailmentResult:
    """
    Verifies that `sentence` is strictly entailed by at least one of `excerpts`,
    via an LLM entailment judge — not lexical overlap. Returns ABSTAIN (not a
    best-guess source) when coverage is incomplete or ambiguous.

    Fails CLOSED: any judge-call failure or malformed response returns ABSTAIN,
    never a fabricated ENTAILED result.
    """
    if not excerpts:
        return EntailmentResult(
            verdict=EntailmentVerdict.ABSTAIN,
            source_session_id=None,
            reasoning="No excerpts provided to check entailment against.",
            raw_model_response=None,
        )

    excerpt_block = "\n".join(
        f"[{i + 1}] (session {e.session_id}{', NEAR-MISS' if e.is_near_miss else ''}): {e.text}"
        for i, e in enumerate(excerpts)
    )
    user_content = f"CANDIDATE SENTENCE:\n{sentence}\n\nSOURCE EXCERPTS:\n{excerpt_block}"

    parsed = _call_llm_json(
        llm_client,
        model=ENTAILMENT_JUDGE_MODEL,
        system_prompt=_ENTAILMENT_JUDGE_SYSTEM_PROMPT,
        user_content=user_content,
    )

    if parsed is None:
        return EntailmentResult(
            verdict=EntailmentVerdict.ABSTAIN,
            source_session_id=None,
            reasoning="Entailment judge call failed or returned unparseable output; failing closed to ABSTAIN.",
            raw_model_response=None,
        )

    verdict_raw = parsed.get("verdict")
    try:
        verdict = EntailmentVerdict(verdict_raw)
    except (ValueError, TypeError):
        return EntailmentResult(
            verdict=EntailmentVerdict.ABSTAIN,
            source_session_id=None,
            reasoning=f"Entailment judge returned an unrecognized verdict field ({verdict_raw!r}); failing closed to ABSTAIN.",
            raw_model_response=parsed,
        )

    reasoning = parsed.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    if verdict != EntailmentVerdict.ENTAILED:
        return EntailmentResult(
            verdict=verdict,
            source_session_id=None,
            reasoning=reasoning,
            raw_model_response=parsed,
        )

    idx = parsed.get("supporting_excerpt_index")
    if not isinstance(idx, int) or not (1 <= idx <= len(excerpts)):
        # Judge claimed ENTAILED but gave no valid pointer to a source — that's
        # an internally inconsistent response. Don't trust the ENTAILED verdict
        # without a resolvable source; fail closed to ABSTAIN.
        return EntailmentResult(
            verdict=EntailmentVerdict.ABSTAIN,
            source_session_id=None,
            reasoning=(
                "Judge returned ENTAILED but supporting_excerpt_index was missing/out of range "
                f"({idx!r}); failing closed to ABSTAIN. Original reasoning: {reasoning}"
            ),
            raw_model_response=parsed,
        )

    source_excerpt = excerpts[idx - 1]
    return EntailmentResult(
        verdict=EntailmentVerdict.ENTAILED,
        source_session_id=source_excerpt.session_id,
        reasoning=reasoning,
        raw_model_response=parsed,
    )


# ---------------------------------------------------------------------------
# Excerpt selection (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CitationChainEntry:
    sentence_index: int
    sentence_text: str
    source_session_id: str
    entailment_reasoning: str = ""


@dataclass(frozen=True)
class GeneratedInsight:
    insight_id: str
    claim_id: str
    text: str
    citation_chain: Sequence[CitationChainEntry]
    routed_to_human_review: bool
    human_review_reason: Optional[str]
    safety_classification: SafetyClassification
    generated_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

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

    try:
        raw_text = llm_client.chat.completions.create(
            model=GENERATION_MODEL, 
            messages=[
                {"role": "system", "content": CONSTRAINED_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        ).choices[0].message.content
        
        if not raw_text:
            raise ValueError("LLM returned an empty response during generation.")
            
    except Exception as e:
        raise RuntimeError(f"Insight generation LLM call failed: {e}")

    # --- S79.7 / XCUT-2: semantic safety classification (replaces denylist) ---
    safety_classification = evaluate_clinical_safety(raw_text, llm_client)

    if safety_classification.verdict == ClinicalSafetyVerdict.REJECT:
        raise ValueError(
            "Generated insight rejected by clinical safety classifier — refusing to ship. "
            f"Concerns: {list(safety_classification.detected_concerns)}. "
            f"Reasoning: {safety_classification.reasoning}"
        )

    routed_to_human_review = (
        safety_classification.verdict == ClinicalSafetyVerdict.HUMAN_REVIEW_REQUIRED
        or claim.level == ClaimLevel.LEVEL_3
    )
    review_reason = None
    if safety_classification.verdict == ClinicalSafetyVerdict.HUMAN_REVIEW_REQUIRED:
        review_reason = f"Clinical safety classifier flagged for review: {safety_classification.reasoning}"
    if claim.level == ClaimLevel.LEVEL_3:
        l3_reason = "Standing 6-month mandatory human-review requirement for all Level 3 text."
        review_reason = f"{review_reason} | {l3_reason}" if review_reason else l3_reason

    sentences = _split_sentences(raw_text)
    if len(sentences) > 3:
        raise ValueError(f"Generated insight exceeds the 3-sentence maximum ({len(sentences)} sentences).")

    # --- S79.6: semantic entailment grounding (replaces token-overlap heuristic) ---
    citation_chain: List[CitationChainEntry] = []
    for i, sentence in enumerate(sentences):
        entailment = verify_claim_entailment(sentence, excerpts, llm_client)
        if entailment.verdict != EntailmentVerdict.ENTAILED or entailment.source_session_id is None:
            raise ValueError(
                f"Sentence {i} failed entailment verification (verdict={entailment.verdict.value}) — "
                f"refusing to ship it. Reasoning: {entailment.reasoning}"
            )
        citation_chain.append(
            CitationChainEntry(
                sentence_index=i,
                sentence_text=sentence,
                source_session_id=entailment.source_session_id,
                entailment_reasoning=entailment.reasoning,
            )
        )

    return GeneratedInsight(
        insight_id=str(uuid.uuid4()),
        claim_id=claim.claim_id,
        text=raw_text.strip(),
        citation_chain=citation_chain,
        routed_to_human_review=routed_to_human_review,
        human_review_reason=review_reason,
        safety_classification=safety_classification,
    )