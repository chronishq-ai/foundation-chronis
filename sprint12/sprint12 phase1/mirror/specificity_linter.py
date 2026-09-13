"""
mirror/specificity_linter.py
Sprint 12, Day 35 — Automated quality gate.

Rejects generic, non-evidenced phrasing BEFORE an insight can ship.
The explicitly banned pattern class per the spec:
    "You were stressed today! Try meditation."
    — Any sentence that names an emotion/state without a specific behavioural
      data reference, then offers a generic wellness recommendation.

Two failure modes are detected independently:
  1. BANNED_PATTERN   — sentence matches a generic-coaching pattern
  2. MISSING_ANCHOR   — sentence has no citation anchor in the citation chain

Both must pass for a PASS result. Either failure blocks the insight entirely.

Bible ref: Part 5.21 (Insight Generation / The Mirror, Module 4.10)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# Banned pattern catalogue
# The spec bans "generic, non-evidenced phrasing" — operationalised as:
#   • Generic emotion naming WITHOUT a behavioural anchor ("You were stressed")
#   • Generic wellness recommendations ("Try meditation", "Get more sleep")
#   • Universal positive-psychology filler ("Be kind to yourself", "You've got this")
#   • Vague temporal attribution without data ("today", "recently", "sometimes")
#     combined with a state claim, with no data reference
# ---------------------------------------------------------------------------

# Each entry: (pattern_name, compiled regex)
_BANNED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Generic emotion label + exclamation / generic suffix
    (
        "generic_emotion_exclamation",
        re.compile(
            r"\byou (were|are|feel|felt|seem(ed)?|sound(ed)?)\s+"
            r"(stressed|anxious|sad|happy|tired|overwhelmed|burnt out|great|amazing|terrible)"
            r"[\s!,.]",
            re.IGNORECASE,
        ),
    ),
    # Wellness recommendation without data reference
    (
        "generic_wellness_recommendation",
        re.compile(
            r"\b(try|consider|maybe try|perhaps try|you (could|should|might) try)\s+"
            r"(meditation|mindfulness|journaling|exercise|sleep|self[- ]care|breathing|a walk|yoga)",
            re.IGNORECASE,
        ),
    ),
    # Universal filler affirmations
    (
        "filler_affirmation",
        re.compile(
            r"\b(be kind to yourself|you('ve| have) got this|"
            r"that('s| is) (okay|ok|normal|totally normal)|"
            r"it('s| is) (okay|ok) to feel|"
            r"everyone (feels|goes through|experiences))\b",
            re.IGNORECASE,
        ),
    ),
    # Generic temporal + state claim with no specific data language
    (
        "vague_temporal_state",
        re.compile(
            r"\b(today|recently|lately|sometimes|often|usually)\b.{0,40}"
            r"\b(you (were|are|feel|felt|seem(ed)?|appear(ed)?))\b.{0,40}"
            r"\b(stressed|anxious|sad|tired|overwhelmed|burnt out)\b",
            re.IGNORECASE,
        ),
    ),
    # "Your [noun] is [evaluation]" without data
    (
        "unanchored_evaluation",
        re.compile(
            r"\byour (mood|energy|focus|wellbeing|mental health|stress level|balance)\s+"
            r"(is|was|has been|seems?|appears?)\s+"
            r"(low|high|good|bad|poor|excellent|improving|declining|off)\b",
            re.IGNORECASE,
        ),
    ),
]

# Word-count window the insight must fall within
WORD_COUNT_MIN = 100
WORD_COUNT_MAX = 200


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class LintStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class LintViolation:
    sentence_index: int
    sentence_text: str
    violation_type: str   # "BANNED_PATTERN" | "MISSING_ANCHOR" | "WORD_COUNT"
    detail: str


@dataclass(frozen=True)
class LintResult:
    """
    Result of running the specificity linter on a generated insight.

    status == FAIL means the insight is BLOCKED and must not ship.
    violations lists every sentence-level failure found.
    """
    status: LintStatus
    violations: List[LintViolation] = field(default_factory=list)
    word_count: int = 0

    @property
    def passed(self) -> bool:
        return self.status == LintStatus.PASS


# ---------------------------------------------------------------------------
# Core linter
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def lint_insight(
    text: str,
    citation_chain: Optional[Sequence] = None,
) -> LintResult:
    """
    Run the specificity linter on a generated insight text.

    Two independent checks:
      1. BANNED_PATTERN  — each sentence is scanned against the banned-pattern
                           catalogue. Any match is a hard failure.
      2. MISSING_ANCHOR  — if a citation_chain is provided, every sentence index
                           must appear in it. A sentence with no citation is a
                           hard failure.
      3. WORD_COUNT      — text must be WORD_COUNT_MIN–WORD_COUNT_MAX words.

    Args:
        text:           The generated insight text (full string).
        citation_chain: Optional sequence of CitationChainEntry objects (or any
                        objects with a sentence_index attribute). When provided,
                        sentences without a matching citation entry are rejected.

    Returns:
        LintResult — status PASS or FAIL, with all violations listed.
    """
    sentences = _split_sentences(text)
    word_count = _count_words(text)
    violations: List[LintViolation] = []

    # --- Check 1: banned patterns -------------------------------------------
    for i, sentence in enumerate(sentences):
        for pattern_name, pattern in _BANNED_PATTERNS:
            if pattern.search(sentence):
                violations.append(LintViolation(
                    sentence_index=i,
                    sentence_text=sentence,
                    violation_type="BANNED_PATTERN",
                    detail=(
                        f"Sentence matches banned pattern '{pattern_name}'. "
                        f"Every sentence must reference a specific behavioural data point. "
                        f"Generic emotion labels and wellness recommendations are not permitted."
                    ),
                ))
                break  # one violation per sentence is enough

    # --- Check 2: citation anchors ------------------------------------------
    if citation_chain is not None:
        cited_indices = set()
        for entry in citation_chain:
            cited_indices.add(entry.sentence_index)
        for i, sentence in enumerate(sentences):
            if i not in cited_indices:
                violations.append(LintViolation(
                    sentence_index=i,
                    sentence_text=sentence,
                    violation_type="MISSING_ANCHOR",
                    detail=(
                        f"Sentence {i} has no citation anchor in the citation chain. "
                        f"Every sentence must resolve to a specific SessionExcerpt."
                    ),
                ))

    # --- Check 3: word count ------------------------------------------------
    if not (WORD_COUNT_MIN <= word_count <= WORD_COUNT_MAX):
        violations.append(LintViolation(
            sentence_index=-1,
            sentence_text="",
            violation_type="WORD_COUNT",
            detail=(
                f"Insight is {word_count} words; must be "
                f"{WORD_COUNT_MIN}–{WORD_COUNT_MAX} words."
            ),
        ))

    status = LintStatus.PASS if not violations else LintStatus.FAIL
    return LintResult(status=status, violations=violations, word_count=word_count)
