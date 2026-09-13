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
from typing import List, Sequence

# S12.2: single canonical sentence splitter — no duplicate tokenizer.
# grounded_generation._split_sentences is the authoritative implementation
# used by the citation chain builder and the linter alike.
from claims_engine.grounded_generation import _split_sentences


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
    # Extended (S12.1): adds appear/look verbs and more emotion/state words.
    # Bypasses like "You appear overwhelmed" or "You look drained" are now caught.
    (
        "generic_emotion_exclamation",
        re.compile(
            r"\byou (were|are|feel|felt|seem(ed)?|sound(ed)?|appear(ed)?|look(ed)?)\s+"
            r"(stressed|anxious|sad|happy|tired|overwhelmed|burnt out|burned out|"
            r"great|amazing|terrible|drained|depleted|exhausted|struggling|"
            r"low|unmotivated|disengaged|burnt[- ]out|burned[- ]out|burnt|burned)"
            r"[\s!,.]",
            re.IGNORECASE,
        ),
    ),
    # Indirect attribution: "it sounds/seems/appears/looks like you're [state]"
    # Catches bypasses like "It sounds like you're going through a tough time."
    # or "Seems like you might be struggling lately." These are generic emotional
    # attributions that don't reference specific behavioural data.
    (
        "indirect_attribution",
        re.compile(
            r"\b(it\s+)?(sounds?|seems?|appears?|looks?)\s+like\s+"
            r"(you('re| are| might be| could be| seem to be| appear to be)|"
            r"things\s+(are|might be|could be))\b",
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
    # Implicit wellness: "might/could/would benefit from rest", "some rest would help"
    # Catches bypasses like "Perhaps some downtime would help" or
    # "You might benefit from a break" that evade the explicit 'try X' pattern.
    # Also catches standalone "might help" / "would help" without a trailing object,
    # and "use a break" / "need some space" phrasing variants.
    (
        "implicit_wellness_recommendation",
        re.compile(
            r"\b("
            # Pattern A: verb + benefit/use/need/want + (some|a)? + activity
            r"(might|could|would|may|should)\s+(benefit\s+from|(use|need|want)\s+(some\s+|a\s+)?)"
            r"(rest|sleep|break|support|time\s+off|self[- ]care|space|downtime|distance|recharge)"
            r"|"
            # Pattern B: (some|a) + activity + (might|could|would) help
            r"(some\s+|a\s+(bit\s+of\s+|little\s+)?)?"
            r"(rest|sleep|break|space|downtime|distance|recharge)"
            r"\s+(might|could|would|may)\s+help"
            r"|"
            # Pattern C: standalone (might|could|would) help with no object required
            r"(might|could|would|may)\s+help\s+(you\s+)?(recharge|recover|rest|reset)"
            r"|"
            # Pattern D: benefit from + activity
            r"(might|could|would|may|should)\s+benefit\s+from\s+(some\s+|a\s+)?"
            r"(rest|sleep|break|support|time\s+off|self[- ]care|space|downtime|distance|recharge)"
            r")",
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
            r"\b(stressed|anxious|sad|tired|overwhelmed|burnt out|drained|depleted|exhausted)\b",
            re.IGNORECASE,
        ),
    ),
    # Vague difficulty reference without a session data anchor
    # Catches "a tough time", "a hard week", "a difficult period" etc.
    # These describe emotional states without citing specific behavioural data.
    (
        "vague_difficulty_reference",
        re.compile(
            r"\ba\s+(tough|hard|difficult|rough|challenging|draining|exhausting)\s+"
            r"(time|day|days|week|weeks|month|months|period|stretch|patch|moment|spell)\b",
            re.IGNORECASE,
        ),
    ),
    # "Your [noun] is [evaluation]" without data
    (
        "unanchored_evaluation",
        re.compile(
            r"\byour (mood|energy|focus|wellbeing|mental health|stress level|balance|productivity)\s+"
            r"(is|was|has been|seems?|appears?)\s+"
            r"(low|high|good|bad|poor|excellent|improving|declining|off|depleted|reduced)\b",
            re.IGNORECASE,
        ),
    ),
    # Unsolicited self-care directives
    # Catches "taking care of yourself is important", "don't forget to rest",
    # "make sure you get some sleep", etc.
    (
        "unsolicited_care_directive",
        re.compile(
            r"\b(taking|take)\s+care\s+of\s+yourself\b|"
            r"\bdon'?t\s+forget\s+to\s+(rest|take\s+breaks?|recharge|relax|sleep)\b|"
            r"\bmake\s+sure\s+(you\s+)?(rest|sleep|get\s+some\s+sleep|take\s+breaks?|recharge)\b",
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

# S12.2: _split_sentences is now imported from claims_engine.grounded_generation
# (see import block above). The local duplicate has been removed.


def _count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def lint_insight(
    text: str,
    citation_chain: Sequence,
) -> LintResult:
    """
    Run the specificity linter on a generated insight text.

    Three independent checks (all must pass):
      1. BANNED_PATTERN  — each sentence is scanned against the banned-pattern
                           catalogue. Any match is a hard failure.
      2. MISSING_ANCHOR  — every sentence index must appear in citation_chain.
                           A sentence with no citation is a hard failure.
      3. WORD_COUNT      — text must be WORD_COUNT_MIN–WORD_COUNT_MAX words.

    S12.1 — citation_chain is REQUIRED (not Optional):
    Accepting None would silently bypass the anchor check, allowing a
    completely uncited insight to pass the linter. Every caller must
    supply a citation chain. Use an empty list [] only if you intend to
    reject all sentences for missing anchors (e.g. in rejection tests).

    Args:
        text:           The generated insight text (full string).
        citation_chain: Sequence of objects with a ``sentence_index`` attribute.
                        Must be provided — passing None is a TypeError.

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

    # --- Check 2: citation anchors (REQUIRED — not optional) ----------------
    # S12.1: citation_chain is always required. This check is never skipped.
    cited_indices = {entry.sentence_index for entry in citation_chain}
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
