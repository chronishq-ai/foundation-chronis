"""
mirror/__init__.py
Sprint 12 — The Mirror: Insight Generation Engine

Public API
----------
Core types:
    InsightRecord          — immutable archive entry (text + citations + metadata)
    InsightDraft           — intermediate pre-archive result from the generator
    ToneMode               — DIRECT | REFLECTIVE | WARM (user-selectable)
    FeedbackRating         — HELPFUL | NOT_YET | TOO_SOON
    FeedbackRecord         — immutable feedback annotation (G2-compliant)
    LintResult / LintStatus / LintViolation — specificity linter results
    TTSStubResult          — synthesised voice stub result

Generator:
    MirrorInsightGenerator — core day-34 generator class
    extract_user_vocabulary — extracts user's recurring vocabulary from excerpts

Linter:
    lint_insight           — Day 35 automated quality gate
    WORD_COUNT_MIN / MAX   — 100–200 word bounds

Tone:
    tone_system_prompt     — builds tone-aware system prompt
    synthesize_voice_stub  — Day 35 TTS stub (synthesised, not a user recording)

Feedback:
    AdaptiveFeedbackStore  — per-user adaptive threshold store (Day 36)

Archive:
    InsightArchive         — append-only archive with full-text + tag search

Pipeline:
    run_mirror_pipeline    — full orchestration entry point
"""

from mirror.insight_generator import (
    MirrorInsightGenerator,
    InsightDraft,
    extract_user_vocabulary,
    MIRROR_SYSTEM_PROMPT,
)
from mirror.specificity_linter import (
    lint_insight,
    LintResult,
    LintStatus,
    LintViolation,
    WORD_COUNT_MIN,
    WORD_COUNT_MAX,
)
from mirror.tone_calibration import (
    ToneMode,
    tone_system_prompt,
    synthesize_voice_stub,
    TTSStubResult,
)
from mirror.feedback_loop import (
    FeedbackRating,
    FeedbackRecord,
    AdaptiveFeedbackStore,
    DEFAULT_ADMISSIBILITY_THRESHOLD,
    MAX_ADMISSIBILITY_THRESHOLD,
)
from mirror.archive import (
    InsightRecord,
    InsightArchive,
)
from mirror.mirror_pipeline import run_mirror_pipeline

__all__ = [
    # Generator
    "MirrorInsightGenerator",
    "InsightDraft",
    "extract_user_vocabulary",
    "MIRROR_SYSTEM_PROMPT",
    # Linter
    "lint_insight",
    "LintResult",
    "LintStatus",
    "LintViolation",
    "WORD_COUNT_MIN",
    "WORD_COUNT_MAX",
    # Tone
    "ToneMode",
    "tone_system_prompt",
    "synthesize_voice_stub",
    "TTSStubResult",
    # Feedback
    "FeedbackRating",
    "FeedbackRecord",
    "AdaptiveFeedbackStore",
    "DEFAULT_ADMISSIBILITY_THRESHOLD",
    "MAX_ADMISSIBILITY_THRESHOLD",
    # Archive
    "InsightRecord",
    "InsightArchive",
    # Pipeline
    "run_mirror_pipeline",
]
