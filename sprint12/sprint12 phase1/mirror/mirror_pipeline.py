"""
mirror/mirror_pipeline.py
Sprint 12 — Full orchestration entry point.

Pipeline flow (stage-first — cold-start gate checked FIRST):

    ColdStartState.can_surface_claims
           ↓ False → return None (zero output, Stage 0/1)
    Domain suppression check (FeedbackLoop.is_domain_suppressed)
           ↓ suppressed → return None
    Level 1–3 Claims (caller-supplied, from claims_engine)
           ↓
    MirrorInsightGenerator.generate()
           → select_excerpts (Sprint 9 contract)
           → user vocabulary extraction
           → tone-aware LLM call (self-hosted only)
           → clinical filter
           → citation chain
           → specificity linter (hard gate)
           ↓ linter FAIL → ValueError (caller must handle)
    InsightRecord.new()
           ↓
    InsightArchive.append()
           ↓
    return InsightRecord

Callers must check cold_start_state.can_surface_claims before calling this
pipeline. The pipeline enforces the Stage 0/1 silence rule internally as a
hard failsafe — even if the caller forgets to check.

Bible ref: Part 5.21 (The Mirror, Module 4.10)
Sprint 10 ref: ColdStartStage.STAGE_0 / STAGE_1 — zero output.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from cold_start.cold_start import ColdStartStage, ColdStartState
from claims_engine.claim_levels import Claim
from claims_engine.grounded_generation import LLMClient
from divergence_engine.state import DivergenceState
from upstream_interfaces import SessionExcerpt

from mirror.insight_generator import MirrorInsightGenerator, ClinicalTerminologyError
from mirror.tone_calibration import ToneMode
from mirror.feedback_loop import AdaptiveFeedbackStore, FeedbackRating
from mirror.archive import InsightArchive, InsightRecord

logger = logging.getLogger(__name__)

# Stages where The Mirror must be completely silent (no output, no inference)
_SILENT_STAGES = {ColdStartStage.STAGE_0, ColdStartStage.STAGE_1}


def run_mirror_pipeline(
    *,
    user_id: str,
    cold_start_state: ColdStartState,
    claims: Sequence[Claim],
    candidate_excerpts: Sequence[SessionExcerpt],
    divergence_state: DivergenceState,
    llm_client: LLMClient,
    archive: InsightArchive,
    feedback_store: Optional[AdaptiveFeedbackStore] = None,
    tone: ToneMode = ToneMode.REFLECTIVE,
    domain_id: Optional[str] = None,
) -> Optional[InsightRecord]:
    """
    Full Sprint 12 Mirror pipeline for a single user evaluation.

    Returns None if The Mirror must be silent (Stage 0/1, or domain
    suppressed, or no admitted claims). Returns an InsightRecord on
    success, already appended to the archive.

    Args:
        user_id:             User identifier.
        cold_start_state:    ColdStartState from Sprint 10 — checked first.
                             Mirror is silent if stage is STAGE_0 or STAGE_1.
        claims:              Level 1–3 Claim objects from Sprint 9 Claims Engine.
        candidate_excerpts:  SessionExcerpts for excerpt selection + vocabulary.
                             Must include at least one near-miss excerpt for
                             Level 2/3-driven generation.
        divergence_state:    Sprint 8 DivergenceState for this user.
        llm_client:          Self-hosted LLMClient (Protocol). Never a
                             third-party API call.
        archive:             InsightArchive — the generated record is appended.
        feedback_store:      AdaptiveFeedbackStore — consulted for domain
                             suppression. Pass None to skip suppression check.
        tone:                ToneMode (default REFLECTIVE). User-selectable.
        domain_id:           Which domain this pipeline run concerns. Used for
                             archive tagging and domain suppression lookup.

    Returns:
        InsightRecord on success; None if pipeline is silent.

    Raises:
        ValueError: if the specificity linter rejects the generated insight,
                    or if citation chain construction fails.
    """
    # -----------------------------------------------------------------------
    # GATE 1: Cold Start stage check — enforced FIRST, no exceptions.
    # The Mirror is completely silent during Stage 0 and Stage 1.
    # Per Sprint 12 DoD: "The Mirror produces zero output for any Stage 0/1
    # synthetic cold-start user — code-enforced."
    # -----------------------------------------------------------------------
    if cold_start_state.stage in _SILENT_STAGES:
        logger.info(
            "user=%s  day=%d  stage=%s — Mirror is silent (Stage 0/1). "
            "Zero output enforced.",
            user_id, cold_start_state.day, cold_start_state.stage.name,
        )
        return None

    # -----------------------------------------------------------------------
    # GATE 2: can_surface_claims — Sprint 10 evidence gate.
    # Stage 4 still requires evidence gate to have passed.
    # -----------------------------------------------------------------------
    if not cold_start_state.can_surface_claims:
        logger.info(
            "user=%s  day=%d  stage=%s  can_surface_claims=False — "
            "evidence gate not passed. Mirror silent.",
            user_id, cold_start_state.day, cold_start_state.stage.name,
        )
        return None

    # -----------------------------------------------------------------------
    # GATE 3: Domain suppression check (TOO_SOON feedback).
    # -----------------------------------------------------------------------
    if feedback_store is not None and domain_id is not None:
        if feedback_store.is_domain_suppressed(user_id, domain_id):
            logger.info(
                "user=%s  domain='%s' is suppressed due to TOO_SOON rating. "
                "Mirror silent for this domain.",
                user_id, domain_id,
            )
            return None

    # -----------------------------------------------------------------------
    # GATE 4: Adaptive admissibility threshold check.
    # Mirror uses a per-user threshold (raised by NOT_YET / TOO_SOON).
    # -----------------------------------------------------------------------
    if feedback_store is not None:
        user_threshold = feedback_store.get_admissibility_threshold(user_id)
        if divergence_state.confidence < user_threshold:
            logger.info(
                "user=%s  divergence_confidence=%.3f < user_threshold=%.3f — "
                "Mirror silent (adaptive threshold not met).",
                user_id, divergence_state.confidence, user_threshold,
            )
            return None

    # -----------------------------------------------------------------------
    # GENERATE: call the core generator
    # -----------------------------------------------------------------------
    generator = MirrorInsightGenerator(llm_client=llm_client)

    logger.info(
        "user=%s  day=%d  stage=%s  tone=%s  domain=%s — running Mirror generator.",
        user_id, cold_start_state.day, cold_start_state.stage.name,
        tone.value, domain_id,
    )

    try:
        draft = generator.generate(
            user_id=user_id,
            claims=claims,
            candidate_excerpts=candidate_excerpts,
            divergence_state=divergence_state,
            tone=tone,
            domain_id=domain_id,
        )
    except ClinicalTerminologyError as exc:
        # Hard stop: archive as human-review-only, return None (nothing surfaced).
        # Per Sprint 9 standing contract: clinical output must not be shown until
        # a human reviewer clears it. The archive record marks it as pending review.
        logger.warning(
            "user=%s: ClinicalTerminologyError — archiving as human-review-only. "
            "No output surfaced to user. term='%s'",
            user_id, exc.term,
        )
        review_record = InsightRecord.new(
            user_id=user_id,
            text="[PENDING HUMAN REVIEW — clinical terminology detected]",
            tone=tone,
            citation_chain=[],
            claim_ids=[c.claim_id for c in claims if hasattr(c, 'claim_id')],
            domain_id=domain_id,
            routed_to_human_review=True,
            human_review_reason=str(exc),
        )
        archive.append(review_record)
        return None  # nothing surfaced to user

    # -----------------------------------------------------------------------
    # ARCHIVE: build InsightRecord and append (append-only, G2-compliant)
    # -----------------------------------------------------------------------
    record = InsightRecord.new(
        user_id=user_id,
        text=draft.text,
        tone=tone,
        citation_chain=draft.citation_chain,
        claim_ids=draft.claim_ids,
        dominant_divergence_type=draft.dominant_divergence_type,
        domain_id=domain_id,
        routed_to_human_review=draft.routed_to_human_review,
        human_review_reason=draft.human_review_reason,
    )
    archive.append(record)

    logger.info(
        "user=%s  insight_id=%s  words=%d  human_review=%s  archived=True",
        user_id, record.insight_id,
        len(record.text.split()),
        record.routed_to_human_review,
    )
    return record
