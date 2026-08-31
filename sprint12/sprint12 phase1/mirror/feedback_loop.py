"""
mirror/feedback_loop.py
Sprint 12, Day 36 — Adaptive feedback loop.

Wires the "helpful / not yet / too soon" user feedback ratings to
ADAPTIVELY raise that specific user's admissibility thresholds on
repeated "not yet" ratings.

Key constraint from the spec:
    "adaptive loop, not a static one"

This means:
  - Every user starts at the global DEFAULT_ADMISSIBILITY_THRESHOLD.
  - Each NOT_YET rating nudges that user's personal threshold upward.
  - Threshold is capped at MAX_ADMISSIBILITY_THRESHOLD to prevent
    the system from permanently locking a user out.
  - HELPFUL ratings can slowly lower the threshold back toward the
    default (evidence that the bar is well-calibrated for this user).
  - TOO_SOON is treated as the strongest negative signal: it raises
    the threshold more sharply than NOT_YET, and also logs the domain
    so the Claims Engine can suppress that domain's claims for longer.

No feedback flag or its underlying data is ever surfaced as product
copy — this is internal signal only.

Bible ref: Part 5.21 (The Mirror, Module 4.10)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------

# The Claims Engine uses domain.confidence >= DOMAIN_CONFIDENCE_FLOOR (0.5)
# as its Level 2 gate. The Mirror's adaptive threshold is an ADDITIONAL,
# per-user multiplier on top of that floor — it raises the effective bar.
DEFAULT_ADMISSIBILITY_THRESHOLD: float = 0.50   # matches DOMAIN_CONFIDENCE_FLOOR in Sprint 9
MAX_ADMISSIBILITY_THRESHOLD: float     = 0.90   # hard ceiling — never locks user out permanently
MIN_ADMISSIBILITY_THRESHOLD: float     = 0.40   # floor — never go below Sprint 9's implicit bar

# How much each rating shifts the per-user threshold
_NOT_YET_DELTA:  float = +0.05   # one NOT_YET   nudges threshold up 5 pp
_TOO_SOON_DELTA: float = +0.10   # one TOO_SOON  nudges threshold up 10 pp (stronger signal)
_HELPFUL_DELTA:  float = -0.02   # one HELPFUL   nudges threshold down 2 pp (slow recovery)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class FeedbackRating(Enum):
    """
    The three user-facing feedback choices for a Mirror insight.

    HELPFUL  — the insight was accurate and useful
    NOT_YET  — the insight was plausible but the user isn't ready for it
    TOO_SOON — the insight surfaced too early; not enough data to be meaningful
    """
    HELPFUL  = "helpful"
    NOT_YET  = "not_yet"
    TOO_SOON = "too_soon"


@dataclass(frozen=True)
class FeedbackRecord:
    """
    Immutable record of a single user feedback event.

    Never surfaced as product copy. Internal signal only.
    Append-only: corrections create new records, never overwrites (G2).
    """
    feedback_id: str
    user_id: str
    insight_id: str
    domain_id: Optional[str]      # which domain the insight concerned
    rating: FeedbackRating
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class _UserThresholdState:
    """
    Mutable per-user threshold state (internal, never exposed).
    """
    threshold: float = DEFAULT_ADMISSIBILITY_THRESHOLD
    total_not_yet: int = 0
    total_too_soon: int = 0
    total_helpful: int = 0
    suppressed_domains: Dict[str, datetime] = field(default_factory=dict)
        # domain_id -> suppressed_until timestamp (for TOO_SOON)


# ---------------------------------------------------------------------------
# Adaptive feedback store
# ---------------------------------------------------------------------------

class AdaptiveFeedbackStore:
    """
    Per-user, in-memory adaptive feedback store.

    Thread-safety: not thread-safe. In production, back this with a
    per-user database row (the threshold and domain-suppression map
    are the only persisted fields).

    Usage
    -----
    >>> store = AdaptiveFeedbackStore()
    >>> store.record_feedback("u_001", "ins_42", FeedbackRating.NOT_YET, domain_id="work")
    >>> store.get_admissibility_threshold("u_001")   # > 0.50
    0.55
    >>> store.is_domain_suppressed("u_001", "work")  # False (NOT_YET doesn't suppress domains)
    False
    >>> store.record_feedback("u_001", "ins_43", FeedbackRating.TOO_SOON, domain_id="work")
    >>> store.is_domain_suppressed("u_001", "work")  # True (TOO_SOON suppresses for 30 days)
    True
    """

    # How long a TOO_SOON rating suppresses that domain's claims (days)
    TOO_SOON_SUPPRESSION_DAYS: int = 30

    def __init__(self) -> None:
        self._states: Dict[str, _UserThresholdState] = {}
        self._records: List[FeedbackRecord] = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_feedback(
        self,
        user_id: str,
        insight_id: str,
        rating: FeedbackRating,
        domain_id: Optional[str] = None,
    ) -> FeedbackRecord:
        """
        Record one feedback event and update this user's threshold.

        Args:
            user_id:    The user who gave the feedback.
            insight_id: Which insight was rated.
            rating:     FeedbackRating.HELPFUL | NOT_YET | TOO_SOON
            domain_id:  Which domain the insight concerned (optional but
                        required for TOO_SOON domain suppression).

        Returns:
            The immutable FeedbackRecord created.
        """
        import uuid
        state = self._get_or_create_state(user_id)

        # Update threshold
        if rating == FeedbackRating.NOT_YET:
            state.total_not_yet += 1
            state.threshold = min(
                MAX_ADMISSIBILITY_THRESHOLD,
                state.threshold + _NOT_YET_DELTA,
            )
        elif rating == FeedbackRating.TOO_SOON:
            state.total_too_soon += 1
            state.threshold = min(
                MAX_ADMISSIBILITY_THRESHOLD,
                state.threshold + _TOO_SOON_DELTA,
            )
            # Suppress the specific domain for TOO_SOON_SUPPRESSION_DAYS
            if domain_id:
                suppressed_until = datetime.now(timezone.utc) + timedelta(
                    days=self.TOO_SOON_SUPPRESSION_DAYS
                )
                state.suppressed_domains[domain_id] = suppressed_until
                logger.info(
                    "user=%s: domain '%s' suppressed for %d days due to TOO_SOON rating.",
                    user_id, domain_id, self.TOO_SOON_SUPPRESSION_DAYS,
                )
        elif rating == FeedbackRating.HELPFUL:
            state.total_helpful += 1
            state.threshold = max(
                MIN_ADMISSIBILITY_THRESHOLD,
                state.threshold + _HELPFUL_DELTA,
            )

        logger.debug(
            "user=%s  rating=%s  new_threshold=%.3f  (not_yet=%d  too_soon=%d  helpful=%d)",
            user_id, rating.value, state.threshold,
            state.total_not_yet, state.total_too_soon, state.total_helpful,
        )

        record = FeedbackRecord(
            feedback_id=str(uuid.uuid4()),
            user_id=user_id,
            insight_id=insight_id,
            domain_id=domain_id,
            rating=rating,
        )
        self._records.append(record)
        return record

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_admissibility_threshold(self, user_id: str) -> float:
        """
        Return this user's current adaptive admissibility threshold.

        Starts at DEFAULT_ADMISSIBILITY_THRESHOLD (0.50).
        Rises on NOT_YET/TOO_SOON ratings. Recovers slowly on HELPFUL.
        Capped at MAX (0.90) and MIN (0.40).

        This value is the effective confidence floor the Claims Engine
        should use for this user instead of the global 0.50 constant.
        """
        return self._get_or_create_state(user_id).threshold

    def is_domain_suppressed(self, user_id: str, domain_id: str) -> bool:
        """
        Return True if this domain is currently suppressed for this user
        due to a TOO_SOON rating within the suppression window.

        Domain suppression is checked at pipeline entry — suppressed
        domains produce no Mirror output regardless of claim admissibility.
        """
        state = self._get_or_create_state(user_id)
        suppressed_until = state.suppressed_domains.get(domain_id)
        if suppressed_until is None:
            return False
        if datetime.now(timezone.utc) < suppressed_until:
            return True
        # Window expired — remove stale entry
        del state.suppressed_domains[domain_id]
        return False

    def get_feedback_history(self, user_id: str) -> List[FeedbackRecord]:
        """Return all feedback records for this user (immutable copies)."""
        return [r for r in self._records if r.user_id == user_id]

    def get_threshold_summary(self, user_id: str) -> dict:
        """Debug helper — returns the full threshold state for a user."""
        state = self._get_or_create_state(user_id)
        return {
            "user_id": user_id,
            "threshold": state.threshold,
            "total_not_yet": state.total_not_yet,
            "total_too_soon": state.total_too_soon,
            "total_helpful": state.total_helpful,
            "suppressed_domains": {
                k: v.isoformat() for k, v in state.suppressed_domains.items()
            },
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_create_state(self, user_id: str) -> _UserThresholdState:
        if user_id not in self._states:
            self._states[user_id] = _UserThresholdState()
        return self._states[user_id]
