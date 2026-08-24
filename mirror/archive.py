"""
mirror/archive.py
Sprint 12, Day 36 — Full insight archive with search indexing.

Append-only store: every Mirror insight ever generated for a user is retained.
Corrections (Sprint 17 "Teach Chronis") are counter-annotations, not overwrites
(G2-compliant).

Search: full-text + tag filtering.
Tags: divergence_type, date (YYYY-MM-DD), tone, feedback_rating.

Bible ref: Part 5.21 (The Mirror, Module 4.10) · Bible Part 2.3 (G2 canonical permanence)
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from mirror.tone_calibration import ToneMode
from mirror.feedback_loop import FeedbackRating


# ---------------------------------------------------------------------------
# InsightRecord — the canonical Mirror output object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InsightRecord:
    """
    The full, immutable output of one Mirror pipeline run.

    Every sentence is traceable to a SessionExcerpt via citation_chain.
    Append-only: once created, never mutated (G2).
    """
    insight_id: str
    user_id: str
    domain_id: Optional[str]
    text: str                             # 100–200 words, second person, grounded
    tone: ToneMode
    citation_chain: Sequence             # List[CitationChainEntry] from Sprint 9
    claim_ids: Sequence[str]             # which Claim objects drove this insight
    dominant_divergence_type: Optional[str]
    routed_to_human_review: bool
    human_review_reason: Optional[str]
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    feedback_rating: Optional[FeedbackRating] = None  # updated via archive.add_feedback()

    # Tags for search (derived at creation time, not queried from text)
    tags: Sequence[str] = field(default_factory=list)

    @staticmethod
    def new(
        user_id: str,
        text: str,
        tone: ToneMode,
        citation_chain: Sequence,
        claim_ids: Sequence[str],
        dominant_divergence_type: Optional[str] = None,
        domain_id: Optional[str] = None,
        routed_to_human_review: bool = False,
        human_review_reason: Optional[str] = None,
        extra_tags: Optional[Sequence[str]] = None,
    ) -> "InsightRecord":
        """
        Factory method: creates an InsightRecord with auto-generated ID,
        timestamp, and derived tags.
        """
        now = datetime.now(timezone.utc)
        tags: List[str] = [now.strftime("date:%Y-%m-%d"), f"tone:{tone.value}"]
        if dominant_divergence_type:
            tags.append(f"divergence_type:{dominant_divergence_type}")
        if domain_id:
            tags.append(f"domain:{domain_id}")
        if extra_tags:
            tags.extend(extra_tags)

        return InsightRecord(
            insight_id=str(uuid.uuid4()),
            user_id=user_id,
            domain_id=domain_id,
            text=text,
            tone=tone,
            citation_chain=citation_chain,
            claim_ids=list(claim_ids),
            dominant_divergence_type=dominant_divergence_type,
            routed_to_human_review=routed_to_human_review,
            human_review_reason=human_review_reason,
            generated_at=now,
            feedback_rating=None,
            tags=tags,
        )


# ---------------------------------------------------------------------------
# InsightArchive
# ---------------------------------------------------------------------------

class InsightArchive:
    """
    Append-only insight archive with full-text and tag search.

    Storage model (in-memory for this sprint):
        _records: Dict[(user_id, insight_id) -> InsightRecord]

    In production, back this with a per-user database table.
    The append-only constraint must be preserved at the storage layer —
    no UPDATE or DELETE on insight rows (G2).

    Search
    ------
    search(user_id, query, tags) performs:
      1. Tag filtering (all supplied tags must be present — AND semantics)
      2. Full-text filtering (query tokens matched against insight text, case-insensitive)
    Both filters are applied if supplied; omit either to skip that filter.
    """

    def __init__(self) -> None:
        # Key: (user_id, insight_id)
        self._records: Dict[tuple, InsightRecord] = {}
        # Feedback annotations: (user_id, insight_id) -> FeedbackRating
        # Stored separately from InsightRecord (which is frozen) per G2.
        self._feedback: Dict[tuple, FeedbackRating] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, record: InsightRecord) -> None:
        """
        Append an InsightRecord to the archive.

        Raises ValueError if an insight with the same insight_id for this
        user already exists (append-only: no overwrites).
        """
        key = (record.user_id, record.insight_id)
        if key in self._records:
            raise ValueError(
                f"InsightRecord ({record.user_id}, {record.insight_id}) already exists. "
                f"The archive is append-only — use add_feedback() to annotate, never overwrite."
            )
        self._records[key] = record

    def add_feedback(
        self,
        user_id: str,
        insight_id: str,
        rating: FeedbackRating,
    ) -> None:
        """
        Attach a feedback rating to an existing insight.

        Per G2 (canonical permanence): the original InsightRecord is immutable.
        Feedback is stored as a separate annotation, not an overwrite.

        Raises KeyError if the (user_id, insight_id) pair doesn't exist.
        """
        key = (user_id, insight_id)
        if key not in self._records:
            raise KeyError(
                f"No InsightRecord found for user={user_id} insight_id={insight_id}. "
                f"Cannot add feedback to a non-existent record."
            )
        self._feedback[key] = rating

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, user_id: str, insight_id: str) -> InsightRecord:
        """
        Retrieve a specific insight by user + insight ID.

        Raises KeyError if not found.
        """
        key = (user_id, insight_id)
        if key not in self._records:
            raise KeyError(f"No insight found: user={user_id}  insight_id={insight_id}")
        return self._records[key]

    def get_feedback(self, user_id: str, insight_id: str) -> Optional[FeedbackRating]:
        """Return the feedback rating for an insight, or None if not yet rated."""
        return self._feedback.get((user_id, insight_id))

    def list_all(self, user_id: str) -> List[InsightRecord]:
        """Return all insights for a user, ordered by generated_at (ascending)."""
        records = [v for (uid, _), v in self._records.items() if uid == user_id]
        return sorted(records, key=lambda r: r.generated_at)

    def search(
        self,
        user_id: str,
        query: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> List[InsightRecord]:
        """
        Full-text + tag search over a user's insight archive.

        Args:
            user_id: Which user's archive to search.
            query:   Free-text query string. Tokenised and matched
                     case-insensitively against insight text (AND semantics
                     across tokens — all tokens must appear).
            tags:    Tag strings to filter by (AND semantics — all must match).
                     Tag format: "key:value" (e.g. "tone:warm",
                     "divergence_type:aspiration", "date:2026-08-24").

        Returns:
            List of matching InsightRecords, ordered by generated_at descending
            (most recent first).
        """
        candidates = self.list_all(user_id)

        # --- Tag filter (AND across all supplied tags) ---
        if tags:
            tag_set = set(tags)
            candidates = [
                r for r in candidates
                if tag_set.issubset(set(r.tags))
            ]

        # --- Full-text filter (AND across all query tokens) ---
        if query and query.strip():
            tokens = re.findall(r"\w+", query.lower())
            def _matches(record: InsightRecord) -> bool:
                text_lower = record.text.lower()
                return all(tok in text_lower for tok in tokens)
            candidates = [r for r in candidates if _matches(r)]

        # Return most-recent first
        return sorted(candidates, key=lambda r: r.generated_at, reverse=True)

    def count(self, user_id: str) -> int:
        """Return total number of insights archived for this user."""
        return sum(1 for (uid, _) in self._records if uid == user_id)
