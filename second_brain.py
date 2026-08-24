"""
second_brain.py

Sprint 11, Day 33 -- Second Brain / Decision Replication scaffolding.

What this does, in plain words:
Builds a snapshot of a user's claims for the Second Brain / Decision
Replication feature -- something that will eventually help predict "what
would this person likely decide here, based on everything Chronis knows
about them."

THE MOST IMPORTANT THING ABOUT THIS FILE, stated plainly because it's
easy to get backwards: this module is DELIBERATELY UNFILTERED. Unlike
behavioral_dna.py (which strictly filters to admissible Level 3 claims
only), this module includes EVERY claim it's given -- Level 0 through 3,
admissible or not. Deciding what's safe to actually show a user is the
constitutional-policy layer's job, a different team, later in the
pipeline. Adding a safety filter HERE would be scope creep into a layer
that isn't ours -- even though it might feel like the "responsible" thing
to do, it would actually be building the wrong thing in the wrong place.

The ONE thing this module still does: scopes claims to the requested
user_id only. That's not a judgment call about safety or quality -- it's
basic data correctness, same as every other module in this sprint (see
behavioral_dna.py's user_id check for the same reasoning).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

from upstream_interfaces import Claim


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class DecisionReplicationSnapshot:
    """Deliberately has NO gating-related fields (no is_approved,
    blocked_reason, gate_passed, etc.) -- see module docstring. If a
    future edit adds one of those, that's a sign scope has crept into
    the constitutional-policy layer's territory."""
    user_id: str
    all_claims: List[Claim]
    generated_at: datetime = field(default_factory=_utc_now)


def build_decision_replication_snapshot(
    user_id: str,
    claims: List[Claim],
) -> DecisionReplicationSnapshot:
    """Includes every claim belonging to this user, unfiltered by level or
    admissibility. This is intentional -- see module docstring."""
    users_claims: List[Claim] = []

    for claim in claims:
        if claim.user_id == user_id:
            users_claims.append(claim)

    return DecisionReplicationSnapshot(
        user_id=user_id,
        all_claims=users_claims,
    )