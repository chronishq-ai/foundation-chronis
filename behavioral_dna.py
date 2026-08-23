"""
behavioral_dna.py

Sprint 11, Day 32 -- Behavioral DNA export.

What this does, in plain words:
Packages a user's most trustworthy claims about themselves (Level 3 --
the rarest, highest-confidence identity claims) together with two other
personal profiles (lexicon, social graph) into one portable object.

Filtering rule, and why it's stricter than it might look at first:
A claim only belongs in this export if ALL THREE of these are true:
  1. claim.level == ClaimLevel.LEVEL_3
     (not Level 0/1/2 -- those aren't identity-level claims)
  2. claim.gate_evaluation.admissible == True
     (a Claim object can exist in the system and still have FAILED its
     gate -- level alone does not mean it's safe to surface)
  3. claim.user_id == the user this export is being built for
     (defends against a caller accidentally passing in a mixed batch of
     claims from multiple users -- this module never trusts the caller
     to have already filtered correctly)

Signing: cryptographic signing infrastructure does not exist yet in
Sprint 11's scope (it's a later HARDENERS/Sprint 13 concern). This module
NEVER fakes a signature -- is_signed stays False and signature stays None
until real signing is wired in. Same "silence over false confidence"
principle used everywhere else in this sprint.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from upstream_interfaces import Claim, ClaimLevel


def _utc_now() -> datetime:
    """Timezone-aware UTC timestamp. datetime.utcnow() is deprecated as of
    Python 3.12 (it returns a naive datetime, which is exactly the kind of
    silent ambiguity the Bible's own logging discipline wouldn't accept
    elsewhere) -- this is the modern, non-deprecated replacement."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class BehavioralDNAExport:
    user_id: str
    level3_claims: List[Claim]
    lexicon_profile: Optional[Dict[str, Any]] = None
    social_graph_summary: Optional[Dict[str, Any]] = None
    signature: Optional[str] = None
    is_signed: bool = False
    generated_at: datetime = field(default_factory=_utc_now)


def build_behavioral_dna_export(
    user_id: str,
    claims: List[Claim],
    lexicon_profile: Optional[Dict[str, Any]] = None,
    social_graph_summary: Optional[Dict[str, Any]] = None,
) -> BehavioralDNAExport:
    """Builds the export object. Does NOT sign it -- see module docstring."""
    qualifying_claims: List[Claim] = []

    for claim in claims:
        is_level3 = claim.level == ClaimLevel.LEVEL_3
        is_admissible = claim.gate_evaluation.admissible
        belongs_to_this_user = claim.user_id == user_id

        if is_level3 and is_admissible and belongs_to_this_user:
            qualifying_claims.append(claim)

    return BehavioralDNAExport(
        user_id=user_id,
        level3_claims=qualifying_claims,
        lexicon_profile=lexicon_profile,
        social_graph_summary=social_graph_summary,
        signature=None,
        is_signed=False,
    )
