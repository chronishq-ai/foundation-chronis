"""
inheritance_protocol.py

Sprint 11, Day 33 -- Inheritance Protocol.

What this does, in plain words:
Takes a user's Behavioral DNA export (built in behavioral_dna.py) and
turns their most recent admissible Level 3 claim into a short, AI-written
"Behavioral Letter" -- reusing Sprint 9's existing constrained-RAG
generation pipeline (Mansi's generate_insight), not new generation logic.

WHY insight_generator IS A PARAMETER, NOT A DIRECT IMPORT: Mansi's real
claims_engine package (which contains generate_insight) is not yet
importable on this repo's PYTHONPATH -- same situation as
clinical_terms.py. Rather than fake her generation logic, this module
takes the generator as an injected dependency with the exact signature
her real function has:

    insight_generator(claim, divergence_state, candidate_excerpts, llm_client) -> object with .text and .citation_chain

At integration time, the real caller passes in
claims_engine.grounded_generation.generate_insight directly -- nothing in
THIS file needs to change.

WHY divergence_state IS TYPED AS Any / left opaque: generate_insight
requires a DivergenceState object (Sprint 8, Mansi), whose exact internal
shape was not part of what Sprint 11 was given. Rather than guess that
shape and risk getting it wrong, this module treats it as an opaque
pass-through -- it forwards whatever it's given straight to the real
generator, which is the only thing that actually needs to understand it.

Claim selection rule: picks the user's MOST RECENTLY CREATED admissible
Level 3 claim (by created_at) to write the letter about. This is a
deliberate, simple, documented choice -- not the only reasonable one --
flagged here so a reviewer can challenge it if a different selection rule
turns out to be wanted later.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

from upstream_interfaces import Claim, SessionExcerpt
from behavioral_dna import BehavioralDNAExport


class NoEligibleClaimError(Exception):
    """Raised when a user's Behavioral DNA export has zero admissible
    Level 3 claims -- there is nothing to write a Behavioral Letter about
    yet. This is expected and normal for a cold-start user (Mayank's
    Sprint 10 gating means most users won't have Level 3 claims for
    months) -- callers should catch this and handle it gracefully, not
    treat it as a crash."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class BehavioralLetter:
    user_id: str
    source_claim_id: str
    letter_text: str
    citation_chain: List[str]
    signature: Optional[str] = None
    is_signed: bool = False
    generated_at: datetime = field(default_factory=_utc_now)


# The exact call shape Mansi's real generate_insight() has. Documented
# here as a type alias so callers know what to pass in without needing
# her real package installed.
InsightGeneratorFn = Callable[[Claim, Any, List[SessionExcerpt], Any], Any]


def _pick_most_recent_claim(claims: List[Claim]) -> Claim:
    most_recent = claims[0]
    for claim in claims:
        if claim.created_at > most_recent.created_at:
            most_recent = claim
    return most_recent


def build_inheritance_letter(
    export: BehavioralDNAExport,
    divergence_state: Any,
    candidate_excerpts: List[SessionExcerpt],
    insight_generator: InsightGeneratorFn,
    llm_client: Any,
) -> BehavioralLetter:
    """Builds one Behavioral Letter from a user's Behavioral DNA export.
    Does NOT sign it -- see module docstring, same "no fake signature"
    rule as behavioral_dna.py."""
    if len(export.level3_claims) == 0:
        raise NoEligibleClaimError(
            f"user {export.user_id} has no admissible Level 3 claims -- "
            f"cannot build a Behavioral Letter yet."
        )

    chosen_claim = _pick_most_recent_claim(export.level3_claims)

    generated = insight_generator(
        chosen_claim,
        divergence_state,
        candidate_excerpts,
        llm_client,
    )

    return BehavioralLetter(
        user_id=export.user_id,
        source_claim_id=chosen_claim.claim_id,
        letter_text=generated.text,
        citation_chain=list(generated.citation_chain),
        signature=None,
        is_signed=False,
    )