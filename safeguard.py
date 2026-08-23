# Day 43 — 30-day post-surfacing check + read-time aspiration exclusion.
#
# Behavioral attractor changes AND narrative regime shifts are flagged
# potentially_claim_influenced when they begin within 30 days of a surfaced
# Level 1–3 claim in the same domain.
#
# Aspiration evidence is excluded at READ time by consulting the index —
# not by trusting a flag written on the event. A caller who forgets to set
# the flag still cannot score the change as independent aspiration proof.
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .index import INFLUENCE_FLAG, SurfacingIndex

KIND_BEHAVIOR = "behavior"
KIND_NARRATIVE = "narrative"
OK_KINDS = frozenset({KIND_BEHAVIOR, KIND_NARRATIVE})


@dataclass
class Change:
    user_id: str
    domain: str
    kind: str  # behavior | narrative
    when: date
    potentially_claim_influenced: bool = False
    influenced: bool = False  # alias

    def __post_init__(self) -> None:
        if self.kind not in OK_KINDS:
            raise ValueError("change kind must be behavior or narrative")


def apply_influence_flag(ch: Change, index: SurfacingIndex) -> Change:
    flagged = index.would_flag(ch.user_id, ch.domain, ch.when)
    ch.potentially_claim_influenced = flagged
    ch.influenced = flagged
    return ch


def aspiration_evidence_weight(ch: Change, index: SurfacingIndex) -> float:
    """Hard rule: flagged (or flaggable) events contribute 0 to aspiration."""
    if index.would_flag(ch.user_id, ch.domain, ch.when):
        return 0.0
    if ch.potentially_claim_influenced or ch.influenced:
        return 0.0
    return 1.0


def product_copy(ch: Change) -> str:
    """Never mention the influence flag or its underlying data."""
    _ = ch.potentially_claim_influenced
    return f"update in {ch.domain}"


def copy_mentions_internal(text: str) -> bool:
    lowered = text.lower()
    banned = (INFLUENCE_FLAG, "influenced", "flag", "observer", "safeguard")
    return any(w in lowered for w in banned)
