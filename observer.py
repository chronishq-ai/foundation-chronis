# Sprint 15 — Observer-Effect Safeguard & Active-Transition Closure (Days 43–45).
#
# This mitigates MP-13. It does not solve it. Showing a claim still changes
# what the person does next; we only refuse to treat that change as
# independent proof of aspiration.
from __future__ import annotations

from datetime import date
from typing import Optional

from claims_engine.claim_levels import Claim, ClaimLevel
from claims_engine.surfacing_policy import SurfaceDecision, SurfacingResult

from .index import INFLUENCE_FLAG, INFLUENCE_WINDOW_DAYS, SurfacedClaim, SurfacingIndex
from .profiles import TYPES, plant_profiles, type_accuracy, type_scores
from .regression import cold_start_180, mirror_allowed, stage_for_sessions
from .safeguard import Change, apply_influence_flag, aspiration_evidence_weight, product_copy


class Observer:
    def __init__(self, index: Optional[SurfacingIndex] = None) -> None:
        self.index = index or SurfacingIndex()
        self.changes: list[Change] = []

    @property
    def surfaced(self) -> list[SurfacedClaim]:
        return list(self.index)

    def record_surfaced(self, rec: SurfacedClaim) -> Optional[SurfacedClaim]:
        return self.index.append(rec)

    def note_shown_claim(
        self,
        claim: Claim,
        result: SurfacingResult,
        when: date,
        *,
        div_type: str = "",
        **_ignored,
    ) -> Optional[SurfacedClaim]:
        if result.decision != SurfaceDecision.SURFACE:
            return None
        if int(claim.level) < int(ClaimLevel.LEVEL_1):
            return None
        rec = SurfacedClaim(
            claim_id=claim.claim_id,
            user_id=claim.user_id,
            domain=claim.domain_id,
            level=int(claim.level),
            div_type=div_type,
            when=when,
        )
        return self.record_surfaced(rec)

    def note_change(self, ch: Change) -> Change:
        apply_influence_flag(ch, self.index)
        self.changes.append(ch)
        return ch

    def aspiration_weight(self, ch: Change) -> float:
        return aspiration_evidence_weight(ch, self.index)

    def product_copy(self, ch: Change) -> str:
        return product_copy(ch)


def cold_start_silent(stage: int, observer: Observer | None = None) -> bool:
    _ = observer
    return not mirror_allowed(stage)


def classify(profile: dict) -> str:
    """Back-compat for tiny dict profiles; Day 44 uses PlantedProfile + type_scores."""
    if "b" in profile and "n" in profile:
        scores = type_scores(profile["b"], profile["n"])
        return max(scores, key=scores.get)
    b = profile.get("behavior")
    n = profile.get("narrative")
    lag = profile.get("lag", 0)
    if b == "strong" and n == "none":
        return "Ignorance"
    if b == "weakening" and n == "agentic":
        return "Aspiration"
    if b == "stable" and n == "avoidant":
        return "Self-Protection"
    if b == "weakening" and n == "changing" and lag != 0:
        return "ActiveTransition"
    return "Ignorance"
