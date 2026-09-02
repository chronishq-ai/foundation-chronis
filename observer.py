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
from .profiles import TYPES, plant_profiles, type_accuracy
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
    """Back-compat for tiny dict profiles.

    Structured b/n arrays go through the real planted-profile pipeline
    (HSSM → NSSM → DivergenceState.type_scores). Scratch type_scores() is
    never used on the live path (closure S15.1).
    """
    if "b" in profile and "n" in profile:
        import numpy as np
        from divergence_engine.engine import DivergenceInputs, compute_divergence_state
        from backbone.hssm import fit_hssm
        from nssm_pipeline import fit_nssm
        from datetime import datetime, timezone

        b = np.asarray(profile["b"], dtype=float)
        n = np.asarray(profile["n"], dtype=float)
        raw = np.stack([b, n], axis=1)
        hssm_out = fit_hssm(raw)
        nssm_out = fit_nssm(raw)
        weakening = bool((hssm_out.m_t[0] - hssm_out.m_t[-1]) > 0.15)
        state = compute_divergence_state(
            DivergenceInputs(
                user_id="classify",
                domain_id="dom",
                window_start=datetime.now(timezone.utc),
                window_end=datetime.now(timezone.utc),
                p_t=hssm_out.p_t,
                q_t=nssm_out.q_t,
                m_t=hssm_out.m_t,
                n_t=nssm_out.n_t,
                behavioral_regime_id=1,
                narrative_regime_id=1,
                n_domain_pairs_tested=1,
                behavioral_attractor_weakening=weakening,
                narrative_conformal_confidence=0.8,
            )
        )
        if state.dominant_type:
            return state.dominant_type
        return max(state.type_scores, key=state.type_scores.get)
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
