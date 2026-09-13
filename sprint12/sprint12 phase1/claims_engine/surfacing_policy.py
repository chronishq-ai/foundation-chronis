"""
claims_engine/surfacing_policy.py

Sprint 9, Day 26 — surface / UNCLEAR / withhold-entirely logic.

"Silence is a valid, correct output" (directive, Non-Negotiables). Below a
gate, the right behavior is often literally nothing.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .claim_levels import Claim, ClaimLevel, GateEvaluation


class SurfaceDecision(Enum):
    SURFACE = "surface"
    UNCLEAR = "unclear"
    WITHHOLD = "withhold"


@dataclass(frozen=True)
class SurfacingContext:
    acute_trauma_markers_present: bool
    has_therapeutic_context: bool
    constitutional_restriction_active: bool   # policy-engine block, e.g. consent_tier < 2
    self_protection_gate_failed: bool         # from Level 3 reflective-engagement gate
    contradiction_without_new_evidence: bool
    conflict_record_unresolved: bool = False  # Sprint 20's typed ConflictRecord, wired for forward-compat


@dataclass(frozen=True)
class SurfacingResult:
    decision: SurfaceDecision
    reason: str


def decide_surfacing(claim: Optional[Claim], gate_eval: GateEvaluation, ctx: SurfacingContext) -> SurfacingResult:
    """
    Level 0/1 claims: surfaced whenever their own gate is admissible — no
    additional policy overlay beyond the constitutional restriction check.

    Level 2/3 claims: subject to the full surface/UNCLEAR/withhold decision.
    """
    if not gate_eval.admissible:
        failed = [c.name for c in gate_eval.failed_checks()]
        return SurfacingResult(SurfaceDecision.WITHHOLD, f"Gate not admissible: {failed}")

    if ctx.constitutional_restriction_active:
        return SurfacingResult(SurfaceDecision.WITHHOLD, "Constitutional policy engine restriction active.")

    if gate_eval.level in (ClaimLevel.LEVEL_0, ClaimLevel.LEVEL_1):
        return SurfacingResult(SurfaceDecision.SURFACE, "Level 0/1 claim, gate admissible, no restriction.")

    # Level 2/3 from here.
    if ctx.acute_trauma_markers_present and not ctx.has_therapeutic_context:
        return SurfacingResult(SurfaceDecision.WITHHOLD, "Acute trauma markers without therapeutic context.")

    if ctx.self_protection_gate_failed:
        return SurfacingResult(SurfaceDecision.WITHHOLD, "Self-protection reflective-engagement gate failed.")

    if ctx.contradiction_without_new_evidence or ctx.conflict_record_unresolved:
        return SurfacingResult(SurfaceDecision.UNCLEAR, "Contradiction without new evidence / unresolved conflict.")

    return SurfacingResult(SurfaceDecision.SURFACE, "All Level 2/3 policy checks cleared.")
