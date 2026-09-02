from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .claim_levels import Claim, GateEvaluation


class SurfaceDecision(str, Enum):
    SURFACE = "surface"
    UNCLEAR = "unclear"
    WITHHOLD = "withhold"


@dataclass(frozen=True)
class SurfacingContext:
    acute_trauma_markers_present: bool
    has_therapeutic_context: bool
    constitutional_restriction_active: bool
    self_protection_gate_failed: bool
    contradiction_without_new_evidence: bool
    conflict_record_unresolved: bool = False


@dataclass(frozen=True)
class SurfacingResult:
    decision: SurfaceDecision
    reason: str


def decide_surfacing(
    claim: Optional[Claim],
    gate_eval: GateEvaluation,
    ctx: SurfacingContext,
) -> SurfacingResult:
    if ctx.constitutional_restriction_active:
        return SurfacingResult(SurfaceDecision.WITHHOLD, "Constitutional restriction active")
    if not gate_eval.admissible:
        return SurfacingResult(SurfaceDecision.WITHHOLD, "Gate not admissible")
    if claim is None:
        return SurfacingResult(SurfaceDecision.UNCLEAR, "no claim object")
    # Level 0/1: constitutional + gate only. Higher-level safety flags apply at 2/3.
    if gate_eval.level.value <= 1:
        return SurfacingResult(SurfaceDecision.SURFACE, "level 0/1 gate cleared")
    if ctx.acute_trauma_markers_present and not ctx.has_therapeutic_context:
        return SurfacingResult(SurfaceDecision.WITHHOLD, "acute trauma without therapeutic context")
    if ctx.self_protection_gate_failed:
        return SurfacingResult(SurfaceDecision.WITHHOLD, "self-protection gate failed")
    if ctx.contradiction_without_new_evidence:
        return SurfacingResult(SurfaceDecision.UNCLEAR, "contradiction without new evidence")
    if ctx.conflict_record_unresolved:
        return SurfacingResult(SurfaceDecision.UNCLEAR, "unresolved conflict record")
    return SurfacingResult(SurfaceDecision.SURFACE, "higher-level gates cleared")
