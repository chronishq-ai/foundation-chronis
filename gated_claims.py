# integration/gated_claims.py — Sprint 14 Day 40.
#
# Sprint 9's claims_engine/surfacing_policy.py already has the right SHAPE:
# SurfacingContext.constitutional_restriction_active is exactly the seam
# the directive describes ("Route every ML data read/write through the
# policy engine's model principal... Verify the Mode C hard block at the
# ML layer specifically"). What Sprint 9 did NOT have is anything that
# actually SETS that flag from a real policy decision — it was a
# caller-supplied bool with no real backend.
#
# This file is that backend. `evaluate_claim_access` is the one place
# constitutional_restriction_active gets computed, from a real
# ModelPrincipal.check() call — never hand-set true/false anywhere else.
# It also satisfies the directive's separate requirement, distinct from
# the surfacing policy's own gates: "Emit an audit-log entry for every
# inference and every claim access, regardless of outcome." Every claim
# lookup here — whether the eventual surfacing decision is SURFACE,
# UNCLEAR, or WITHHOLD — produces exactly one audit entry, because
# ModelPrincipal.check()/is_permitted() always audits regardless of the
# outcome it returns.
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from claims_engine.claim_levels import Claim, ClaimLevel, GateEvaluation
from claims_engine.surfacing_policy import SurfacingContext, SurfacingResult, decide_surfacing

from policy_engine.consent import ConsentRecord
from policy_engine.errors import PolicyDenied
from policy_engine.policy_rule import RuleAction
from policy_engine.principal import AccessRequest, ModelPrincipal


@dataclass(frozen=True)
class ClaimAccessInputs:
    """
    Everything gated_claims needs beyond what the policy engine itself
    tracks — the claim-specific safety-gate inputs Sprint 9 already
    defined. This module does not reinterpret or relax any of these; it
    only supplies the one field (constitutional_restriction_active) that
    was previously a bare caller-supplied bool.
    """

    acute_trauma_markers_present: bool
    has_therapeutic_context: bool
    self_protection_gate_failed: bool
    contradiction_without_new_evidence: bool
    conflict_record_unresolved: bool = False


def evaluate_claim_access(
    principal: ModelPrincipal,
    consent: ConsentRecord,
    claim: Optional[Claim],
    gate_eval: GateEvaluation,
    inputs: ClaimAccessInputs,
    *,
    domain_id: Optional[str] = None,
) -> SurfacingResult:
    """
    The real, policy-engine-backed replacement for hand-constructing a
    SurfacingContext. `domain_id` defaults to `claim.domain_id` when a
    claim exists (Level 0/1/2/3 claims always carry one); pass it
    explicitly when `claim` is None (a lookup that failed to even produce
    a claim object still needs a domain to scope the access check against).
    """
    resolved_domain = domain_id or (claim.domain_id if claim is not None else None)

    constitutional_restriction_active = not principal.is_permitted(
        AccessRequest(
            action=RuleAction.CLAIM_ACCESS,
            consent=consent,
            claim_level=gate_eval.level.value,
            domain=resolved_domain,
            detail={"claim_id": claim.claim_id if claim is not None else None},
        )
    )

    ctx = SurfacingContext(
        acute_trauma_markers_present=inputs.acute_trauma_markers_present,
        has_therapeutic_context=inputs.has_therapeutic_context,
        constitutional_restriction_active=constitutional_restriction_active,
        self_protection_gate_failed=inputs.self_protection_gate_failed,
        contradiction_without_new_evidence=inputs.contradiction_without_new_evidence,
        conflict_record_unresolved=inputs.conflict_record_unresolved,
    )

    # decide_surfacing itself is untouched Sprint 9 code — we don't
    # reimplement or shadow its logic, only feed it a real ctx.
    return decide_surfacing(claim, gate_eval, ctx)