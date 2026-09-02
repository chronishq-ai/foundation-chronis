# integration/gated_divergence.py — Sprint 14 Day 40.
#
# compute_divergence_state (Sprint 8) is the clearest example in the whole
# codebase of what "inference" means under the directive: it runs Fisher's
# exact and Bayesian MS-VAR Granger tests over a user's already-abstracted
# regime arrays (p_t/q_t/m_t/n_t — never raw audio/sensor data, those were
# consumed upstream in Sprint 1-7) and produces a new DivergenceState. That
# computation is squarely "inference is only permitted where
# consent_tier >= 2" territory — distinct from CLAIM_ACCESS (reading an
# already-computed claim) and MODEL_READ/WRITE (touching stored artifacts).
#
# Sprint 8's engine.py is untouched. This wrapper does exactly one thing:
# check() before compute_divergence_state() runs, so a denied user's
# regime data never even reaches the Fisher/Granger tests.
#
# Note on "transform always abstracted": DivergenceInputs already only
# carries regime labels and fast-state arrays (m_t/n_t), never transcript
# text or raw sensor streams — that abstraction is inherited from Sprint
# 8's own design, not added here. This wrapper's job is purely the
# consent/mode gate in front of it, not re-deriving that guarantee.
from __future__ import annotations

from typing import Optional

from divergence_engine.engine import DivergenceInputs, compute_divergence_state
from divergence_engine.state import DivergenceState

from policy_engine.audit_log import AuditAction, AuditOutcome
from policy_engine.consent import ConsentRecord
from policy_engine.policy_rule import RuleAction
from policy_engine.principal import AccessRequest, ModelPrincipal


def gated_compute_divergence_state(
    principal: ModelPrincipal,
    consent: ConsentRecord,
    inputs: DivergenceInputs,
    previous_state_id: Optional[str] = None,
) -> DivergenceState:
    """
    Real policy-engine-gated entry point for Sprint 8's divergence
    computation. Raises PolicyDenied (or a subtype) if the subject's
    consent/mode don't clear the inference floor — compute_divergence_state
    is never called in that case, so no Fisher/Granger test ever runs
    against a denied user's data, and the denial is audited exactly like
    a grant.
    """
    if consent.user_id != inputs.user_id:
        # Same discipline as gated_store/gated_registry: never fabricate a
        # substitute consent record for inputs.user_id — deny directly,
        # audit explicitly, don't risk a coincidental rule match.
        reason = (
            f"attempted divergence inference for {inputs.user_id!r} using "
            f"{consent.user_id!r}'s consent record — consent and target "
            "user must match."
        )
        principal.audit.record(
            action=AuditAction.INFERENCE,
            outcome=AuditOutcome.DENIED,
            principal_id=consent.user_id,
            reason=reason,
            detail={
                "attempted_by": consent.user_id,
                "target_user": inputs.user_id,
                "domain_id": inputs.domain_id,
            },
        )
        raise PermissionError(reason)

    principal.check(AccessRequest(
        action=RuleAction.INFERENCE,
        consent=consent,
        domain=inputs.domain_id,
        detail={
            "behavioral_regime_id": inputs.behavioral_regime_id,
            "narrative_regime_id": inputs.narrative_regime_id,
            "n_domain_pairs_tested": inputs.n_domain_pairs_tested,
        },
    ))
    return compute_divergence_state(inputs, previous_state_id)