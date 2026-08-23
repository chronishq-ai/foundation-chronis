# ModelPrincipal — the single choke point.
#
# Directive (Day 40): "Route every ML data read/write through the policy
# engine's model principal." This is that principal. Nothing downstream
# (gated_store, gated_registry, gated_claims, gated_divergence) is allowed
# to do its own consent/mode check — they all call through here, so there
# is exactly one place the rule "inference is only permitted where
# consent_tier >= 2, transform always abstracted, no bypass" can drift out
# of sync with reality.
#
# Every check() call — granted or denied — produces exactly one audit
# entry. That's enforced structurally: check() is the only method that
# calls self._audit.record(), and every code path through it (success,
# ConsentTierError, ModeCBlocked, no-matching-rule) ends in exactly one
# record() call before returning or raising.
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .audit_log import AuditAction, AuditLog, AuditOutcome
from .consent import ConsentRecord, check_inference_consent
from .errors import ConsentTierError, ModeCBlocked, PolicyDenied
from .policy_rule import PolicyRule, RuleAction, default_system_inference_rule

# RuleAction and AuditAction are deliberately separate enums (different
# modules, different concerns) but must stay in lockstep — this map is the
# one place that fact is encoded, so a new action added to one and not the
# other fails loudly here instead of silently miscategorizing in the log.
_RULE_TO_AUDIT_ACTION: dict[RuleAction, AuditAction] = {
    RuleAction.INFERENCE: AuditAction.INFERENCE,
    RuleAction.CLAIM_ACCESS: AuditAction.CLAIM_ACCESS,
    RuleAction.MODEL_READ: AuditAction.MODEL_READ,
    RuleAction.MODEL_WRITE: AuditAction.MODEL_WRITE,
    RuleAction.REGISTRY_REGISTER: AuditAction.REGISTRY_REGISTER,
}


@dataclass(frozen=True)
class AccessRequest:
    """What a caller (gated_store, gated_registry, gated_claims, ...) is
    asking the principal to authorize. `mode` and `consent` both describe
    the *subject* user's current state — the principal doesn't look these
    up itself, callers supply them, so this class stays testable without a
    real consent/mode backend."""

    action: RuleAction
    consent: ConsentRecord
    claim_level: Optional[int] = None
    domain: Optional[str] = None
    at: Optional[datetime] = None
    detail: Optional[dict] = None


class ModelPrincipal:
    """
    Holds: an audit log, and a per-user set of active PolicyRules.

    check(request) is the ONLY entry point. It either returns None (granted)
    or raises PolicyDenied (or a subclass). Callers must treat "no
    exception" as the only valid signal of grant — never inspect audit log
    contents to infer authorization, since that inverts the dependency the
    audit log exists to record.
    """

    def __init__(self, audit_log: Optional[AuditLog] = None) -> None:
        self.audit = audit_log or AuditLog()
        self._rules: dict[str, list[PolicyRule]] = {}

    def register_rule(self, rule: PolicyRule) -> None:
        self._rules.setdefault(rule.subject_user_id, []).append(rule)

    def ensure_default_rule(self, user_id: str) -> None:
        """Convenience for tests/integration code: wires the standing
        system-inference rule for a user if nothing's registered yet.
        Does NOT overwrite or duplicate if a rule set already exists."""
        if user_id not in self._rules:
            self.register_rule(default_system_inference_rule(user_id))

    def _matching_rule(self, request: AccessRequest) -> Optional[PolicyRule]:
        for rule in self._rules.get(request.consent.user_id, []):
            if rule.covers(
                action=request.action,
                mode=request.consent.mode,
                claim_level=request.claim_level,
                domain=request.domain,
                at=request.at,
            ):
                return rule
        return None

    def check(self, request: AccessRequest) -> None:
        """
        Raises on denial, returns None on grant. Every path logs exactly
        once before returning/raising.

        Order of checks matters for the audit trail's `reason` field, but
        NOT for correctness — Mode C block and consent-tier floor are both
        independently enforced regardless of which is checked first, since
        check_inference_consent() itself re-verifies Mode C internally
        (see consent.py). A rule can never widen either of those two
        floors; it can only add scoping restrictions on top.
        """
        audit_action = _RULE_TO_AUDIT_ACTION[request.action]
        detail = dict(request.detail or {})
        if request.claim_level is not None:
            detail["claim_level"] = request.claim_level
        if request.domain is not None:
            detail["domain"] = request.domain

        try:
            check_inference_consent(request.consent)
        except (ConsentTierError, ModeCBlocked) as e:
            self.audit.record(
                action=audit_action,
                outcome=AuditOutcome.DENIED,
                principal_id=request.consent.user_id,
                reason=e.reason,
                detail=detail,
            )
            raise

        rule = self._matching_rule(request)
        if rule is None:
            reason = (
                f"no active PolicyRule grants {request.action.value} for "
                f"user {request.consent.user_id!r} under mode "
                f"{request.consent.mode.name} at the requested scope."
            )
            self.audit.record(
                action=audit_action,
                outcome=AuditOutcome.DENIED,
                principal_id=request.consent.user_id,
                reason=reason,
                detail=detail,
            )
            raise PolicyDenied(reason, principal_id=request.consent.user_id)

        self.audit.record(
            action=audit_action,
            outcome=AuditOutcome.GRANTED,
            principal_id=request.consent.user_id,
            reason=f"matched rule {rule.rule_id!r}",
            detail={**detail, "rule_id": rule.rule_id},
        )

    def is_permitted(self, request: AccessRequest) -> bool:
        """Non-raising probe. Still audits — a probe that silently skipped
        auditing would violate the 'every inference/claim access, regardless
        of outcome' requirement, since a probe result can and does inform
        real product behavior (e.g. whether to even attempt a flow)."""
        try:
            self.check(request)
        except PolicyDenied:
            return False
        return True