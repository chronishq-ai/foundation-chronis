# PolicyRule — the one schema every access rule in the constitutional layer
# is expressed as. Sprint 14 only needs the inference/claim-access rules
# below, but the schema itself is written to be the same primitive Sprint
# 17 Day 6 extends for `emergency_access_grant` (principal = trusted
# contact, scope = allowlist of claim levels/domains, duration = time-boxed
# and renewal-required). Building it generic now means Sprint 17 adds a new
# PolicyRule *instance*, not a new schema.
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from .consent import ConsentTier, OperationalMode
from .errors import PolicyRuleError


class RuleAction(str, Enum):
    """What kind of access this rule governs. Mirrors AuditAction so a rule
    evaluation and its resulting audit entry use the same vocabulary."""

    INFERENCE = "inference"
    CLAIM_ACCESS = "claim_access"
    MODEL_READ = "model_read"
    MODEL_WRITE = "model_write"
    REGISTRY_REGISTER = "registry_register"


@dataclass(frozen=True)
class Scope:
    """
    What a rule's grant actually covers. Defaults are maximally restrictive
    (nothing) — a rule must explicitly opt in to actions/domains/claim
    levels rather than a caller having to know to restrict it.
    """

    actions: frozenset[RuleAction] = field(default_factory=frozenset)
    claim_levels: frozenset[int] = field(default_factory=frozenset)  # empty = all levels
    domains: frozenset[str] = field(default_factory=frozenset)       # empty = all domains

    def allows_action(self, action: RuleAction) -> bool:
        return action in self.actions

    def allows_claim_level(self, level: int) -> bool:
        return not self.claim_levels or level in self.claim_levels

    def allows_domain(self, domain: str) -> bool:
        return not self.domains or domain in self.domains


@dataclass(frozen=True)
class PolicyRule:
    """
    principal      — who this rule grants access to. For Sprint 14's own
                      rules this is the model principal itself (or 'system').
                      Sprint 17 sets this to a user-designated trusted
                      contact's id for emergency_access_grant.
    subject_user_id — whose data the rule governs access to.
    scope           — what's covered (see Scope above).
    min_consent_tier — floor the subject's own consent record must meet
                      independent of this rule; a rule can never lower the
                      subject's own consent floor, only add scoping on top
                      of it.
    allowed_modes   — operational modes this rule's grant is valid under.
                      Mode C is never a legal value here — see __post_init__.
    granted_at / expires_at — time-boxing. `expires_at=None` means the rule
                      never auto-expires (only Sprint 14's own system rules
                      should ever use this; anything principal-scoped to a
                      person, per Sprint 17, must set an expiry).
    requires_renewal — if True, callers must not treat a still-valid rule as
                      permanent; product surface must prompt renewal before
                      expires_at. (Enforcing the prompt is a product-layer
                      concern; this flag just carries the requirement.)
    """

    rule_id: str
    principal: str
    subject_user_id: str
    scope: Scope
    min_consent_tier: ConsentTier
    allowed_modes: frozenset[OperationalMode]
    granted_at: datetime
    expires_at: Optional[datetime] = None
    requires_renewal: bool = False

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise PolicyRuleError("rule_id must be non-empty")
        if OperationalMode.MODE_C in self.allowed_modes:
            raise PolicyRuleError(
                f"rule {self.rule_id!r}: Mode C cannot appear in allowed_modes — "
                "Raw Vault is never a legal grant target for any rule."
            )
        if not self.allowed_modes:
            raise PolicyRuleError(f"rule {self.rule_id!r}: allowed_modes must be non-empty")
        if self.expires_at is not None and self.expires_at <= self.granted_at:
            raise PolicyRuleError(f"rule {self.rule_id!r}: expires_at must be after granted_at")
        if self.requires_renewal and self.expires_at is None:
            raise PolicyRuleError(
                f"rule {self.rule_id!r}: requires_renewal=True needs a real expires_at"
            )

    def is_active(self, at: Optional[datetime] = None) -> bool:
        at = at or datetime.now(timezone.utc)
        if self.expires_at is None:
            return True
        return at < self.expires_at

    def covers(self, *, action: RuleAction, mode: OperationalMode, claim_level: Optional[int] = None,
               domain: Optional[str] = None, at: Optional[datetime] = None) -> bool:
        """True only if this single rule, on its own, would authorize the
        request. Combining multiple rules (e.g. "any matching rule grants")
        is the model principal's job, not this class's."""
        if not self.is_active(at):
            return False
        if mode not in self.allowed_modes:
            return False
        if not self.scope.allows_action(action):
            return False
        if claim_level is not None and not self.scope.allows_claim_level(claim_level):
            return False
        if domain is not None and not self.scope.allows_domain(domain):
            return False
        return True


def default_system_inference_rule(subject_user_id: str, *, granted_at: Optional[datetime] = None) -> PolicyRule:
    """
    The standing rule Sprint 14 wires by default: the model principal itself
    may run inference/claim_access/model_read/model_write for a user, under
    Mode A or B, provided the user's own ConsentRecord separately clears
    `check_inference_consent` (this rule does not replace that check — see
    principal.py, both must pass). Never expires because it's not a
    person-scoped grant; it's the system's own standing operating rule.
    """
    return PolicyRule(
        rule_id=f"system-inference-{subject_user_id}",
        principal="system",
        subject_user_id=subject_user_id,
        scope=Scope(
            actions=frozenset(
                {RuleAction.INFERENCE, RuleAction.CLAIM_ACCESS, RuleAction.MODEL_READ, RuleAction.MODEL_WRITE}
            )
        ),
        min_consent_tier=ConsentTier.INFERENCE,
        allowed_modes=frozenset({OperationalMode.MODE_A, OperationalMode.MODE_B}),
        granted_at=granted_at or datetime.now(timezone.utc),
        expires_at=None,
        requires_renewal=False,
    )