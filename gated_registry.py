# integration/gated_registry.py — Sprint 14 Day 40.
#
# Wraps Sprint 13's chronis_ml.ops.Registry. Registry.register() already
# enforces the MLflow logging-contract gate (training_data_hash,
# hyperparameters, metrics, fit_date — Global Standard #7, no silent magic
# numbers) — that guarantee is untouched here. What this file adds is the
# constitutional-layer check in front of it: a model version cannot be
# registered for a user unless that user's own consent clears the
# inference floor and a legal operational mode, exactly like every other
# ML data write.
#
# Registry.register() is also, structurally, a MODEL_WRITE — it writes a
# training-data hash and a fitted artifact pointer into MLflow, tagged to
# a specific user. It gets its own RuleAction (REGISTRY_REGISTER) rather
# than reusing MODEL_WRITE, because a rule that grants "read/write my own
# model artifacts" (default_system_inference_rule) should not automatically
# also grant "publish a new registered version" — those are different
# operational stakes and Sprint 14's default rule intentionally does NOT
# cover REGISTRY_REGISTER (see principal.py smoke test #7). A caller must
# register a rule that explicitly includes REGISTRY_REGISTER before this
# will ever grant.
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from chronis_ml.ops import Registry

from policy_engine.audit_log import AuditAction, AuditOutcome
from policy_engine.consent import ConsentRecord
from policy_engine.policy_rule import RuleAction
from policy_engine.principal import AccessRequest, ModelPrincipal


class GatedRegistry:
    """
    Drop-in wrapper around chronis_ml.ops.Registry. Same register()
    contract, plus a required `consent` parameter and a policy-engine
    check before anything reaches MLflow.
    """

    def __init__(self, principal: ModelPrincipal, registry: Optional[Registry] = None,
                 tracking_uri: Optional[str] = None, root: Optional[Path] = None) -> None:
        self._principal = principal
        self._registry = registry or Registry(tracking_uri=tracking_uri, root=root)

    def register(self, consent: ConsentRecord, uid: str, kind: str, artifact: Path,
                 payload: dict[str, Any], why: str) -> str:
        if consent.user_id != uid:
            # Never fabricate a substitute ConsentRecord for `uid` here —
            # doing so (even just to force a check() call) would mean
            # forging consent state that was never actually verified for
            # that user, and if a rule happened to exist matching that
            # fabricated shape, it would wrongly grant. Deny directly,
            # audit it explicitly on the same stream as every other
            # denial, and never touch principal.check()'s rule-matching
            # path for this case at all.
            reason = (
                f"attempted to register a model version for {uid!r} using "
                f"{consent.user_id!r}'s consent record — consent and target "
                "user must match."
            )
            self._principal.audit.record(
                action=AuditAction.REGISTRY_REGISTER,
                outcome=AuditOutcome.DENIED,
                principal_id=consent.user_id,
                reason=reason,
                detail={"attempted_by": consent.user_id, "target_user": uid, "kind": kind},
            )
            raise PermissionError(reason)

        self._principal.check(AccessRequest(
            action=RuleAction.REGISTRY_REGISTER,
            consent=consent,
            domain=kind,
            detail={
                "training_data_hash": payload.get("training_data_hash"),
                "fit_date": str(payload.get("fit_date")),
                "why": why,
            },
        ))
        return self._registry.register(uid, kind, artifact, payload, why)