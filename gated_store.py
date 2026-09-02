# integration/gated_store.py — Sprint 14 Day 40.
#
# Wraps Sprint 13's chronis_ml.store.IsolatedModelStore. The Sprint 13
# store already enforces cross-user isolation (a user can't read another
# user's model folder) — that guarantee is untouched here. What was
# missing, and what this file adds, is the constitutional-layer check:
# even a user's OWN model read/write must clear consent_tier >= 2 and a
# non-Mode-C operational mode BEFORE the underlying store method runs.
#
# This is intentionally a thin wrapper, not a reimplementation — Sprint
# 13's store.py is untouched (read-only upstream dependency). If Sprint 13
# ever changes its store's method signatures, this file is the one place
# that needs updating, not every caller across the codebase.
from __future__ import annotations

from pathlib import Path
from typing import Optional

from chronis_ml.store import IsolatedModelStore, IsolationError

from policy_engine.audit_log import AuditAction, AuditOutcome
from policy_engine.consent import ConsentRecord
from policy_engine.errors import RawDataRetentionError
from policy_engine.policy_rule import RuleAction
from policy_engine.principal import AccessRequest, ModelPrincipal

# Sprint 13's store only ever writes/reads model ARTIFACTS (hssm weights,
# personal_lm adapters) — never raw sensor/transcript data. This wrapper
# asserts that expectation at the boundary rather than trusting it
# silently, per "raw data is never retained by the model layer."
_DISALLOWED_RAW_EXTENSIONS = {".wav", ".flac", ".mp3", ".raw", ".pcm", ".transcript"}


def _reject_raw_payload(name: str, *, consent: ConsentRecord, principal: ModelPrincipal) -> None:
    suffix = Path(name).suffix.lower()
    if suffix in _DISALLOWED_RAW_EXTENSIONS:
        reason = (
            f"refusing to write {name!r} through the model store — extension "
            f"{suffix!r} looks like raw sensor/audio data, not a model artifact. "
            "The model layer must never retain raw data (Global Standard #3)."
        )
        # This is a denial like any other — audit it on the same stream,
        # even though it's rejected before reaching principal.check()'s
        # normal consent/rule flow (it's a content-shape rejection, not a
        # consent/mode rejection, so it doesn't go through check() itself).
        principal.audit.record(
            action=AuditAction.MODEL_WRITE,
            outcome=AuditOutcome.DENIED,
            principal_id=consent.user_id,
            reason=reason,
            detail={"name": name, "rejected_extension": suffix},
        )
        raise RawDataRetentionError(reason)


class GatedModelStore:
    """
    Drop-in replacement for chronis_ml.store.IsolatedModelStore in any code
    path that also needs constitutional-layer enforcement. Every method
    signature mirrors the wrapped store's, plus a required `consent`
    parameter so the caller can't accidentally skip supplying the subject's
    current consent/mode state.
    """

    def __init__(self, principal: ModelPrincipal, store: Optional[IsolatedModelStore] = None,
                 root: Optional[Path] = None) -> None:
        self._principal = principal
        self._store = store or IsolatedModelStore(root=root)

    def write(self, consent: ConsentRecord, uid: str, kind: str, name: str, data: bytes) -> Path:
        if consent.user_id != uid:
            # A caller writing to a different user's model folder than the
            # consent record describes is either a bug or an attempted
            # cross-user bypass. Audit it explicitly as a denial (so it's
            # visible in the log as a policy event, not just a generic
            # isolation exception further down), then raise unconditionally
            # — this never falls through to the real store, regardless of
            # what any registered rule says.
            self._principal.audit.record(
                action=AuditAction.MODEL_WRITE,
                outcome=AuditOutcome.DENIED,
                principal_id=consent.user_id,
                reason=f"attempted write to {uid!r}'s model folder using {consent.user_id!r}'s consent record.",
                detail={"attempted_by": consent.user_id, "target_user": uid, "name": name},
            )
            raise IsolationError(f"{consent.user_id!r} can't write to {uid!r}'s model folder")
        _reject_raw_payload(name, consent=consent, principal=self._principal)
        self._principal.check(AccessRequest(
            action=RuleAction.MODEL_WRITE,
            consent=consent,
            domain=kind,
            detail={"name": name, "size_bytes": len(data)},
        ))
        return self._store.write(uid, kind, name, data)

    def read(self, consent: ConsentRecord, path: Path) -> bytes:
        self._principal.check(AccessRequest(
            action=RuleAction.MODEL_READ,
            consent=consent,
            detail={"path": str(path)},
        ))
        return self._store.read(consent.user_id, path)

    def put_base(self, consent: ConsentRecord, ckpt: str, data: bytes) -> Path:
        # Writing the shared base checkpoint is still gated — "no bypass,
        # not even for shared/system artifacts" — but scoped to Class A
        # (shared, non-identity-bearing) territory, per Sprint 17's later
        # A/B split. We don't have that split's tagging yet in Sprint 14;
        # flagging this as the seam Sprint 17 will need to revisit.
        self._principal.check(AccessRequest(
            action=RuleAction.MODEL_WRITE,
            consent=consent,
            domain="_shared",
            detail={"checkpoint_id": ckpt, "size_bytes": len(data)},
        ))
        return self._store.put_base(ckpt, data)

    def load_base(self, consent: ConsentRecord, ckpt: str) -> bytes:
        self._principal.check(AccessRequest(
            action=RuleAction.MODEL_READ,
            consent=consent,
            domain="_shared",
            detail={"checkpoint_id": ckpt},
        ))
        return self._store.load_base(ckpt)

    # Isolation scanning (assert_src_isolated / scan) is a static-analysis
    # CI check, not a per-request data access — deliberately NOT wrapped
    # here. It doesn't touch user data at request time, so it isn't a
    # policy-engine concern; it stays a Sprint 13 CI-time tool, called
    # directly, not through this gate.