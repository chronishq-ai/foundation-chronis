# typed errors for the policy engine. every deny/tamper case raises one of
# these — never a bare Exception/ValueError — so callers (and tests) can
# assert on the exact failure mode instead of parsing a message string.
from __future__ import annotations


class PolicyEngineError(Exception):
    """Base class for everything in policy_engine. Never raised directly."""


class PolicyDenied(PolicyEngineError):
    """
    Raised whenever the model principal blocks a read/write/inference.

    Every raise site MUST also produce a matching audit-log entry — a denial
    is audited exactly like a grant (Sprint 14 Day 40 requirement). This
    exception carries `reason` so the audit entry and the raised error say
    the same thing.
    """

    def __init__(self, reason: str, *, principal_id: str | None = None) -> None:
        self.reason = reason
        self.principal_id = principal_id
        super().__init__(reason)


class ConsentTierError(PolicyDenied):
    """consent_tier < 2 for an operation that requires inference-level consent."""


class ModeCBlocked(PolicyDenied):
    """
    Mode C (Raw Vault) hard block at the ML layer.

    This must never be bypassable — not by a higher consent_tier, not by a
    retry, not in dev. If code ever needs to special-case this away, that is
    a bug, not a feature.
    """


class RawDataRetentionError(PolicyEngineError):
    """
    Raised if any code path attempts to have the model layer retain raw
    (non-abstracted) data past the boundary of a single inference call.
    """


class AuditTamperError(PolicyEngineError):
    """
    Raised when the audit log's hash-chain verification fails — i.e. an
    entry was altered or removed after the fact. This is a detection signal,
    not a recovery mechanism: the log itself never self-heals.
    """

    def __init__(self, reason: str, *, at_index: int | None = None) -> None:
        self.reason = reason
        self.at_index = at_index
        super().__init__(reason)


class PolicyRuleError(PolicyEngineError):
    """Malformed or invalid PolicyRule definition."""