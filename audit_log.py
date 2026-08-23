# append-only, hash-chained audit log.
#
# Directive requirement (Day 40): "Emit an audit-log entry for every
# inference and every claim access, regardless of outcome — a denied access
# is audited exactly like a granted one." There is no separate "denial log"
# and "grant log" — one stream, one schema, outcome is just a field.
#
# Directive requirement (Day 41): simulate audit-log tampering and verify
# the hash-chained, append-only structure detects it. That's `verify()`
# below — it walks the chain and raises AuditTamperError the moment a link
# doesn't match, naming the index where it broke.
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterator

from .errors import AuditTamperError

GENESIS_HASH = "0" * 64


class AuditOutcome(str, Enum):
    GRANTED = "granted"
    DENIED = "denied"


class AuditAction(str, Enum):
    INFERENCE = "inference"
    CLAIM_ACCESS = "claim_access"
    MODEL_READ = "model_read"
    MODEL_WRITE = "model_write"
    REGISTRY_REGISTER = "registry_register"


@dataclass(frozen=True)
class AuditEntry:
    """
    One immutable audit record. `prev_hash` + this entry's own content
    determine `entry_hash` — that's the chain link. Two identical entries
    appended at different chain positions still get different hashes,
    because `prev_hash` differs.
    """

    index: int
    timestamp: str  # ISO-8601 UTC, string not datetime so hashing is deterministic
    action: AuditAction
    outcome: AuditOutcome
    principal_id: str
    reason: str
    detail: dict[str, Any]
    prev_hash: str
    entry_hash: str = field(compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "action": self.action.value,
            "outcome": self.outcome.value,
            "principal_id": self.principal_id,
            "reason": self.reason,
            "detail": self.detail,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }


def _content_hash(
    index: int,
    timestamp: str,
    action: AuditAction,
    outcome: AuditOutcome,
    principal_id: str,
    reason: str,
    detail: dict[str, Any],
    prev_hash: str,
) -> str:
    # sort_keys=True so dict ordering never changes the hash; separators
    # strip whitespace so re-serialization elsewhere can't drift the digest.
    payload = json.dumps(
        {
            "index": index,
            "timestamp": timestamp,
            "action": action.value,
            "outcome": outcome.value,
            "principal_id": principal_id,
            "reason": reason,
            "detail": detail,
            "prev_hash": prev_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditLog:
    """
    In-memory reference implementation of the hash-chained audit log.

    Real deployment backs this with an append-only store (e.g. a
    write-once table or object storage with object-lock) — this class
    defines the chaining/verification contract that backend must satisfy;
    it is not itself a durability guarantee.
    """

    def __init__(self, *, now: Any = None) -> None:
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._entries: list[AuditEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[AuditEntry]:
        return iter(self._entries)

    @property
    def head_hash(self) -> str:
        return self._entries[-1].entry_hash if self._entries else GENESIS_HASH

    def record(
        self,
        *,
        action: AuditAction,
        outcome: AuditOutcome,
        principal_id: str,
        reason: str,
        detail: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Append one entry. This is the ONLY way entries enter the log —
        there is no update/delete method anywhere on this class, by design
        (mirrors Layer 0's "nothing is ever deleted" doctrine)."""
        index = len(self._entries)
        timestamp = self._now().isoformat()
        prev_hash = self.head_hash
        detail = detail or {}
        entry_hash = _content_hash(
            index, timestamp, action, outcome, principal_id, reason, detail, prev_hash
        )
        entry = AuditEntry(
            index=index,
            timestamp=timestamp,
            action=action,
            outcome=outcome,
            principal_id=principal_id,
            reason=reason,
            detail=detail,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )
        self._entries.append(entry)
        return entry

    def verify(self) -> None:
        """
        Walk the full chain from genesis. Raises AuditTamperError at the
        first index whose recomputed hash doesn't match what's stored, or
        whose prev_hash doesn't match the prior entry's entry_hash.

        This catches: content edited in place, an entry deleted from the
        middle, entries reordered, or entry_hash forged without updating
        the content it should reflect.
        """
        prev_hash = GENESIS_HASH
        for i, entry in enumerate(self._entries):
            if entry.index != i:
                raise AuditTamperError(
                    f"entry at position {i} claims index {entry.index} — reordered or deleted.",
                    at_index=i,
                )
            if entry.prev_hash != prev_hash:
                raise AuditTamperError(
                    f"entry {i} prev_hash does not match prior entry's hash — chain broken.",
                    at_index=i,
                )
            recomputed = _content_hash(
                entry.index,
                entry.timestamp,
                entry.action,
                entry.outcome,
                entry.principal_id,
                entry.reason,
                entry.detail,
                entry.prev_hash,
            )
            if recomputed != entry.entry_hash:
                raise AuditTamperError(
                    f"entry {i} content does not match its stored hash — edited after the fact.",
                    at_index=i,
                )
            prev_hash = entry.entry_hash

    def entries_for(self, principal_id: str) -> list[AuditEntry]:
        return [e for e in self._entries if e.principal_id == principal_id]