"""
frontier/claims_engine_adapter.py

Sprint 20 / S1720.10 / R2-F20.6 / R2-F20.7

Production adapter over the Sprint 13 Claims Engine.
Implements a durable append-only store with immutable versions.

Key fixes vs previous implementation:
  - NO pickle.loads / pickle.dumps anywhere (R2-F20.6)
  - Serialisation: dataclasses.asdict() → json.dumps()
  - Atomic writes: write to .tmp file, os.replace() on success
  - Truncated-tail handling: per-line try/except json.JSONDecodeError
  - schema_version field on every written record
  - get_claim_status(unknown_id) → None, not "SURFACED"  (R2-F20.7)
  - update_claim_status(nonexistent_id) → raises KeyError  (R2-F20.7)
"""

import copy
import dataclasses
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .interfaces.claims_store import ClaimsStoreProvider
from claims_engine.claim_levels import Claim, ClaimLevel

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def _claim_to_dict(claim: Claim) -> Dict:
    """Safe JSON-serialisable representation of a Claim (no pickle)."""
    d = dataclasses.asdict(claim)
    # dataclasses.asdict recurses; convert non-serialisable types
    d["level"] = claim.level.value
    d["gate_evaluation"] = {
        "level": claim.gate_evaluation.level.value,
        "admissible": claim.gate_evaluation.admissible,
        "checks": [
            {"name": c.name, "passed": c.passed, "detail": c.detail}
            for c in claim.gate_evaluation.checks
        ],
    }
    if claim.created_at is not None:
        d["created_at"] = claim.created_at.isoformat()
    # dual_structure_components may contain nested Claims — drop for now
    d.pop("dual_structure_components", None)
    return d


def _dict_to_claim(d: Dict) -> Claim:
    """Reconstruct a Claim from its safe JSON representation."""
    from claims_engine.claim_levels import GateEvaluation, GateCheck
    from datetime import datetime

    gate_data = d.get("gate_evaluation", {})
    checks = [
        GateCheck(name=c["name"], passed=c["passed"], detail=c.get("detail", ""))
        for c in gate_data.get("checks", [])
    ]
    gate_eval = GateEvaluation(
        level=ClaimLevel(gate_data["level"]),
        admissible=gate_data["admissible"],
        checks=checks,
    )
    created_at_raw = d.get("created_at")
    created_at = datetime.fromisoformat(created_at_raw) if created_at_raw else None

    return Claim(
        claim_id=d["claim_id"],
        user_id=d["user_id"],
        domain_id=d["domain_id"],
        level=ClaimLevel(d["level"]),
        dominant_divergence_type=d.get("dominant_divergence_type"),
        gate_evaluation=gate_eval,
        created_at=created_at,
        is_dual_structured=d.get("is_dual_structured", False),
        dual_structure_components=None,
    )


class ClaimsEngineAdapter(ClaimsStoreProvider):
    """
    Production adapter implementing a durable append-only claims store.
    Note: Claims persistence is single-writer. Concurrent writes require an external lock.
    """

    def __init__(self, db_path: str = "claims_store.jsonl"):
        self.db_path = db_path
        self._claims: Dict[str, List[Any]] = {}   # claim_id → [version_0, version_1, ...]
        self._claim_statuses: Dict[str, str] = {}

        if os.path.exists(self.db_path):
            self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        with open(self.db_path, "r", encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        for i, line in enumerate(lines):
            is_last = (i == len(lines) - 1)
            try:
                record = json.loads(line)
                self._apply(record, i + 1)
            except json.JSONDecodeError as e:
                if is_last:
                    logger.warning("Skipping truncated final record in %s: %s", self.db_path, e)
                else:
                    logger.error(
                        "STORE CORRUPTION: malformed interior record at line %d in %s: %s",
                        i + 1, self.db_path, e
                    )
                    raise

    def _apply(self, record: Dict, line_num: int = 0) -> None:
        action = record.get("action")
        claim_id = record.get("claim_id")
        if action == "append":
            payload = record.get("payload")
            if payload:
                try:
                    claim_obj = _dict_to_claim(payload)
                    if claim_id not in self._claims:
                        self._claims[claim_id] = []
                    self._claims[claim_id].append(claim_obj)
                except Exception as e:
                    logger.warning("Could not reconstruct claim at record (line ~%d): %s", line_num, e)
        elif action == "update_status":
            self._claim_statuses[claim_id] = record.get("status")
        elif action == "batch_update_status":
            for item in record.get("updates", []):
                cid = item.get("claim_id")
                st = item.get("status")
                if cid and st:
                    self._claim_statuses[cid] = st

    def _atomic_append(self, record: Dict) -> None:
        """Write a single record atomically (tmp-file + os.replace)."""
        line = json.dumps({"schema_version": SCHEMA_VERSION, **record}) + "\n"
        tmp = self.db_path + ".tmp"
        # Read existing content
        existing = ""
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding="utf-8") as f:
                existing = f.read()
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(existing)
            f.write(line)
        os.replace(tmp, self.db_path)

    # ------------------------------------------------------------------
    # ClaimsStoreProvider interface
    # ------------------------------------------------------------------

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        versions = self._claims.get(claim_id)
        if not versions:
            return None
        return copy.deepcopy(versions[-1])

    def get_claim_status(self, claim_id: str) -> Optional[str]:
        """
        Returns the status for a known claim, or None for unknown claim_ids.
        NEVER returns "SURFACED" for claims that don't exist (R2-F20.7).
        """
        if claim_id not in self._claims:
            return None
        return self._claim_statuses.get(claim_id, "SURFACED")

    def append_claim_version(self, claim_id: str, claim_obj: Any) -> None:
        """Appends a new immutable version of a claim (no in-place mutation)."""
        if claim_id not in self._claims:
            self._claims[claim_id] = []
        frozen = copy.deepcopy(claim_obj)
        self._claims[claim_id].append(frozen)
        self._atomic_append({
            "action": "append",
            "claim_id": claim_id,
            "payload": _claim_to_dict(frozen),
        })

    def delete_claim(self, claim_id: str):
        raise PermissionError("In-place deletion is rejected. Append a new version instead.")

    def mutate_claim(self, claim_id: str):
        raise PermissionError("In-place mutation is rejected. Append a new version instead.")

    def update_claim_status(self, claim_id: str, new_status: str) -> None:
        """
        Raises KeyError when claim_id is not found (R2-F20.7).
        Silent no-ops are not acceptable.
        """
        if claim_id not in self._claims:
            raise KeyError(f"Cannot update status: claim '{claim_id}' not found in store.")
        self._claim_statuses[claim_id] = new_status
        self._atomic_append({
            "action": "update_status",
            "claim_id": claim_id,
            "status": new_status,
        })

    def apply_status_updates_batch(
        self, updates: List[Any]  # List[Tuple[str, str]]
    ) -> None:
        """
        Updates multiple claim statuses atomically (all succeed or none).
        Validates all claim_ids first; if any are missing, raises KeyError
        before mutating state (fail-validate-before-write).
        """
        missing = [cid for cid, _ in updates if cid not in self._claims]
        if missing:
            raise KeyError(f"Cannot batch-update: claim IDs not found: {missing}")

        for claim_id, new_status in updates:
            self._claim_statuses[claim_id] = new_status

        self._atomic_append({
            "action": "batch_update_status",
            "updates": [{"claim_id": cid, "status": s} for cid, s in updates],
        })

    def iter_claims(self):
        for versions in self._claims.values():
            yield copy.deepcopy(versions[-1])
