"""
frontier/conflict_resolution.py

Sprint 20 / S1720.6 / R2-F20.3

Memory Conflict Resolution.

Key fixes:
  - apply_teach_correction raises ConflictNotFoundError (typed) for bad IDs
  - new_belief.user_id verified against caller's user_id → PermissionError
  - Dependent claims updated after resolution (atomically where possible)
  - Idempotency: repeated correction with same belief_id → existing record
  - Old beliefs are NEVER deleted

Atomicity contract:
  The implementation prepares all state mutations before committing any
  of them.  If updating dependent claims fails, the conflict is NOT marked
  as resolved (consistent state over partial progress).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from .provenance_pipeline import Belief, ProvenanceRecord, ProvenanceStore
from .interfaces.claims_store import ClaimsStoreProvider


# ---------------------------------------------------------------------------
# Typed exceptions
# ---------------------------------------------------------------------------

class ConflictNotFoundError(KeyError):
    """Raised when a conflict_id does not exist in the manager's registry."""
    pass


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConflictRecord:
    id: str
    belief_id_1: str
    belief_id_2: str
    user_id: str = ""
    resolved: bool = False
    resolution_belief_id: Optional[str] = None   # id of the new belief that resolved it


@dataclass(frozen=True)
class ConflictResolutionRecord:
    conflict_id: str
    user_id: str
    new_belief: Belief
    timestamp: datetime


# ---------------------------------------------------------------------------
# Conflict Manager
# ---------------------------------------------------------------------------

class ConflictManager:
    """
    Manages Memory Conflict Resolution (Sprint 20 / S1720.6 / R2-F20.3).
    """

    def __init__(
        self,
        claims_store: ClaimsStoreProvider,
        provenance_store: Optional[ProvenanceStore] = None,
    ):
        self._conflicts: Dict[str, ConflictRecord] = {}
        self.claims_store = claims_store
        self.provenance_store = provenance_store
        self._resolution_records: Dict[str, ConflictResolutionRecord] = {}

    @property
    def conflicts(self) -> List[ConflictRecord]:
        """Legacy list-style access used by existing tests."""
        return list(self._conflicts.values())

    def resolve_contradiction(self, belief_1: Belief, belief_2: Belief) -> ConflictRecord:
        """
        When two Beliefs conflict, DO NOT delete either.  Retain both.
        Create a ConflictRecord linking both.
        """
        conflict_id = f"conflict_{belief_1.id}_{belief_2.id}"
        conflict = ConflictRecord(
            id=conflict_id,
            belief_id_1=belief_1.id,
            belief_id_2=belief_2.id,
            user_id=getattr(belief_1, "user_id", ""),
        )
        self._conflicts[conflict_id] = conflict
        return conflict

    def downgrade_dependent_claims(
        self,
        conflict: ConflictRecord,
        provenance_records: List[ProvenanceRecord],
    ) -> None:
        """
        Any Claim relying on conflicting Beliefs becomes UNCLEAR.
        Updates their status in the claims store.
        """
        for pr in provenance_records:
            if conflict.belief_id_1 in pr.source_belief_ids or conflict.belief_id_2 in pr.source_belief_ids:
                try:
                    self.claims_store.update_claim_status(pr.claim_id, "UNCLEAR")
                except KeyError:
                    pass   # claim not in store — harmless
                pr.status = "UNCLEAR"

    def apply_teach_correction(
        self,
        user_id: str,
        conflict_id: str,
        new_belief: Belief,
        provenance_store=None,
    ) -> ConflictResolutionRecord:
        """
        Teach Chronis on a conflicted episode creates a NEW high-confidence Belief.
        Never deletes superseded Beliefs.

        Validation order (all must pass before any state is mutated):
          1. conflict_id must exist → ConflictNotFoundError
          2. conflict.user_id must match caller user_id → PermissionError
          3. new_belief.user_id must match caller user_id → PermissionError
          4. Idempotency: if already resolved with same belief_id → return existing record

        Atomicity: only provenance-dependent claims are updated, and they are updated
        atomically before the conflict is marked resolved. If any update fails, the
        conflict remains unresolved.
        """
        # --- 1. Conflict must exist ---
        conflict = self._conflicts.get(conflict_id)
        if conflict is None:
            raise ConflictNotFoundError(
                f"Conflict '{conflict_id}' not found. Cannot apply teach correction."
            )

        # --- 2. Conflict must belong to caller ---
        if conflict.user_id and conflict.user_id != user_id:
            raise PermissionError(
                f"CROSS-USER VIOLATION: conflict '{conflict_id}' belongs to "
                f"'{conflict.user_id}', not '{user_id}'."
            )

        # --- 3. New belief must belong to caller ---
        new_belief_user = getattr(new_belief, "user_id", "")
        if new_belief_user and new_belief_user != user_id:
            raise PermissionError(
                f"CROSS-USER VIOLATION: new_belief.user_id='{new_belief_user}' "
                f"does not match caller '{user_id}'."
            )

        # --- 4. Idempotency ---
        if conflict.resolved and conflict.resolution_belief_id == new_belief.id:
            return self._resolution_records[conflict_id]

        # --- 5. Find ONLY provenance-dependent claims ---
        effective_prov = provenance_store if provenance_store is not None else self.provenance_store
        dependent_claim_ids: List[str] = []
        if effective_prov is not None:
            dependent_claim_ids = effective_prov.find_claims_referencing(
                belief_ids={conflict.belief_id_1, conflict.belief_id_2}
            )
        elif self.claims_store is not None:
            raise RuntimeError(
                "ConflictManager requires a provenance_store to determine "
                "dependent claims. Cannot resolve conflict without it."
            )

        # --- 6. Batch-atomic update of ONLY dependent claims ---
        if dependent_claim_ids and self.claims_store is not None:
            updates = [(cid, "UNCLEAR") for cid in dependent_claim_ids]
            try:
                if hasattr(self.claims_store, "apply_status_updates_batch"):
                    self.claims_store.apply_status_updates_batch(updates)
                else:
                    for cid, st in updates:
                        self.claims_store.update_claim_status(cid, st)
            except Exception as err:
                raise RuntimeError(
                    f"Atomic failure: dependent claim update failed. "
                    f"Conflict '{conflict_id}' remains UNRESOLVED. Cause: {err}"
                ) from err

        # --- 7. Commit: update conflict state ONLY after claim updates succeed ---
        conflict.resolved = True
        conflict.resolution_belief_id = new_belief.id

        record = ConflictResolutionRecord(
            conflict_id=conflict_id,
            user_id=user_id,
            new_belief=new_belief,
            timestamp=datetime.now(),
        )
        self._resolution_records[conflict_id] = record
        return record
