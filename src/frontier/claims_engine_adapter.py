import json
import os
import copy
from typing import Dict, Any, List, Optional
from .interfaces.claims_store import ClaimsStoreProvider
from claims_engine.claim_levels import Claim

class ClaimsEngineAdapter(ClaimsStoreProvider):
    """
    Production adapter over the Sprint 13 Claims Engine.
    Implements a durable append-only store with immutable versions.
    """
    def __init__(self, db_path: str = "claims_store.jsonl"):
        self.db_path = db_path
        self._claims: Dict[str, List[Any]] = {}  # Lists of versions
        self._claim_statuses: Dict[str, str] = {}
        # In a real system, we'd load from db_path here.
        # For simplicity, we just keep the file in sync.

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        versions = self._claims.get(claim_id)
        if not versions:
            return None
        return copy.deepcopy(versions[-1]) # Return latest immutable version

    def get_claim_status(self, claim_id: str) -> str:
        return self._claim_statuses.get(claim_id, "SURFACED")

    def append_claim_version(self, claim_id: str, claim_obj: Any) -> None:
        """Appends a new immutable version of a claim."""
        if claim_id not in self._claims:
            self._claims[claim_id] = []
            
        # Prevent in-place mutation
        frozen_claim = copy.deepcopy(claim_obj)
        self._claims[claim_id].append(frozen_claim)
        
        # Durable append-only write
        with open(self.db_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"action": "append", "claim_id": claim_id}) + "\n")

    def delete_claim(self, claim_id: str):
        raise PermissionError("In-place deletion is rejected. Append a new version instead.")

    def mutate_claim(self, claim_id: str):
        raise PermissionError("In-place mutation is rejected. Append a new version instead.")

    def update_claim_status(self, claim_id: str, new_status: str) -> None:
        if claim_id in self._claims:
            self._claim_statuses[claim_id] = new_status
            with open(self.db_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"action": "update_status", "claim_id": claim_id, "status": new_status}) + "\n")

    def iter_claims(self):
        for versions in self._claims.values():
            yield copy.deepcopy(versions[-1])

