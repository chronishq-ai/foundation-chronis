from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from .provenance_pipeline import Belief, ProvenanceRecord
from .interfaces.claims_store import ClaimsStoreProvider

@dataclass
class ConflictRecord:
    id: str
    belief_id_1: str
    belief_id_2: str
    user_id: str = ""
    resolved: bool = False

@dataclass(frozen=True)
class ConflictResolutionRecord:
    conflict_id: str
    user_id: str
    new_belief: Belief
    timestamp: datetime

class ConflictManager:
    """
    Manages Memory Conflict Resolution (Sprint 20).
    """
    def __init__(self, claims_store: ClaimsStoreProvider):
        self.conflicts = []
        self.claims_store = claims_store

    def resolve_contradiction(self, belief_1: Belief, belief_2: Belief) -> ConflictRecord:
        """
        When two Beliefs conflict, DO NOT delete either. Retain both.
        Create a ConflictRecord linking both.
        """
        conflict = ConflictRecord(
            id=f"conflict_{belief_1.id}_{belief_2.id}",
            belief_id_1=belief_1.id,
            belief_id_2=belief_2.id,
            user_id=getattr(belief_1, "user_id", "")
        )
        self.conflicts.append(conflict)
        return conflict

    def downgrade_dependent_claims(self, conflict: ConflictRecord, provenance_records: List[ProvenanceRecord]) -> None:
        """
        Any Claim relying on conflicting Beliefs becomes UNCLEAR.
        We check the provenance records to find which claims rely on the conflicting beliefs,
        then we update their status in the claims store.
        """
        for pr in provenance_records:
            if conflict.belief_id_1 in pr.source_belief_ids or conflict.belief_id_2 in pr.source_belief_ids:
                # Update status of the claim to UNCLEAR
                self.claims_store.update_claim_status(pr.claim_id, "UNCLEAR")
                pr.status = "UNCLEAR"

    def apply_teach_correction(self, user_id: str, conflict_id: str, new_belief: Belief) -> ConflictResolutionRecord:
        """
        Teach Chronis on a conflicted episode creates a NEW high-confidence Belief.
        Never deletes superseded Beliefs.
        """
        for conflict in self.conflicts:
            if conflict.id == conflict_id and conflict.user_id == user_id:
                conflict.resolved = True
                
        # Produce immutable ConflictResolutionRecord
        record = ConflictResolutionRecord(
            conflict_id=conflict_id,
            user_id=user_id,
            new_belief=new_belief,
            timestamp=datetime.now()
        )
        return record
            
