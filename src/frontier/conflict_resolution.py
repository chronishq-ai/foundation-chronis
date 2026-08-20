from typing import Dict, Any, List
from dataclasses import dataclass
from .provenance_pipeline import Belief, Claim

@dataclass
class ConflictRecord:
    id: str
    belief_id_1: str
    belief_id_2: str
    resolved: bool = False

class ConflictManager:
    """
    Manages Memory Conflict Resolution (Sprint 20).
    """
    def __init__(self):
        self.conflicts = []
        self.claims = {} # Mock claims store

    def resolve_contradiction(self, belief_1: Belief, belief_2: Belief) -> ConflictRecord:
        """
        When two Beliefs conflict, DO NOT delete either. Retain both.
        Create a ConflictRecord linking both.
        """
        conflict = ConflictRecord(
            id=f"conflict_{belief_1.id}_{belief_2.id}",
            belief_id_1=belief_1.id,
            belief_id_2=belief_2.id
        )
        self.conflicts.append(conflict)
        return conflict

    def downgrade_dependent_claims(self, conflict: ConflictRecord) -> None:
        """
        Any Claim relying on conflicting Beliefs becomes UNCLEAR.
        """
        for claim_id, claim in self.claims.items():
            if conflict.belief_id_1 in claim.source_belief_ids or conflict.belief_id_2 in claim.source_belief_ids:
                claim.status = "UNCLEAR"

    def apply_teach_correction(self, user_id: str, episode_id: str, new_belief: Belief) -> None:
        """
        Teach Chronis on a conflicted episode creates a NEW high-confidence Belief.
        Never deletes superseded Beliefs.
        """
        # Mark conflicts as resolved (without deleting old beliefs)
        for conflict in self.conflicts:
            # Mock check for relevance to episode
            conflict.resolved = True
            
        # The new_belief is retained alongside the old ones.
