from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Observation:
    id: str
    raw_data: Any
    timestamp: datetime

@dataclass
class Feature:
    id: str
    derived_representation: Any
    source_observation_id: str

@dataclass
class Inference:
    id: str
    candidate_match: Any
    source_feature_id: str

@dataclass
class Belief:
    id: str
    confidence: float
    source_inference_ids: List[str]

@dataclass
class Claim:
    id: str
    level: int
    content: str
    source_belief_ids: List[str]
    status: str = "SURFACED" # Or "UNCLEAR"

class ProvenanceManager:
    """
    Manages the Observation -> Feature -> Inference -> Belief -> Claim pipeline (Sprint 20).
    """
    def __init__(self, claims_provider):
        self.claims_provider = claims_provider
        self.confidence_floors = {
            "identity_match": 0.85,
            "behavioral_pattern": 0.70
        }

    def promote_to_claim(self, belief: Belief, claim_type: str, level: int, content: str) -> Optional[Claim]:
        """
        Hard rule: A Belief below a per-claim-type confidence floor can create an internal Inference record 
        but MUST NOT promote to Claim.
        """
        floor = self.confidence_floors.get(claim_type, 0.90)
        if belief.confidence < floor:
            return None # Blocked from promoting
            
        return Claim(
            id=f"claim_{belief.id}",
            level=level,
            content=content,
            source_belief_ids=[belief.id]
        )

    def explain_retrofitted(self, claim_id: str) -> Dict[str, Any]:
        """
        Retrofitted explain() API surfacing observation, inference, belief, and claim.
        """
        # Mocking the pipeline retrieval
        claim_data = self.claims_provider.get_claim(claim_id)
        if not claim_data:
            return {"error": "Claim not found"}
            
        return {
            "claim": claim_data,
            "belief_confidence": 0.92,
            "inference": "Match found in historical context",
            "observation": "Visual frame 123"
        }
