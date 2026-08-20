from typing import Dict, Any, List
from claims_engine.grounded_generation import contains_clinical_terminology
from .interfaces.mirror import MirrorProvider
from .interfaces.layer0 import Layer0Storage

class ExplainabilityAPI:
    def __init__(self, layer0: Layer0Storage, mirror: MirrorProvider):
        self.layer0 = layer0
        self.mirror = mirror

    def explain(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        explain() API.
        """
        if not claim_data:
            return {"error": "Claim not found"}
            
        citations = claim_data.get("citation_chain", [])
        
        for cite in citations:
            text = cite.get("sentence_text", "")
            if contains_clinical_terminology(text):
                return {
                    "error": "Content blocked by clinical terminology filter. Routed to human review.",
                    "routed_to_review": True
                }
                
        return {
            "claim_id": claim_data.get("claim_id"),
            "confidence": claim_data.get("confidence"),
            "evidence_list": citations
        }

    def teach_chronis(self, user_id: str, claim_id: str, correction_data: Dict[str, Any]) -> str:
        """
        Correction endpoint.
        """
        correction = {
            "type": "teach_chronis_correction",
            "target_claim_id": claim_id,
            "correction_data": correction_data
        }
        
        pointer = self.layer0.write_record(user_id, correction)
        self.mirror.process_teach_correction(user_id, correction)
        
        return pointer
