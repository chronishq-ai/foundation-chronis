from typing import Dict, Any, List
from .interfaces.claims import ClaimsProvider
from .interfaces.mirror import MirrorProvider
from .interfaces.layer0 import Layer0Storage

class ExplainabilityAPI:
    def __init__(self, claims_provider: ClaimsProvider, layer0: Layer0Storage, mirror: MirrorProvider):
        self.claims = claims_provider
        self.layer0 = layer0
        self.mirror = mirror

    def explain(self, claim_id: str) -> Dict[str, Any]:
        """
        explain(claim_id) API.
        Returns the existing citation chain restructured as a short, ranked evidence list.
        """
        claim = self.claims.get_claim(claim_id)
        if not claim:
            return {"error": "Claim not found"}
            
        # Restructure citation chain
        citations = claim.get("citation_chain", [])
        
        # Apply clinical terminology filter
        for cite in citations:
            text = cite.get("text", "")
            if not self.claims.filter_clinical_terminology(text):
                return {
                    "error": "Content blocked by clinical terminology filter. Routed to human review.",
                    "routed_to_review": True
                }
                
        return {
            "claim_id": claim_id,
            "confidence": claim.get("confidence"),
            "evidence_list": citations
        }

    def teach_chronis(self, user_id: str, claim_id: str, correction_data: Dict[str, Any]) -> str:
        """
        Correction endpoint.
        Logs to canonical record (counter-annotation, never deletes original evidence).
        Feeds back into adaptive-threshold mechanism (Mirror).
        """
        correction = {
            "type": "teach_chronis_correction",
            "target_claim_id": claim_id,
            "correction_data": correction_data
        }
        
        # Log to Layer 0 as counter-annotation (G2 compliant, no deletion)
        pointer = self.layer0.write_record(user_id, correction)
        
        # Feed back to Mirror's adaptive threshold
        self.mirror.process_teach_correction(user_id, correction)
        
        return pointer
