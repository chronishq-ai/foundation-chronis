from typing import Dict, Any, List

class CentralRetrievalCore:
    """
    Central Retrieval Core (Sprint 18).
    A routing and composition layer, owning zero data of its own.
    Sitting in front of all retrieval primitives.
    """
    def __init__(self, memory_orchestrator):
        self.orchestrator = memory_orchestrator

    def retrieve(self, query: str, query_type: str, requesting_interface: str, user_id: str, consent_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Contract: {query, query_type, requesting_interface, user_id, consent_context}
        Returns a single ranked EvidencePackage.
        """
        if consent_context.get("consent_tier", 0) < 2:
            return {"error": "Insufficient consent tier for retrieval"}
            
        evidence_package = self.orchestrator.orchestrate(
            user_id=user_id,
            query=query,
            query_type=query_type
        )
        
        # Merge, dedup, and rank logic
        evidence_items = evidence_package.get("evidence_items", [])
        
        unique_items = {}
        for item in evidence_items:
            ptr = item.get("content_pointer")
            if ptr not in unique_items or item.get("confidence", 0) > unique_items[ptr].get("confidence", 0):
                unique_items[ptr] = item
                
        ranked_items = sorted(unique_items.values(), key=lambda x: x.get("confidence", 0), reverse=True)
        evidence_package["evidence_items"] = ranked_items
        
        return evidence_package
