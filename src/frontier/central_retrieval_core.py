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
        # Validate consent context
        if consent_context.get("consent_tier", 0) < 2:
            return {"error": "Insufficient consent tier for retrieval"}
            
        # Route through MemoryOrchestrator
        evidence_package = self.orchestrator.orchestrate(
            user_id=user_id,
            query=query,
            query_type=query_type
        )
        
        return evidence_package
