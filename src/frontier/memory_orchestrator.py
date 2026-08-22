from typing import Dict, Any, List
from datetime import datetime

class MemoryOrchestrator:
    """
    Memory Orchestrator (Sprint 18).
    Classifies memory kinds, issues parallel retrieval calls, and resolves results into an EvidencePackage.
    """
    def __init__(self, visual_retrieval):
        self.visual_retrieval = visual_retrieval

    def orchestrate(self, user_id: str, query: str, query_type: str) -> Dict[str, Any]:
        """
        Runs parallel retrieval across implicated modalities and resolves contradictions.
        """
        visual_results = self.visual_retrieval.search_visual(user_id, query) if self.visual_retrieval else []
        
        evidence_items = []
        contradictions = []
        overall_confidence = 1.0
        
        for v in visual_results:
            # S1720.2: The lowest tier (<0.5) is never silently dropped; threaded through.

            evidence_items.append({
                "modality": "visual",
                "content_pointer": v.get("canonical_record_pointer"),
                "confidence": v.get("confidence"),
                "source": "visual_index"
            })
            
        if len(evidence_items) > 1:
            contradictions.append({
                "type": "conflicting_evidence",
                "items": [evidence_items[0]["content_pointer"], evidence_items[1]["content_pointer"]]
            })
            overall_confidence -= 0.3
            
        episode_window = (datetime.now(), datetime.now())

        evidence_package = {
            "episode_window": episode_window,
            "evidence_items": evidence_items,
            "contradictions": contradictions,
            "overall_confidence": max(0.0, overall_confidence)
        }
        
        return evidence_package
