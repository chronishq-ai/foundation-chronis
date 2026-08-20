from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

from .visual_memory import VisualMemoryIndex, SelfHostedCLIPEncoder
from .interfaces.layer0 import Layer0Storage

class RetrievalAPI:
    """
    Visual and Temporal Retrieval API (Sprint 17).
    """
    def __init__(self, visual_indexes: Dict[str, VisualMemoryIndex], layer0: Layer0Storage, encoder: SelfHostedCLIPEncoder):
        self.visual_indexes = visual_indexes
        self.layer0 = layer0
        self.encoder = encoder

    def _calculate_confidence(self, ann_distance: float, gps_match: bool, salience: str) -> float:
        """
        Calculates confidence combining ANN distance with corroborating signals.
        All thresholds must be configurable and logged to MLflow.
        """
        base_conf = max(0.0, 1.0 - ann_distance)
        
        if gps_match:
            base_conf += 0.15
            
        salience_weights = {"L2": 0.05, "L3": 0.10, "L4": 0.15, "L5": 0.20}
        base_conf += salience_weights.get(salience, 0.0)
        
        return min(1.0, base_conf)

    def search_visual(self, user_id: str, query_text: str, current_gps: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        "Have I been on this road before?" style queries.
        """
        if user_id not in self.visual_indexes:
            return []
            
        index = self.visual_indexes[user_id]
        query_embedding = self.encoder.encode(query_text)
        
        raw_results = index.retrieve(query_embedding, k=10)
        
        ranked_results = []
        for res in raw_results:
            gps_match = False
            if current_gps and res.get("gps_if_present"):
                gps_match = True
                
            confidence = self._calculate_confidence(
                res.get("ann_distance", 1.0), 
                gps_match, 
                res.get("salience_level", "L0")
            )
            
            ranked_results.append({
                "timestamp": res.get("timestamp_ntp"),
                "confidence": confidence,
                "canonical_record_pointer": res.get("canonical_record_pointer")
            })
            
        ranked_results.sort(key=lambda x: x["confidence"], reverse=True)
        return ranked_results

    def get_context(self, user_id: str, time_range: tuple, query_type: str) -> List[Dict[str, Any]]:
        """
        Temporal retrieval API shared by Voice Assistant, Action Button, and Timeline Dial.
        query_type: 'past' or 'future'
        """
        if query_type not in ["past", "future"]:
            raise ValueError("query_type must be 'past' or 'future'")
            
        return [{"event": "mock_event", "time": time_range[0], "type": query_type}]

    def action_button_bookmark(self, user_id: str, timestamp: datetime, context: str) -> str:
        """
        Action Button bookmark path.
        Writes a lightweight canonical-record annotation. Does NOT duplicate raw frame data.
        """
        annotation = {
            "type": "action_button_bookmark",
            "timestamp": timestamp,
            "context": context
        }
        # Write to Layer 0
        pointer = self.layer0.write_record(user_id, annotation)
        return pointer
