"""
frontier/retrieval.py

Sprint 17 / R2-F17.1

Visual and Temporal Retrieval API.

Key fixes:
  - get_context() performs real user-scoped retrieval from the visual index
    filtered by time_range (R2-F17.1). All stub/placeholder returns removed.
  - Only entries belonging to the requesting user are returned
  - Returns [] (not a placeholder) when no entries match
  - Results are sorted by timestamp ascending
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .visual_memory import VisualMemoryIndex, SelfHostedCLIPEncoder
from .interfaces.layer0 import Layer0Storage

logger = logging.getLogger(__name__)


class RetrievalAPI:
    """
    Visual and Temporal Retrieval API (Sprint 17 / R2-F17.1).
    """

    def __init__(
        self,
        visual_indexes: Dict[str, VisualMemoryIndex],
        layer0: Layer0Storage,
        encoder: SelfHostedCLIPEncoder,
    ):
        self.visual_indexes = visual_indexes
        self.layer0 = layer0
        self.encoder = encoder

    def _calculate_confidence(
        self, ann_distance: float, gps_match: bool, salience: str
    ) -> float:
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

    def search_visual(
        self,
        user_id: str,
        query_text: str,
        current_gps: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        "Have I been on this road before?" style queries.
        Returns only entries belonging to user_id.
        """
        if user_id not in self.visual_indexes:
            return []

        index = self.visual_indexes[user_id]
        query_embedding = self.encoder.encode(query_text)

        raw_results = index.retrieve(query_embedding, k=10)

        ranked_results = []
        for res in raw_results:
            gps_match = bool(current_gps and res.get("gps_if_present"))
            confidence = self._calculate_confidence(
                res.get("ann_distance", 1.0),
                gps_match,
                res.get("salience_level", "L0"),
            )
            ranked_results.append({
                "timestamp": res.get("timestamp_ntp"),
                "confidence": confidence,
                "canonical_record_pointer": res.get("canonical_record_pointer"),
                "owner_user_id": user_id,   # enforce ownership tag on output
            })

        ranked_results.sort(key=lambda x: x["confidence"], reverse=True)
        return ranked_results

    def get_context(
        self,
        user_id: str,
        time_range: tuple,
        query_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Temporal retrieval API shared by Voice Assistant, Action Button, and
        Timeline Dial.

        Returns real user-scoped entries whose timestamp_ntp falls within
        [time_range[0], time_range[1]].  Returns [] when nothing matches.
        All placeholder returns have been removed (R2-F17.1).

        query_type: 'past' or 'future'
        """
        if query_type not in ("past", "future"):
            raise ValueError("query_type must be 'past' or 'future'")

        if user_id not in self.visual_indexes:
            return []

        start, end = time_range
        if start > end:
            raise ValueError(
                f"Invalid time_range: start ({start}) must be <= end ({end})"
            )
        index = self.visual_indexes[user_id]

        matching = []
        for entry in index.entries:
            # Each entry must belong to this user (ownership enforced at store time)
            if entry.get("user_id") and entry["user_id"] != user_id:
                logger.error(
                    "SECURITY: visual index for user '%s' contains entry owned by '%s'. "
                    "This should never happen. Entry skipped.",
                    user_id, entry["user_id"]
                )
                continue   # cross-user entry — should never happen; skip defensively

            ts = entry.get("timestamp_ntp")
            if ts is None:
                continue
            if start <= ts <= end:
                matching.append({
                    "canonical_record_pointer": entry.get("canonical_record_pointer"),
                    "timestamp": ts,
                    "type": query_type,
                    "salience_level": entry.get("salience_level"),
                    "owner_user_id": user_id,
                })

        # Sort chronologically
        matching.sort(key=lambda x: x["timestamp"])
        return matching

    def action_button_bookmark(
        self,
        user_id: str,
        timestamp: datetime,
        context: str,
    ) -> str:
        """
        Action Button bookmark path.
        Writes a lightweight canonical-record annotation.  Does NOT duplicate
        raw frame data.
        """
        annotation = {
            "type": "action_button_bookmark",
            "timestamp": timestamp,
            "context": context,
        }
        pointer = self.layer0.write_record(user_id, annotation)
        return pointer
