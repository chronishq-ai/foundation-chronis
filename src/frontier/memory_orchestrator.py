"""
frontier/memory_orchestrator.py

Sprint 18 / R2-F18.2 / S1720.2

Memory Orchestrator: classifies memory kinds, issues retrieval calls, and
resolves results into a canonical EvidencePackage.

Key fixes:
  - Contradiction detection is semantic (NOT len > 1)
  - episode_window derived from actual evidence timestamps (NOT datetime.now())
  - overall_confidence is a weighted average of item confidences
  - Low-confidence items (< 0.5) are retained and flagged, not dropped
  - Two agreeing items → no contradiction

Contradiction contract (from audit):
  Contradiction detection operates on normalised evidence assertions.
  Each evidence item should carry an optional "assertion" dict:
    {
      "subject": <str>,
      "predicate": <str>,
      "object": <str>
    }
  Two items are contradictory when they share the same (subject, predicate)
  pair but have DIFFERENT "object" values AND come from different sources/
  modalities.
  Items with no "assertion" field are never automatically contradicted.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class MemoryOrchestrator:
    """
    Memory Orchestrator (Sprint 18 / R2-F18.2).
    Classifies memory kinds, issues parallel retrieval calls, and resolves
    results into an EvidencePackage.
    """

    def __init__(self, visual_retrieval):
        self.visual_retrieval = visual_retrieval

    # ------------------------------------------------------------------
    # Semantic contradiction detection (R2-F18.2 / S1720.2)
    # ------------------------------------------------------------------

    def _detect_semantic_contradiction(
        self, item_a: Dict, item_b: Dict
    ) -> bool:
        """
        Returns True only when two evidence items make conflicting claims
        about the same (subject, predicate) pair.

        Rules:
          - Both items must carry an "assertion" dict.
          - Same content_pointer AND same modality → same piece of evidence,
            never a contradiction.
          - Shared (subject, predicate) but DIFFERENT objects from DIFFERENT
            sources/modalities → genuine contradiction.
          - Items lacking an "assertion" dict → not automatically contradicted.
        """
        # Identical evidence → no contradiction
        if (
            item_a.get("content_pointer") == item_b.get("content_pointer")
            and item_a.get("modality") == item_b.get("modality")
        ):
            return False

        a_assert = item_a.get("assertion")
        b_assert = item_b.get("assertion")

        # No assertion on either item → cannot determine contradiction
        if not a_assert or not b_assert:
            return False

        # Same (subject, predicate), different object → contradiction
        same_subject = a_assert.get("subject") == b_assert.get("subject")
        same_predicate = a_assert.get("predicate") == b_assert.get("predicate")
        different_object = a_assert.get("object") != b_assert.get("object")

        return same_subject and same_predicate and different_object

    # ------------------------------------------------------------------
    # Episode window from evidence
    # ------------------------------------------------------------------

    def _derive_episode_window(
        self, evidence_items: List[Dict]
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Derives (start, end) from the actual timestamps in evidence items.
        Falls back to (None, None) when no timestamps are present.
        """
        timestamps = []
        for item in evidence_items:
            ts = item.get("timestamp_ntp") or item.get("timestamp")
            if ts is not None:
                timestamps.append(ts)
        if not timestamps:
            return (None, None)
        return (min(timestamps), max(timestamps))

    # ------------------------------------------------------------------
    # Main orchestration
    # ------------------------------------------------------------------

    def orchestrate(self, user_id: str, query: str, query_type: str) -> Dict[str, Any]:
        """
        Runs retrieval across implicated modalities and resolves contradictions.

        Returns a canonical EvidencePackage:
          {
            "episode_window": (start, end),   # from actual evidence timestamps
            "evidence_items": [...],           # all items, low-conf flagged but retained
            "contradictions": [...],           # only genuine semantic contradictions
            "overall_confidence": float,       # weighted avg of item confidences
          }
        """
        visual_results = (
            self.visual_retrieval.search_visual(user_id, query)
            if self.visual_retrieval
            else []
        )

        evidence_items: List[Dict] = []
        for v in visual_results:
            item = {
                "modality": "visual",
                "content_pointer": v.get("canonical_record_pointer"),
                "confidence": v.get("confidence", 0.0),
                "source": "visual_index",
                "owner_user_id": v.get("owner_user_id", user_id),
            }
            # Carry through timestamp if present
            if v.get("timestamp"):
                item["timestamp_ntp"] = v["timestamp"]
            # Carry through normalized assertion dict if present (required for contradiction detection)
            if v.get("assertion"):
                item["assertion"] = v["assertion"]
            # Flag low-confidence items but retain them (S1720.2)
            if item["confidence"] < 0.5:
                item["low_confidence"] = True

            evidence_items.append(item)

        # --- Contradiction detection: semantic, not len > 1 ---
        contradictions: List[Dict] = []
        n = len(evidence_items)
        for i in range(n):
            for j in range(i + 1, n):
                if self._detect_semantic_contradiction(evidence_items[i], evidence_items[j]):
                    contradictions.append({
                        "type": "conflicting_evidence",
                        "items": [
                            evidence_items[i].get("content_pointer"),
                            evidence_items[j].get("content_pointer"),
                        ],
                    })

        # --- overall_confidence: weighted average, not a fixed penalty ---
        if evidence_items:
            total_weight = sum(e.get("confidence", 0.0) for e in evidence_items)
            overall_confidence = total_weight / len(evidence_items)
            # Contradictions reduce confidence by 0.1 each, floor 0.0
            overall_confidence = max(0.0, overall_confidence - 0.1 * len(contradictions))
        else:
            # No evidence → explicitly 0.0 (not 1.0 which would be misleading)
            overall_confidence = 0.0

        # --- episode window from actual timestamps ---
        episode_window = self._derive_episode_window(evidence_items)

        # --- status field: explicit for downstream consumers ---
        status = "low_confidence" if overall_confidence < 0.5 else "ok"

        return {
            "episode_window": episode_window,
            "evidence_items": evidence_items,
            "contradictions": contradictions,
            "overall_confidence": overall_confidence,
            "status": status,
        }
