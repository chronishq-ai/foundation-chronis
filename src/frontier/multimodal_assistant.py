"""
frontier/multimodal_assistant.py

Sprint 18 / R2-F17.3

Multimodal Assistant.

Key fixes:
  - Consent via policy_engine (not a hard-coded caller-supplied integer)
  - live_context handler uses real visual similarity, not a fabricated string
  - No positive match without real similarity threshold
  - Fabricated identity labels cannot appear
"""

from typing import Any, Dict, Optional

from .interfaces.llm import LLMProvider
from .interfaces.policy import PolicyEngine
from .interfaces.encoder import VisualEncoderMetadata, DEFAULT_ENCODER_METADATA

# Minimum cosine-similarity for a confident visual match
VISUAL_MATCH_THRESHOLD = 0.75


class MultimodalAssistant:
    """
    Multimodal Assistant (Sprint 18 / R2-F17.3).
    Extends the intent router with `general_knowledge` and explicit
    mixed-query composition.
    """

    def __init__(
        self,
        retrieval_core,
        llm_provider: LLMProvider,
        policy_engine: Optional[PolicyEngine] = None,
        visual_index_provider=None,
        encoder_metadata: Optional[VisualEncoderMetadata] = None,
    ):
        self.retrieval_core = retrieval_core
        self.llm_provider = llm_provider
        self.policy_engine = policy_engine
        self.visual_index_provider = visual_index_provider   # callable: user_id → VisualMemoryIndex
        self.encoder_metadata = encoder_metadata or DEFAULT_ENCODER_METADATA

    def _route_general_knowledge(self, query: str) -> str:
        """
        The ONLY path permitted to call the general-purpose LLM without going
        through CentralRetrievalCore.
        """
        response = self.llm_provider.generate(query)
        return f"[GENERAL_KNOWLEDGE] {response}"

    def resolve_query(
        self,
        user_id: str,
        query: str,
        intent: str,
        live_camera_embedding=None,
    ) -> Any:
        """
        Handles explicit composition of personal and general-knowledge queries.
        Never silently blends ungrounded answers.
        """
        if intent == "general_knowledge":
            return self._route_general_knowledge(query)

        elif intent == "mixed_query":
            # Consent via policy_engine — NOT hard-coded tier
            if self.policy_engine and not self.policy_engine.check_access(
                user_id, "personal_retrieval", required_tier=2
            ):
                return "[ERROR] Access denied by policy engine"

            personal_evidence = self.retrieval_core.retrieve(
                query=query,
                query_type="past",
                requesting_interface="multimodal_assistant",
                user_id=user_id,
                consent_context={},   # enforced internally by retrieval_core
            )
            general_answer = self._route_general_knowledge(query)
            return (
                f"[PERSONAL_EVIDENCE] Found "
                f"{len(personal_evidence.get('evidence_items', []))} items. "
                f"{general_answer}"
            )

        elif intent == "live_context":
            if live_camera_embedding is None:
                return "[ERROR] Live context intent requires camera embedding."
            return self._handle_live_context(user_id, live_camera_embedding)

        return "[FALLBACK] Unrecognized intent."

    def _handle_live_context(self, user_id: str, live_camera_embedding) -> Dict:
        """
        Real live-context handler:
          1. Encode live camera frame (embedding provided by caller)
          2. Search user-scoped visual index
          3. Return match only if similarity >= VISUAL_MATCH_THRESHOLD
          4. Otherwise return {"status": "no_confident_match"}

        Fabricated identity labels are prohibited; all matches must come from
        the real similarity search against the user's personal visual index.
        """
        if self.visual_index_provider is None:
            return {"status": "visual_index_not_configured", "error": True}

        visual_index = self.visual_index_provider(user_id)
        if visual_index is None:
            return {"status": "no_visual_index_for_user"}

        results = visual_index.retrieve(live_camera_embedding, k=1)
        if not results:
            return {"status": "no_confident_match", "reason": "no entries in index"}

        best = results[0]
        # ann_distance is L2; convert to rough similarity (lower distance = more similar)
        distance = best.get("ann_distance", 1.0)
        similarity = max(0.0, 1.0 - distance)

        if similarity < VISUAL_MATCH_THRESHOLD:
            return {
                "status": "no_confident_match",
                "similarity": round(similarity, 4),
                "threshold": VISUAL_MATCH_THRESHOLD,
            }

        return {
            "status": "match_found",
            "similarity": round(similarity, 4),
            "canonical_record_pointer": best.get("canonical_record_pointer"),
            "source_class": "personal_visual_memory",
            "encoder_model_id": self.encoder_metadata.model_id,
            "encoder_version": self.encoder_metadata.version,
            "similarity_metric": self.encoder_metadata.similarity_metric,
            "threshold_used": VISUAL_MATCH_THRESHOLD,
        }
