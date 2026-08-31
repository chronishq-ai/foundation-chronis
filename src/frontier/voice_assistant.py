"""
frontier/voice_assistant.py

Sprint 17 / R2-F17.2

Voice Assistant with closed-set intent router.

Key fixes:
  - Handlers return structured dicts (not stub strings)
  - Handlers call real retrieval_core / explainability_api if configured
  - If a subsystem is not injected, returns {"status": "not_configured", "error": True}
    (temporary dev state — Sprint 17-20 is not closed until real subsystems are wired)
  - process_query passes user_id through to all handlers
  - All 21 adversarial fallback cases still covered
"""

from typing import Any, Dict, Optional

from .interfaces.mirror import MirrorProvider
from .interfaces.wake_word import WakeWordProvider


class IntentRouter:
    """
    Closed-set intent router.
    Routes transcribed query to: visual_temporal, mirror_insight,
    explainability, or fallback.

    NO general-purpose LLM answering arbitrary questions.
    """

    def __init__(self):
        self.supported_intents = ["visual_temporal", "mirror_insight", "explainability"]

    def classify_intent(self, query: str) -> str:
        q = query.lower()
        if "have i been" in q or "when did i" in q or "do i remember" in q:
            return "visual_temporal"
        elif "insight" in q or "mirror insight" in q or "detect a pattern" in q:
            return "mirror_insight"
        elif "explain this to me" in q or "explain the claim" in q or "how do you know" in q or "why do you believe" in q:
            return "explainability"
        return "fallback"


class VoiceAssistant:
    def __init__(
        self,
        mirror: MirrorProvider,
        wake_word_provider: WakeWordProvider,
        retrieval_core=None,
        explainability_api=None,
    ):
        self.wake_word_provider = wake_word_provider
        self.intent_router = IntentRouter()
        self.mirror = mirror
        self.retrieval_core = retrieval_core
        self.explainability_api = explainability_api

    def set_mirror_aggressiveness(self, user_id: str, aggressiveness: str = "occasional"):
        """silent / occasional / proactive"""
        valid_settings = ["silent", "occasional", "proactive"]
        if aggressiveness not in valid_settings:
            raise ValueError(f"Invalid aggressiveness. Must be one of {valid_settings}")
        self.mirror.set_aggressiveness(user_id, aggressiveness)

    def process_query(self, user_id: str, query: str) -> Any:
        intent = self.intent_router.classify_intent(query)

        if intent == "visual_temporal":
            return self._handle_visual_temporal(user_id, query)
        elif intent == "mirror_insight":
            return self._handle_mirror_insight(user_id, query)
        elif intent == "explainability":
            return self._handle_explainability(user_id, query)
        else:
            return self._handle_fallback()

    def _handle_visual_temporal(self, user_id: str, query: str) -> Any:
        """
        Calls the Central Retrieval Core for real visual/temporal retrieval.
        Returns structured dict, not a stub string.
        """
        if self.retrieval_core is None:
            return {
                "status": "retrieval_core_not_configured",
                "error": True,
                "intent": "visual_temporal",
            }
        try:
            result = self.retrieval_core.retrieve(
                query=query,
                query_type="past",
                requesting_interface="voice_assistant",
                user_id=user_id,
                consent_context={},   # policy enforced by retrieval_core internally
            )
            return {
                "status": "ok",
                "intent": "visual_temporal",
                "evidence_package": result,
            }
        except PermissionError as e:
            return {"status": "access_denied", "error": True, "detail": str(e)}

    def _handle_mirror_insight(self, user_id: str, query: str) -> Any:
        """
        Surfaces a Mirror-style grounded insight.
        Returns structured dict.
        """
        # get_insight() is now part of the MirrorProvider interface contract.
        # No hasattr guard needed — all implementations must provide it.
        insight = self.mirror.get_insight(user_id, query)
        if insight is None:
            return {
                "status": "no_insight_available",
                "intent": "mirror_insight",
                "message": "No relevant mirror insight found for this query.",
            }
        return {
            "status": "ok",
            "intent": "mirror_insight",
            "insight": insight,
        }


    def _extract_claim_id(self, query: str) -> Optional[str]:
        import re
        m = re.search(r"claim_id=([^\s]+)", query, re.IGNORECASE)
        if m:
            return m.group(1)
        # Tighten regex to avoid matching generic phrases like "claim_form_was_filed"
        # Require it to end with digits, or have multiple underscores/hyphens.
        # (Testing often uses claim_test_123 or claim_c1)
        m = re.search(r"\b(claim_[a-zA-Z0-9_]*[0-9]+[a-zA-Z0-9_]*)\b", query, re.IGNORECASE)
        if m:
            return m.group(1)
        return None

    def _handle_explainability(self, user_id: str, query: str) -> Any:
        """
        Calls the canonical ExplainabilityAPI.
        Returns structured dict, not a stub string.
        """
        if self.explainability_api is None:
            return {
                "status": "explainability_not_configured",
                "error": True,
                "intent": "explainability",
            }
        claim_id = self._extract_claim_id(query)
        if claim_id is None:
            # Bug 4 fix: gracefully fall back to returning recent claim IDs
            # rather than returning an error for a valid intent.
            available_claims = []
            if getattr(self.explainability_api, "claims_store", None):
                for c in self.explainability_api.claims_store.iter_claims():
                    if getattr(c, "user_id", None) == user_id:
                        available_claims.append(getattr(c, "claim_id", str(c)))
            
            return {
                "status": "no_claim_id_in_query",
                "error": False,  # It's a valid intent, just needs disambiguation
                "intent": "explainability",
                "detail": "Please specify which claim you want explained.",
                "available_claims": available_claims[-5:],  # Return up to 5 most recent
            }
        return self.explainability_api.explain(claim_id, user_id)

    def _handle_fallback(self) -> dict:
        """
        Explicit fallback for unsupported/out-of-scope queries.

        Returns a structured dict (not a bare string) so that all callers of
        process_query() can unconditionally do result.get("status") / result["intent"]
        without type-checking the return value.
        """
        return {
            "status": "fallback",
            "intent": "fallback",
            "message": (
                "I'm sorry, I cannot help with that general question. "
                "I can only assist with your personal memories, insights, and explanations."
            ),
        }
