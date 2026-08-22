from typing import Dict, Any, List
from .interfaces.mirror import MirrorProvider

from .interfaces.wake_word import WakeWordProvider

class IntentRouter:
    """
    Closed-set intent router.
    Routes transcribed query to: visual_temporal, mirror_insight, explainability, or fallback.
    NO general-purpose LLM answering arbitrary questions.
    """
    def __init__(self):
        self.supported_intents = ["visual_temporal", "mirror_insight", "explainability"]

    def classify_intent(self, query: str) -> str:
        query_lower = query.lower()
        if "have i been" in query_lower or "when did i" in query_lower:
            return "visual_temporal"
        elif "insight" in query_lower or "mirror" in query_lower:
            return "mirror_insight"
        elif "why" in query_lower or "explain" in query_lower:
            return "explainability"
        
        return "fallback"

class VoiceAssistant:
    def __init__(self, mirror: MirrorProvider, wake_word_provider: WakeWordProvider):
        self.wake_word_provider = wake_word_provider
        self.intent_router = IntentRouter()
        self.mirror = mirror

    def set_mirror_aggressiveness(self, user_id: str, aggressiveness: str = "occasional"):
        """
        silent / occasional / proactive
        """
        valid_settings = ["silent", "occasional", "proactive"]
        if aggressiveness not in valid_settings:
            raise ValueError(f"Invalid aggressiveness. Must be one of {valid_settings}")
        self.mirror.set_aggressiveness(user_id, aggressiveness)

    def process_query(self, user_id: str, query: str) -> str:
        intent = self.intent_router.classify_intent(query)
        
        if intent == "visual_temporal":
            return self._handle_visual_temporal(query)
        elif intent == "mirror_insight":
            return self._handle_mirror_insight(query)
        elif intent == "explainability":
            return self._handle_explainability(query)
        else:
            return self._handle_fallback()

    def _handle_visual_temporal(self, query: str) -> str:
        return "Executing visual/temporal retrieval."

    def _handle_mirror_insight(self, query: str) -> str:
        return "Surfacing Mirror-style grounded insight."

    def _handle_explainability(self, query: str) -> str:
        return "Surfacing explainability."

    def _handle_fallback(self) -> str:
        """Explicit fallback for unsupported queries."""
        return "I'm sorry, I cannot help with that general question. I can only assist with your personal memories, insights, and explanations."
