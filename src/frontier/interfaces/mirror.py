from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class MirrorProvider(ABC):
    """
    Thin interface for The Mirror (Sprint 12).
    """

    @abstractmethod
    def set_aggressiveness(self, user_id: str, setting: str) -> None:
        """
        Sprint 17: silent / occasional / proactive (default: occasional).
        """
        pass

    @abstractmethod
    def process_teach_correction(self, user_id: str, correction: Dict[str, Any]) -> None:
        """
        Sprint 17: Feed correction into the adaptive-threshold mechanism.
        """
        pass

    @abstractmethod
    def get_insight(self, user_id: str, query: str) -> Optional[Dict[str, Any]]:
        """
        Return a grounded Mirror-style insight for the given query and user,
        or None if no relevant insight is available.

        The returned dict must include at least:
          {"insight_text": str, "confidence": float}
        Returns None (not raises) when nothing relevant is found.
        """
        pass


class MockMirrorProvider(MirrorProvider):
    """Deterministic mock for testing."""
    def __init__(self):
        self.aggressiveness_settings = {}
        self.corrections = []
        self._insights: Dict[str, Dict[str, Any]] = {}  # user_id -> insight

    def set_aggressiveness(self, user_id: str, setting: str) -> None:
        self.aggressiveness_settings[user_id] = setting

    def process_teach_correction(self, user_id: str, correction: Dict[str, Any]) -> None:
        self.corrections.append({"user_id": user_id, "correction": correction})

    def get_insight(self, user_id: str, query: str) -> Optional[Dict[str, Any]]:
        """Returns a canned insight if one has been pre-planted via set_insight(), else None."""
        return self._insights.get(user_id)

    def set_insight(self, user_id: str, insight: Dict[str, Any]) -> None:
        """Test helper: pre-plant an insight for a user."""
        self._insights[user_id] = insight

