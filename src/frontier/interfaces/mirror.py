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

class MockMirrorProvider(MirrorProvider):
    """Deterministic mock for testing."""
    def __init__(self):
        self.aggressiveness_settings = {}
        self.corrections = []

    def set_aggressiveness(self, user_id: str, setting: str) -> None:
        self.aggressiveness_settings[user_id] = setting

    def process_teach_correction(self, user_id: str, correction: Dict[str, Any]) -> None:
        self.corrections.append({"user_id": user_id, "correction": correction})
