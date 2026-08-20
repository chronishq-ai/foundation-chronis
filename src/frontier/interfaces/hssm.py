from abc import ABC, abstractmethod
from typing import Dict, Any, List

class HSSMProvider(ABC):
    """
    Interface matching origin/palash/backbone-sprint3-4.
    """

    @abstractmethod
    def get_behavioral_state(self, user_id: str, time_range: tuple) -> Dict[str, Any]:
        pass

class MockHSSMProvider(HSSMProvider):
    def __init__(self, states=None):
        self._states = states or []

    def get_behavioral_state(self, user_id: str, time_range: tuple) -> Dict[str, Any]:
        return {"state": "mock_state", "confidence": 0.9}
