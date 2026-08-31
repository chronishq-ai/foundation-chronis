from abc import ABC, abstractmethod
from typing import Dict, Any, List
from datetime import datetime

class CSEProvider(ABC):
    """
    Thin interface for CSE frames/chunks (Sprint 2).
    """

    @abstractmethod
    def get_frames(self, user_id: str, start_time: datetime, end_time: datetime, min_salience: str = "L0") -> List[Dict[str, Any]]:
        """
        Returns CSE frames with salience levels, GPS, NTP timestamps.
        """
        pass

class MockCSEProvider(CSEProvider):
    """Deterministic mock for testing."""
    def __init__(self, planted_frames=None):
        self._frames = planted_frames or []

    def get_frames(self, user_id: str, start_time: datetime, end_time: datetime, min_salience: str = "L0") -> List[Dict[str, Any]]:
        salience_order = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}
        min_s = salience_order.get(min_salience, 0)
        
        result = []
        for f in self._frames:
            if f.get("user_id") == user_id and start_time <= f.get("timestamp_ntp") <= end_time:
                s = salience_order.get(f.get("salience_level", "L0"), 0)
                if s >= min_s:
                    result.append(f)
        return result
