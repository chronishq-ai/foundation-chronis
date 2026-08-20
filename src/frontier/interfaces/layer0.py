from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

class Layer0Storage(ABC):
    """
    Thin interface for Layer 0 / Canonical records (Sprint 1).
    Enforces the 'never delete from Layer 0' rule.
    """

    @abstractmethod
    def read_record(self, record_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def write_record(self, user_id: str, data: Dict[str, Any]) -> str:
        """Returns the canonical_record_pointer."""
        pass

    @abstractmethod
    def get_row_count(self, user_id: str) -> int:
        """For testing that Layer 0 is unaffected by deletion."""
        pass

class MockLayer0Storage(Layer0Storage):
    """Deterministic mock for Sprint 17 tests."""
    def __init__(self):
        self._records = {}
        self._counts = {}
        self._next_id = 1

    def read_record(self, record_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        record = self._records.get(record_id)
        if record and record.get('user_id') == user_id:
            return record
        return None

    def write_record(self, user_id: str, data: Dict[str, Any]) -> str:
        record_id = f"ptr_{self._next_id}"
        self._next_id += 1
        data_copy = dict(data)
        data_copy['user_id'] = user_id
        self._records[record_id] = data_copy
        self._counts[user_id] = self._counts.get(user_id, 0) + 1
        return record_id

    def get_row_count(self, user_id: str) -> int:
        return self._counts.get(user_id, 0)
