from abc import ABC, abstractmethod
from typing import Dict, Any, List

class PolicyEngine(ABC):
    """
    Thin interface for Policy/Access Engine (Sprint 1).
    """

    @abstractmethod
    def check_access(self, user_id: str, resource_type: str, required_tier: int) -> bool:
        pass

    @abstractmethod
    def grant_emergency_access(self, granter_id: str, principal: str, scope: List[str], duration_hours: int) -> bool:
        """
        Sprint 17 extension for Tier 4/5 emergency access.
        """
        pass

class MockPolicyEngine(PolicyEngine):
    """Deterministic mock for testing."""
    def __init__(self):
        self._emergency_grants = []

    def check_access(self, user_id: str, resource_type: str, required_tier: int) -> bool:
        return True # Default open for tests unless specifically mocked otherwise.

    def grant_emergency_access(self, granter_id: str, principal: str, scope: List[str], duration_hours: int) -> bool:
        self._emergency_grants.append({
            "granter": granter_id,
            "principal": principal,
            "scope": scope,
            "duration": duration_hours
        })
        return True
