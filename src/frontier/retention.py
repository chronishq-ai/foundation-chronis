from typing import Dict, Any, List
from .interfaces.policy import PolicyEngine

class RetrievalCache:
    """
    Retrieval-tier cache for visual memory performance.
    Explicitly NOT a Layer-0 retention policy. Deleting cache entries has ZERO effect on Layer 0.
    """
    def __init__(self):
        self.cache = {}

    def get(self, key: str) -> Any:
        return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        self.cache[key] = value

    def invalidate(self, key: str) -> None:
        """
        Cache invalidation. Does not touch canonical storage.
        """
        if key in self.cache:
            del self.cache[key]

class EmergencyAccessManager:
    """
    Emergency access extending the existing Tier 4/5 schema.
    Must pass through PolicyEngine.
    """
    def __init__(self, policy_engine: PolicyEngine):
        self.policy_engine = policy_engine

    def request_emergency_access(self, granter_id: str, principal: str, scope: List[str], duration_hours: int) -> bool:
        """
        Grants a time-boxed emergency access grant to a trusted contact.
        """
        return self.policy_engine.grant_emergency_access(
            granter_id=granter_id,
            principal=principal,
            scope=scope,
            duration_hours=duration_hours
        )
