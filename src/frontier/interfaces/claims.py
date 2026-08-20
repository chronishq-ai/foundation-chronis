from abc import ABC, abstractmethod
from typing import Dict, Any, List

class ClaimsProvider(ABC):
    """
    Interface matching origin/feature/sprints-7-to-9.
    """

    @abstractmethod
    def get_claim(self, claim_id: str) -> Dict[str, Any]:
        """
        Retrieves a claim including its citation chain.
        """
        pass

    @abstractmethod
    def filter_clinical_terminology(self, text: str) -> bool:
        """
        Returns True if text passes the filter, False if it needs human review.
        """
        pass

class MockClaimsProvider(ClaimsProvider):
    def __init__(self, claims=None):
        self._claims = claims or {}

    def get_claim(self, claim_id: str) -> Dict[str, Any]:
        return self._claims.get(claim_id, {"id": claim_id, "confidence": 0.8, "citation_chain": []})

    def filter_clinical_terminology(self, text: str) -> bool:
        # Simple mock filter: reject if it contains 'depression' or 'trauma'
        lower_text = text.lower()
        if "depression" in lower_text or "trauma" in lower_text:
            return False
        return True
