from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from claims_engine.claim_levels import Claim

class ClaimsStoreProvider(ABC):
    """
    Interface bridging the Frontier layer to the Claims Engine.
    """

    @abstractmethod
    def get_claim(self, claim_id: str) -> Optional[Claim]:
        pass

    @abstractmethod
    def update_claim_status(self, claim_id: str, new_status: str) -> None:
        pass
    
    @abstractmethod
    def iter_claims(self):
        """Yields all claims."""
        pass
