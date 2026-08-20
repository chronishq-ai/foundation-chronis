from typing import Dict, Any, List, Optional
from .interfaces.claims_store import ClaimsStoreProvider
from claims_engine.claim_levels import Claim

class ClaimsEngineAdapter(ClaimsStoreProvider):
    """
    Production adapter over the Sprint 13 Claims Engine.
    """
    def __init__(self):
        # In a full integration, this would connect to the actual ClaimsEngine database/store.
        self._claims: Dict[str, Claim] = {}
        self._claim_statuses: Dict[str, str] = {}

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        return self._claims.get(claim_id)
        
    def get_claim_status(self, claim_id: str) -> str:
        return self._claim_statuses.get(claim_id, "SURFACED")

    def update_claim_status(self, claim_id: str, new_status: str) -> None:
        """
        Updates the surfacing status of a claim (e.g., to UNCLEAR).
        Because the underlying Claim is frozen (G2: append-only), we maintain
        status as a separate property in the adapter/store.
        """
        if claim_id in self._claims:
            self._claim_statuses[claim_id] = new_status

    def iter_claims(self):
        for claim in self._claims.values():
            yield claim
