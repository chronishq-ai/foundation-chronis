from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class ClaimsStoreProvider(ABC):
    """
    Interface bridging the Frontier layer to the Claims Engine.
    """

    @abstractmethod
    def get_claim(self, claim_id: str) -> Optional[Any]:
        pass

    @abstractmethod
    def get_claim_status(self, claim_id: str) -> Optional[str]:
        """
        Returns the current status string for a known claim, or None if the
        claim_id is not found.  Returning None (not "SURFACED") for unknown
        claim IDs is a hard requirement — the system must distinguish between
        "claim exists and is surfaced" and "claim does not exist."
        """
        pass

    @abstractmethod
    def update_claim_status(self, claim_id: str, new_status: str) -> None:
        """
        Raises KeyError when claim_id is not found.
        Silent no-ops are not acceptable (R2-F20.7).
        """
        pass

    @abstractmethod
    def append_claim_version(self, claim_id: str, claim_obj: Any) -> None:
        """Append a new immutable version of a claim."""
        pass

    @abstractmethod
    def apply_status_updates_batch(
        self, updates: List[Any]  # List[Tuple[str, str]]
    ) -> None:
        """
        Updates multiple claim statuses atomically.
        Validates all claim_ids before modifying any state.
        Raises KeyError if any claim_id is not found.
        """
        pass

    @abstractmethod
    def iter_claims(self):
        """Yields the latest version of every claim."""
        pass
