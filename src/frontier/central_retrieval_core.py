from typing import Dict, Any, List, Optional

from .interfaces.policy import PolicyEngine
from .interfaces.mirror import MirrorProvider


class CrossUserEvidenceError(PermissionError):
    """
    Raised when the retrieval provider returns evidence whose owner does not
    match the requesting user.  This is a security violation — the Central
    Retrieval Core fails closed; it never silently strips bad records and
    continues (R2-F18.1 / S1720.9).
    """
    pass


class CentralRetrievalCore:
    """
    Central Retrieval Core (Sprint 18 / R2-F18.1).

    This is the ONLY authorised path for personal-memory retrieval.  Every
    caller — voice assistant, multimodal assistant, explainability — MUST go
    through this class.  Direct calls to visual_memory, transcript_search, or
    any other retrieval primitive from interface modules are a CI violation.

    Security contract (fail-closed):
      1. Consent is verified through the injected PolicyEngine — never via a
         caller-supplied consent_context dict.
      2. Every evidence item returned by the orchestrator is checked:
             item["owner_user_id"] == requesting_user_id
         If ANY item fails this check, CrossUserEvidenceError is raised
         immediately.  We do NOT strip bad records and continue.
      3. Deduplicate + rank ONLY after ownership verification.
    """

    def __init__(self, memory_orchestrator, policy_engine: PolicyEngine):
        self.orchestrator = memory_orchestrator
        self.policy_engine = policy_engine

    def retrieve(
        self,
        query: str,
        query_type: str,
        requesting_interface: str,
        user_id: str,
        consent_context: Dict[str, Any],  # kept for API compat but NOT trusted
    ) -> Dict[str, Any]:
        """
        Contract: {query, query_type, requesting_interface, user_id, consent_context}
        Returns a single ranked EvidencePackage.

        consent_context is accepted for API compatibility but the actual
        authorisation decision is made by the injected PolicyEngine, not by
        reading consent_context["consent_tier"].
        """
        # --- 1. Consent via authenticated policy engine, not caller-supplied tier ---
        if not self.policy_engine.check_access(user_id, "personal_retrieval", required_tier=2):
            return {"error": "Access denied by policy engine"}

        # --- 2. Retrieve ---
        evidence_package = self.orchestrator.orchestrate(
            user_id=user_id,
            query=query,
            query_type=query_type,
        )

        # --- 3. Per-evidence ownership verification — FAIL CLOSED ---
        # Items with NO owner_user_id are also a violation: we cannot return
        # evidence of unknown provenance.  The guard is intentionally strict:
        # owner != user_id  (this includes owner == None).
        evidence_items = evidence_package.get("evidence_items", [])
        for item in evidence_items:
            owner = item.get("owner_user_id")
            if owner != user_id:
                raise CrossUserEvidenceError(
                    f"SECURITY VIOLATION: evidence item owner {owner!r} does not match "
                    f"requesting user {user_id!r}. Retrieval aborted (fail-closed). "
                    f"Interface: {requesting_interface}"
                )

        # --- 4. Deduplicate + rank (only after verification) ---
        # Items with no content_pointer are skipped: a None key would collapse
        # all pointer-less items under a single dict entry, silently dropping
        # all but the highest-confidence one (Bug 2 remediation).
        unique_items: Dict[str, Dict] = {}
        for item in evidence_items:
            ptr = item.get("content_pointer")
            if ptr is None:
                continue   # cannot deduplicate an item with no content pointer
            if ptr not in unique_items or item.get("confidence", 0) > unique_items[ptr].get("confidence", 0):
                unique_items[ptr] = item

        ranked_items = sorted(unique_items.values(), key=lambda x: x.get("confidence", 0), reverse=True)
        evidence_package["evidence_items"] = ranked_items

        return evidence_package
