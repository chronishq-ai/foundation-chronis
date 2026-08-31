"""
frontier/explainability.py

Sprint 17 / S1720.4 / R2-F20.1

Canonical Explainability API.

Key fix: explain() now accepts only (claim_id, requesting_user_id).
No caller-provided evidence or citation chains are accepted.
The server fetches canonical provenance from the store and walks the graph.
"""

from typing import Any, Dict, Optional

from claims_engine.grounded_generation import contains_clinical_terminology
from .interfaces.mirror import MirrorProvider
from .interfaces.layer0 import Layer0Storage


class ExplainabilityAPI:
    def __init__(
        self,
        layer0: Layer0Storage,
        mirror: MirrorProvider,
        claims_store=None,
        provenance_manager=None,
    ):
        self.layer0 = layer0
        self.mirror = mirror
        self.claims_store = claims_store
        self.provenance_manager = provenance_manager

    def explain(self, claim_id: str, requesting_user_id: str) -> Dict[str, Any]:
        """
        Canonical explain() API (R2-F20.1 / S1720.4).

        Fetches claim and provenance from canonical store.
        No caller-provided citation_chain is accepted.

        Steps:
          1. Retrieve claim from claims_store
          2. Verify user ownership
          3. Walk canonical provenance chain
          4. Re-apply clinical/safety filter on each citation
          5. Return structured explanation
        """
        if not self.claims_store:
            return {"error": "Claims store not configured"}

        # 1. Fetch canonical claim
        claim = self.claims_store.get_claim(claim_id)
        if claim is None:
            return {"error": f"Claim '{claim_id}' not found"}

        # 2. Verify ownership
        claim_user = getattr(claim, "user_id", None)
        if claim_user and claim_user != requesting_user_id:
            return {"error": "Access denied: claim belongs to a different user"}

        # 3. Walk canonical provenance
        chain: Dict = {}
        if self.provenance_manager:
            chain = self.provenance_manager.explain_retrofitted(
                claim_id, requesting_user_id
            )
            if chain.get("status") == "error":
                return chain   # propagate provenance error

        # 4. Re-apply clinical filter on every citation in the chain
        citations = self._extract_citations(chain)
        for cite_text in citations:
            if contains_clinical_terminology(cite_text):
                return {
                    "error": "Content blocked by clinical terminology filter. Routed to human review.",
                    "routed_to_review": True,
                }

        # 5. Return structured explanation
        gate_eval = getattr(claim, "gate_evaluation", None)
        if hasattr(gate_eval, "admissible"):
            confidence = gate_eval.admissible
        elif hasattr(claim, "confidence"):
            confidence = claim.confidence
        else:
            confidence = None

        return {
            "claim_id": claim_id,
            "confidence": confidence,
            "provenance_chain": chain,
            "evidence_list": citations,
        }

    def _extract_citations(self, chain: Dict) -> list:
        """Flatten all observation text/IDs from the provenance chain for clinical filtering.

        Returns a list of strings: raw_data text when available, otherwise the
        observation_id.  This ensures the clinical filter operates on actual
        content rather than opaque ID strings that would never trigger it.
        """
        citations = []
        for belief in chain.get("beliefs", []):
            for inf in belief.get("inferences", []):
                for feat in inf.get("features", []):
                    for obs in feat.get("observations", []):
                        # Prefer raw_data text for clinical filtering; fall back to ID
                        text = obs.get("raw_data") or obs.get("observation_id", "")
                        if text:
                            citations.append(str(text))
        return citations

    def teach_chronis(
        self,
        user_id: str,
        claim_id: str,
        correction_data: Dict[str, Any],
    ) -> str:
        """
        Correction endpoint.
        """
        correction = {
            "type": "teach_chronis_correction",
            "target_claim_id": claim_id,
            "correction_data": correction_data,
        }
        pointer = self.layer0.write_record(user_id, correction)
        self.mirror.process_teach_correction(user_id, correction)
        return pointer
