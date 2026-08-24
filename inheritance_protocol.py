"""Sprint 11 Day 33 -- Behavioral Letter inheritance orchestration."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional
from behavioral_dna import BehavioralDNAExport
from upstream_interfaces import Claim, SessionExcerpt
from signing import DeviceSigner

class NoEligibleClaimError(Exception):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

@dataclass(frozen=True)
class BehavioralLetter:
    user_id: str
    source_claim_id: str
    letter_text: str
    citation_chain: List[str]
    signature: Optional[str] = None
    is_signed: bool = False
    generated_at: datetime = field(default_factory=_utc_now)

    def signing_payload(self) -> dict:
        return {
            "user_id": self.user_id,
            "source_claim_id": self.source_claim_id,
            "letter_text": self.letter_text,
            "citation_chain": self.citation_chain,
            "generated_at": self.generated_at,
        }

    def verify_signature(self, signer: DeviceSigner) -> bool:
        return bool(self.signature and self.is_signed and signer.verify(self.signing_payload(), self.signature))

InsightGeneratorFn = Callable[[Claim, Any, List[SessionExcerpt], Any], Any]

def _pick_most_recent_claim(claims: List[Claim]) -> Claim:
    return max(claims, key=lambda claim: claim.created_at)

def build_inheritance_letter(
    export: BehavioralDNAExport,
    divergence_state: Any,
    candidate_excerpts: List[SessionExcerpt],
    insight_generator: InsightGeneratorFn,
    llm_client: Any,
    device_signer: Optional[DeviceSigner] = None,
) -> BehavioralLetter:
    if not export.level3_claims:
        raise NoEligibleClaimError(f"user {export.user_id} has no admissible Level 3 claims -- cannot build a Behavioral Letter yet.")
    if any(excerpt.user_id != export.user_id for excerpt in candidate_excerpts):
        raise ValueError("candidate excerpts must belong to the export user")
    chosen_claim = _pick_most_recent_claim(export.level3_claims)
    generated = insight_generator(chosen_claim, divergence_state, candidate_excerpts, llm_client)
    text = getattr(generated, "text", None)
    citations = getattr(generated, "citation_chain", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("constrained-RAG generator must return non-empty text")
    if not isinstance(citations, (list, tuple)):
        raise ValueError("constrained-RAG generator must return a citation chain")
    for excerpt in candidate_excerpts:
        if excerpt.text and excerpt.text in text:
            raise ValueError("Behavioral Letter must not expose raw session-excerpt text")
    citation_ids = [str(x) for x in citations]
    allowed_sessions = {e.session_id for e in candidate_excerpts}
    if any(c not in allowed_sessions for c in citation_ids):
        raise ValueError("citation chain contains a session outside the candidate evidence")
    letter = BehavioralLetter(export.user_id, chosen_claim.claim_id, text, citation_ids)
    if device_signer is None:
        return letter
    signature = device_signer.sign(letter.signing_payload())
    return BehavioralLetter(export.user_id, chosen_claim.claim_id, text, citation_ids, signature, True, letter.generated_at)
