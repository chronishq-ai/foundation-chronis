"""Sprint 11 Day 32 -- portable Behavioral DNA export."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from upstream_interfaces import Claim, ClaimLevel
from signing import DeviceSigner


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_social_summary(summary: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if summary is None:
        return None
    forbidden = {"name", "full_name", "email", "phone", "address", "raw_fingerprint", "fingerprint", "participant_name"}
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                if str(k).lower() in forbidden:
                    raise ValueError("social_graph_summary contains identity/raw-fingerprint data")
                walk(v)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
    walk(summary)
    return summary


@dataclass(frozen=True)
class BehavioralDNAExport:
    user_id: str
    level3_claims: List[Claim]
    lexicon_profile: Optional[Dict[str, Any]] = None
    social_graph_summary: Optional[Dict[str, Any]] = None
    signature: Optional[str] = None
    is_signed: bool = False
    generated_at: datetime = field(default_factory=_utc_now)

    def signing_payload(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "level3_claims": self.level3_claims,
            "lexicon_profile": self.lexicon_profile,
            "social_graph_summary": self.social_graph_summary,
            "generated_at": self.generated_at,
        }

    def verify_signature(self, signer: DeviceSigner) -> bool:
        return bool(self.signature and self.is_signed and signer.verify(self.signing_payload(), self.signature))


def build_behavioral_dna_export(
    user_id: str,
    claims: List[Claim],
    lexicon_profile: Optional[Dict[str, Any]] = None,
    social_graph_summary: Optional[Dict[str, Any]] = None,
    device_signer: Optional[DeviceSigner] = None,
) -> BehavioralDNAExport:
    if not user_id:
        raise ValueError("user_id must not be empty")
    qualifying_claims = [
        claim for claim in claims
        if claim.level == ClaimLevel.LEVEL_3
        and claim.gate_evaluation.admissible
        and claim.user_id == user_id
    ]
    social_graph_summary = _safe_social_summary(social_graph_summary)
    export = BehavioralDNAExport(user_id, qualifying_claims, lexicon_profile, social_graph_summary)
    if device_signer is None:
        return export
    signature = device_signer.sign(export.signing_payload())
    return BehavioralDNAExport(user_id, qualifying_claims, lexicon_profile, social_graph_summary, signature, True, export.generated_at)
