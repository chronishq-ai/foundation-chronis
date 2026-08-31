"""
frontier/provenance_pipeline.py

Sprint 20 / S1720.7 / R2-F20.2

Implements the canonical Observation → Feature → Inference → Belief → Claim
provenance chain with:
  - Explicit stable IDs and typed references at every edge
  - User ownership on every node
  - ProvenanceStore for canonical persistence (JSON, no pickle)
  - explain_retrofitted() walks the stored graph — no caller-provided data accepted
  - Cross-user edges raise PermissionError
  - Missing nodes return a structured error dict with a defined schema

Provenance missing-node error schema:
  {
    "status": "error",
    "error_type": "MISSING_PROVENANCE_NODE",
    "node_type": "<Observation|Feature|Inference|Belief|ClaimLink>",
    "node_id": "<id that was not found>"
  }
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed provenance nodes — every node carries user_id for ownership checks
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Typed provenance nodes — every node carries user_id for ownership checks
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    observation_id: str
    user_id: str
    raw_data: Any
    timestamp: datetime

    @property
    def id(self) -> str:          # convenience alias used by older call sites
        return self.observation_id


@dataclass
class Feature:
    feature_id: str
    user_id: str
    source_observation_ids: List[str]
    derived_representation: Any

    @property
    def id(self) -> str:
        return self.feature_id

    # Legacy alias
    @property
    def source_observation_id(self) -> Optional[str]:
        return self.source_observation_ids[0] if self.source_observation_ids else None


@dataclass
class Inference:
    inference_id: str
    user_id: str
    source_feature_ids: List[str]
    candidate_match: Any

    @property
    def id(self) -> str:
        return self.inference_id

    # Legacy alias
    @property
    def source_feature_id(self) -> Optional[str]:
        return self.source_feature_ids[0] if self.source_feature_ids else None


@dataclass
class Belief:
    id: str
    confidence: float
    source_inference_ids: List[str]
    user_id: str = ""

    @property
    def belief_id(self) -> str:
        return self.id


@dataclass
class ClaimLink:
    """Links a claim_id to the belief IDs that produced it, with user scope."""
    claim_id: str
    source_belief_ids: List[str]
    user_id: str
    status: str = "SURFACED"


# Legacy alias kept for backwards-compat with test_sprint20.py
@dataclass
class ProvenanceRecord:
    claim_id: str
    source_belief_ids: List[str]
    status: str = "SURFACED"


# ---------------------------------------------------------------------------
# ProvenanceStore — canonical JSON-backed storage
# ---------------------------------------------------------------------------

class ProvenanceStore:
    """
    Stores and retrieves provenance nodes, persisting to a JSON lines file.

    All writes are append-only; reads reconstruct state by replaying.
    No pickle is used anywhere.
    Note: ProvenanceStore is single-writer. Concurrent writes require an external lock.
    """

    def __init__(self, store_path: Optional[str] = None):
        self._observations: Dict[str, Observation] = {}
        self._features: Dict[str, Feature] = {}
        self._inferences: Dict[str, Inference] = {}
        self._beliefs: Dict[str, Belief] = {}
        self._claim_links: Dict[str, ClaimLink] = {}
        self._store_path = store_path

        if store_path and os.path.exists(store_path):
            self._load(store_path)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if l.strip()]
        for i, line in enumerate(lines):
            is_last = (i == len(lines) - 1)
            try:
                record = json.loads(line)
                self._apply(record)
            except json.JSONDecodeError as e:
                if is_last:
                    logger.warning("Skipping truncated tail record in %s: %s", path, e)
                else:
                    logger.error(
                        "STORE CORRUPTION: malformed interior record at line %d in %s: %s",
                        i + 1, path, e
                    )
                    raise

    def find_claims_referencing(self, belief_ids: set) -> List[str]:
        """
        Returns claim_ids whose ClaimLink.source_belief_ids intersect belief_ids.
        Used by ConflictManager to find provenance-dependent claims only.
        """
        return [
            cl.claim_id
            for cl in self._claim_links.values()
            if belief_ids.intersection(cl.source_belief_ids)
        ]

    def _apply(self, record: Dict) -> None:
        node_type = record.get("node_type")
        data = record.get("data", {})
        if node_type == "Observation":
            ts = data.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            obs = Observation(
                observation_id=data["observation_id"],
                user_id=data["user_id"],
                raw_data=data.get("raw_data"),
                timestamp=ts,
            )
            self._observations[obs.observation_id] = obs
        elif node_type == "Feature":
            feat = Feature(
                feature_id=data["feature_id"],
                user_id=data["user_id"],
                source_observation_ids=data.get("source_observation_ids", []),
                derived_representation=data.get("derived_representation"),
            )
            self._features[feat.feature_id] = feat
        elif node_type == "Inference":
            inf = Inference(
                inference_id=data["inference_id"],
                user_id=data["user_id"],
                source_feature_ids=data.get("source_feature_ids", []),
                candidate_match=data.get("candidate_match"),
            )
            self._inferences[inf.inference_id] = inf
        elif node_type == "Belief":
            b = Belief(
                id=data["id"],
                user_id=data["user_id"],
                confidence=data["confidence"],
                source_inference_ids=data.get("source_inference_ids", []),
            )
            self._beliefs[b.id] = b
        elif node_type == "ClaimLink":
            cl = ClaimLink(
                claim_id=data["claim_id"],
                source_belief_ids=data.get("source_belief_ids", []),
                user_id=data["user_id"],
                status=data.get("status", "SURFACED"),
            )
            self._claim_links[cl.claim_id] = cl

    def _persist(self, node_type: str, data: Dict) -> None:
        if self._store_path is None:
            return
        record = json.dumps({"node_type": node_type, "data": data})
        tmp = self._store_path + ".tmp"
        # Atomic append: load existing + new, write to tmp, rename
        existing = ""
        if os.path.exists(self._store_path):
            with open(self._store_path, "r", encoding="utf-8") as f:
                existing = f.read()
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(existing)
            f.write(record + "\n")
        os.replace(tmp, self._store_path)

    def store_observation(self, obs: Observation) -> None:
        self._observations[obs.observation_id] = obs
        self._persist("Observation", {
            "observation_id": obs.observation_id,
            "user_id": obs.user_id,
            "raw_data": str(obs.raw_data),
            "timestamp": obs.timestamp.isoformat() if obs.timestamp else None,
        })

    def store_feature(self, feat: Feature) -> None:
        self._features[feat.feature_id] = feat
        self._persist("Feature", {
            "feature_id": feat.feature_id,
            "user_id": feat.user_id,
            "source_observation_ids": feat.source_observation_ids,
            "derived_representation": str(feat.derived_representation),
        })

    def store_inference(self, inf: Inference) -> None:
        self._inferences[inf.inference_id] = inf
        self._persist("Inference", {
            "inference_id": inf.inference_id,
            "user_id": inf.user_id,
            "source_feature_ids": inf.source_feature_ids,
            "candidate_match": str(inf.candidate_match),
        })

    def store_belief(self, belief: Belief) -> None:
        self._beliefs[belief.id] = belief
        self._persist("Belief", {
            "id": belief.id,
            "user_id": belief.user_id,
            "confidence": belief.confidence,
            "source_inference_ids": belief.source_inference_ids,
        })

    def link_claim(self, claim_id: str, source_belief_ids: List[str], user_id: str) -> ClaimLink:
        cl = ClaimLink(claim_id=claim_id, source_belief_ids=source_belief_ids, user_id=user_id)
        self._claim_links[claim_id] = cl
        self._persist("ClaimLink", {
            "claim_id": claim_id,
            "source_belief_ids": source_belief_ids,
            "user_id": user_id,
            "status": cl.status,
        })
        return cl

    def _missing_node_error(self, node_type: str, node_id: str) -> Dict:
        """Defined provenance missing-node error schema."""
        return {
            "status": "error",
            "error_type": "MISSING_PROVENANCE_NODE",
            "node_type": node_type,
            "node_id": node_id,
        }

    def _check_ownership(self, node, requesting_user_id: str, node_type: str) -> None:
        if node.user_id != requesting_user_id:
            raise PermissionError(
                f"CROSS-USER PROVENANCE EDGE DETECTED: {node_type} '{getattr(node, node_type.lower() + '_id', node.id if hasattr(node, 'id') else '')}' "
                f"belongs to user '{node.user_id}', not '{requesting_user_id}'."
            )

    def reconstruct_chain(self, claim_id: str, requesting_user_id: str) -> Dict:
        """
        Walks the canonical provenance graph:
            ClaimLink → Beliefs → Inferences → Features → Observations

        Every node's user_id is checked against requesting_user_id.
        - Cross-user edge → PermissionError (fail closed)
        - Missing node → structured error dict (defined schema, not exception)

        Returns a full chain dict on success.
        """
        # --- Claim link ---
        claim_link = self._claim_links.get(claim_id)
        if claim_link is None:
            return self._missing_node_error("ClaimLink", claim_id)
        if claim_link.user_id != requesting_user_id:
            raise PermissionError(
                f"CROSS-USER PROVENANCE: claim '{claim_id}' belongs to '{claim_link.user_id}', "
                f"not '{requesting_user_id}'."
            )

        chain: Dict = {"claim_id": claim_id, "beliefs": []}

        for belief_id in claim_link.source_belief_ids:
            belief = self._beliefs.get(belief_id)
            if belief is None:
                chain["beliefs"].append(self._missing_node_error("Belief", belief_id))
                continue
            self._check_ownership(belief, requesting_user_id, "Belief")

            belief_entry: Dict = {
                "belief_id": belief.id,
                "confidence": belief.confidence,
                "inferences": [],
            }

            for inf_id in belief.source_inference_ids:
                inf = self._inferences.get(inf_id)
                if inf is None:
                    belief_entry["inferences"].append(self._missing_node_error("Inference", inf_id))
                    continue
                self._check_ownership(inf, requesting_user_id, "Inference")

                inf_entry: Dict = {"inference_id": inf.inference_id, "features": []}

                for feat_id in inf.source_feature_ids:
                    feat = self._features.get(feat_id)
                    if feat is None:
                        inf_entry["features"].append(self._missing_node_error("Feature", feat_id))
                        continue
                    self._check_ownership(feat, requesting_user_id, "Feature")

                    feat_entry: Dict = {"feature_id": feat.feature_id, "observations": []}

                    for obs_id in feat.source_observation_ids:
                        obs = self._observations.get(obs_id)
                        if obs is None:
                            feat_entry["observations"].append(self._missing_node_error("Observation", obs_id))
                            continue
                        self._check_ownership(obs, requesting_user_id, "Observation")
                        feat_entry["observations"].append({
                            "observation_id": obs.observation_id,
                            "timestamp": obs.timestamp.isoformat() if obs.timestamp else None,
                        })

                    inf_entry["features"].append(feat_entry)
                belief_entry["inferences"].append(inf_entry)

            chain["beliefs"].append(belief_entry)

        return chain


# ---------------------------------------------------------------------------
# ProvenanceManager — high-level facade used by the rest of the system
# ---------------------------------------------------------------------------

class ProvenanceManager:
    """
    Manages the Observation → Feature → Inference → Belief → Claim pipeline.
    (Sprint 20 / S1720.7 / R2-F20.2)
    """

    def __init__(self, claims_store=None, provenance_store: Optional[ProvenanceStore] = None):
        self.claims_store = claims_store
        self.provenance_store = provenance_store or ProvenanceStore()
        self.confidence_floors = {
            "identity_match": 0.85,
            "behavioral_pattern": 0.70,
        }

    def promote_to_claim(
        self,
        belief: Belief,
        claim_type: str,
        level: int,
        content: str,
    ) -> Optional[ProvenanceRecord]:
        """
        Hard rule: a Belief below the per-claim-type confidence floor creates
        only an internal Inference record and MUST NOT promote to Claim.
        """
        floor = self.confidence_floors.get(claim_type, 0.90)
        if belief.confidence < floor:
            return None

        claim_id = f"claim_{belief.id}"
        record = ProvenanceRecord(
            claim_id=claim_id,
            source_belief_ids=[belief.id],
        )
        # Also register in the provenance store so explain_retrofitted can walk it
        self.provenance_store.link_claim(
            claim_id=claim_id,
            source_belief_ids=[belief.id],
            user_id=belief.user_id,
        )
        return record

    def explain_retrofitted(self, claim_id: str, requesting_user_id: str = "") -> Dict:
        """
        Canonical explain() API — walks the stored provenance graph.
        No caller-provided citation chain is accepted (R2-F20.1 / S1720.4).

        If requesting_user_id is provided, cross-user edges raise PermissionError.
        If not provided (legacy usage), ownership checks are skipped.
        """
        if requesting_user_id:
            return self.provenance_store.reconstruct_chain(claim_id, requesting_user_id)

        warnings.warn(
            "explain_retrofitted() without requesting_user_id bypasses ownership checks "
            "and is deprecated. Provide requesting_user_id.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Legacy path: fall back to claims_store attribute lookup
        if not self.claims_store:
            return {"error": "Claims store not configured"}
        claim_data = self.claims_store.get_claim(claim_id)
        if not claim_data:
            return {"error": "Claim not found"}

        return {
            "claim": claim_data,
            "belief_confidence": getattr(claim_data, "confidence", 0.0),
            "inference": getattr(claim_data, "inference", "No inference data"),
            "observation": getattr(claim_data, "observation", "No observation data"),
        }
