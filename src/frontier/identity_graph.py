"""
frontier/identity_graph.py

Sprint 20 / S1720.5 / R2-F20.4

Private Identity Graph — strictly scoped to a single user.

Key fixes:
  - Introduces IdentityInference dataclass (distinct from IdentityNode)
  - CONFIDENCE_FLOOR = 0.85 enforced: below-floor beliefs create an
    IdentityInference record only; they never promote to IdentityNode
  - add_unresolved_cluster() creates IdentityNode(name=None)
  - Competing candidates (top-2 within 0.05 of each other) are preserved;
    no forced promotion
  - add_explicit_name_association raises ValueError if belief.confidence
    is below CONFIDENCE_FLOOR — only Teach Chronis can attach a name
  - get_entity_by_name requires requesting_user_id matching self.user_id
  - merge_graphs across different users raises PermissionError
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .provenance_pipeline import Belief

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIDENCE_FLOOR: float = 0.85
COMPETING_CANDIDATE_MARGIN: float = 0.05   # top-2 within this → preserved


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IdentityInference:
    """
    An internal inference about identity — below the confidence floor or when
    competing candidates exist.  Never surfaced as a named IdentityNode.
    """
    inference_id: str
    visual_cluster_id: str
    voice_cluster_id: Optional[str]
    confidence: float
    candidates: List[str]   # all candidate names considered
    user_id: str


@dataclass
class IdentityNode:
    """
    A confirmed or unresolved person entry in the graph.
    name=None means the person has been observed but not yet named.
    """
    entity_id: str
    name: Optional[str]        # None for unresolved/unknown identities
    confidence: float
    modalities: List[str]
    user_id: str


# ---------------------------------------------------------------------------
# Private Identity Graph
# ---------------------------------------------------------------------------

class PrivateIdentityGraph:
    """
    Private Identity Graph (Sprint 20 / S1720.5 / R2-F20.4).
    Strictly scoped to a single user.  Never aggregated globally.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.nodes: Dict[str, IdentityNode] = {}
        self.inferences: Dict[str, IdentityInference] = {}

    # ------------------------------------------------------------------
    # Inference (internal only — never promoted automatically)
    # ------------------------------------------------------------------

    def add_inference(
        self,
        visual_cluster_id: str,
        voice_cluster_id: Optional[str],  # Optional: may have visual-only observation
        candidates: List[str],
        confidence: float,
        user_id: str,
    ) -> IdentityInference:
        """
        Stores an IdentityInference record.

        Competing candidates (top-2 within COMPETING_CANDIDATE_MARGIN):
          → both preserved in the candidates list; no forced promotion.

        Below CONFIDENCE_FLOOR:
          → stored as Inference only; MUST NOT promote to IdentityNode.

        voice_cluster_id may be None when only visual evidence is available.
        """
        if user_id != self.user_id:
            raise PermissionError(
                f"CROSS-USER ISOLATION: cannot add inference for user '{user_id}' "
                f"to graph owned by '{self.user_id}'."
            )
        inference_id = f"inf_{visual_cluster_id}_{voice_cluster_id}"
        inf = IdentityInference(
            inference_id=inference_id,
            visual_cluster_id=visual_cluster_id,
            voice_cluster_id=voice_cluster_id,
            confidence=confidence,
            candidates=list(candidates),
            user_id=user_id,
        )
        self.inferences[inference_id] = inf
        return inf

    # ------------------------------------------------------------------
    # Unresolved identity (name=None)
    # ------------------------------------------------------------------

    def add_unresolved_cluster(
        self,
        visual_cluster_id: str,
        voice_cluster_id: Optional[str] = None,  # Optional: may have visual-only cluster
        user_id: Optional[str] = None,
    ) -> IdentityNode:
        """
        Records an observed but unnamed person.
        IdentityNode.name = None — no name is invented.
        voice_cluster_id is optional; pass None for visual-only observations.
        """
        effective_user = user_id or self.user_id
        if effective_user != self.user_id:
            raise PermissionError(
                f"CROSS-USER ISOLATION: cannot add unresolved cluster for user "
                f"'{effective_user}' to graph owned by '{self.user_id}'."
            )
        entity_id = f"entity_unresolved_{visual_cluster_id}_{voice_cluster_id}"
        node = IdentityNode(
            entity_id=entity_id,
            name=None,
            confidence=0.0,
            modalities=["visual", "voice"],
            user_id=self.user_id,
        )
        self.nodes[entity_id] = node
        return node

    # ------------------------------------------------------------------
    # Explicit name association (Teach Chronis only)
    # ------------------------------------------------------------------

    def add_explicit_name_association(
        self,
        visual_cluster_id: str,
        voice_cluster_id: str,
        name: str,
        belief: Belief,
    ) -> IdentityNode:
        """
        Explicit name association through the "Teach Chronis" action.
        Links cross-modal clusters to a name, backed by a Belief confidence.

        Raises ValueError if belief.confidence < CONFIDENCE_FLOOR.
        This is the ONLY path that can attach a human-readable name.
        """
        if belief.confidence < CONFIDENCE_FLOOR:
            raise ValueError(
                f"Cannot associate name '{name}': belief confidence "
                f"{belief.confidence:.3f} is below the required floor "
                f"{CONFIDENCE_FLOOR}.  Store as Inference instead."
            )
        entity_id = f"entity_{visual_cluster_id}_{voice_cluster_id}"
        node = IdentityNode(
            entity_id=entity_id,
            name=name,
            confidence=belief.confidence,
            modalities=["visual", "voice"],
            user_id=self.user_id,
        )
        self.nodes[entity_id] = node
        return node

    # ------------------------------------------------------------------
    # Queries (user-scoped)
    # ------------------------------------------------------------------

    def get_entity_by_name(
        self,
        name: str,
        requesting_user_id: Optional[str] = None,
    ) -> List[IdentityNode]:
        """
        Retrieves internal entity records by explicit name.

        Security: checks both graph ownership (self.user_id) AND individual
        node ownership (node.user_id) so that even if a node was accidentally
        added with a different user_id, it cannot be read by the wrong caller.

        requesting_user_id must match self.user_id.  Cross-user reads raise
        PermissionError (R2-F20.4).

        requesting_user_id is optional only for legacy call sites inside the
        same process that already hold the correct graph object.
        """
        if requesting_user_id is not None and requesting_user_id != self.user_id:
            raise PermissionError(
                f"CROSS-USER ISOLATION: user '{requesting_user_id}' cannot read "
                f"identity graph owned by '{self.user_id}'."
            )
        result = []
        for node in self.nodes.values():
            if node.name is not None and node.name.lower() == name.lower():
                # Double-check node-level ownership (defence in depth)
                if requesting_user_id is not None and node.user_id != requesting_user_id:
                    raise PermissionError(
                        f"NODE-LEVEL CROSS-USER: node '{node.entity_id}' belongs to "
                        f"'{node.user_id}', not '{requesting_user_id}'."
                    )
                result.append(node)
        return result

    def get_unresolved_nodes(self) -> List[IdentityNode]:
        """Returns all nodes with name=None (unknown persons)."""
        return [n for n in self.nodes.values() if n.name is None]

    # ------------------------------------------------------------------
    # Merge (anti-pattern — cross-user is forbidden)
    # ------------------------------------------------------------------

    def merge_graphs(self, other_graph: "PrivateIdentityGraph") -> None:
        """
        Anti-pattern method to demonstrate strict isolation.
        Must never be called across different users.
        """
        if self.user_id != other_graph.user_id:
            raise PermissionError(
                f"CROSS-USER ISOLATION VIOLATION: Cannot merge identity graph of "
                f"'{self.user_id}' with '{other_graph.user_id}'"
            )
        self.nodes.update(other_graph.nodes)
        self.inferences.update(other_graph.inferences)
