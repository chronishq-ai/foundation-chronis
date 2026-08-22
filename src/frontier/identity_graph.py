from typing import Dict, Any, List
from .provenance_pipeline import Belief
from dataclasses import dataclass

@dataclass
class IdentityNode:
    entity_id: str
    name: str
    confidence: float # Typed as Belief in pipeline, represented here as float
    modalities: List[str]

class PrivateIdentityGraph:
    """
    Private Identity Graph (Sprint 20).
    Strictly scoped to a single user. Never aggregated globally.
    """
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.nodes: Dict[str, IdentityNode] = {}
        
    def add_explicit_name_association(self, visual_cluster_id: str, voice_cluster_id: str, name: str, belief: Belief) -> None:
        """
        Explicit name association through Teach Chronis.
        Links cross-modal clusters to a name, backed by a Belief confidence.
        """
        entity_id = f"entity_{visual_cluster_id}_{voice_cluster_id}"
        
        self.nodes[entity_id] = IdentityNode(
            entity_id=entity_id,
            name=name,
            confidence=belief.confidence,
            modalities=["visual", "voice"]
        )
        
    def get_entity_by_name(self, name: str) -> List[IdentityNode]:
        """Retrieves internal entity records by explicit name."""
        return [node for node in self.nodes.values() if node.name.lower() == name.lower()]

    def merge_graphs(self, other_graph: 'PrivateIdentityGraph') -> None:
        """
        Anti-pattern method to demonstrate strict isolation.
        Should never be called across different users.
        """
        if self.user_id != other_graph.user_id:
            raise PermissionError(f"CROSS-USER ISOLATION VIOLATION: Cannot merge identity graph of {self.user_id} with {other_graph.user_id}")
        
        # Merge logic for same user would go here
        self.nodes.update(other_graph.nodes)
