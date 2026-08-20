import pytest
from src.frontier.provenance_pipeline import ProvenanceManager, Belief
from src.frontier.conflict_resolution import ConflictManager
from src.frontier.identity_graph import PrivateIdentityGraph

def test_provenance_pipeline():
    """Validates Observation -> Claim promotion rules (Sprint 20)."""
    pm = ProvenanceManager()
    
    # Below floor -> Should not promote
    belief_low = Belief(id="b1", confidence=0.5, source_inference_ids=["i1"])
    assert pm.promote_to_claim(belief_low, "identity_match", 1, "test") is None
    
    # Above floor -> Promotes
    belief_high = Belief(id="b2", confidence=0.9, source_inference_ids=["i2"])
    claim = pm.promote_to_claim(belief_high, "identity_match", 1, "test")
    assert claim is not None
    assert claim.id == "claim_b2"

def test_conflict_resolution():
    """Validates that conflicts are retained, not deleted (Sprint 20)."""
    cm = ConflictManager()
    b1 = Belief(id="b1", confidence=0.9, source_inference_ids=[])
    b2 = Belief(id="b2", confidence=0.9, source_inference_ids=[])
    
    conflict = cm.resolve_contradiction(b1, b2)
    assert conflict.belief_id_1 == "b1"
    assert conflict.belief_id_2 == "b2"
    # Note: b1 and b2 are not deleted.

def test_identity_graph_isolation():
    """Validates PrivateIdentityGraph isolation (Sprint 20)."""
    graph1 = PrivateIdentityGraph("u1")
    graph2 = PrivateIdentityGraph("u2")
    
    with pytest.raises(PermissionError):
        graph1.merge_graphs(graph2)
