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
    assert claim.claim_id == "claim_b2"

from src.frontier.interfaces.claims_store import ClaimsStoreProvider

class MockClaimsStore(ClaimsStoreProvider):
    def __init__(self):
        self.claims = {}
    def get_claim(self, claim_id: str):
        return self.claims.get(claim_id)
    def update_claim_status(self, claim_id: str, new_status: str):
        if claim_id in self.claims:
            self.claims[claim_id].status = new_status
    def iter_claims(self):
        yield from self.claims.values()

def test_conflict_resolution():
    """Validates that conflicts are retained, not deleted (Sprint 20)."""
    store = MockClaimsStore()
    cm = ConflictManager(store)
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

from claims_engine.claim_levels import Claim, ClaimLevel, GateEvaluation
from src.frontier.claims_engine_adapter import ClaimsEngineAdapter
from src.frontier.provenance_pipeline import ProvenanceRecord

def test_claims_engine_integration():
    """Integration test: ClaimsEngine -> ClaimsStoreProvider -> ConflictManager"""
    adapter = ClaimsEngineAdapter()
    
    claim = Claim.new(
        user_id="u1",
        domain_id="d1",
        level=ClaimLevel.LEVEL_0,
        gate_evaluation=GateEvaluation(level=ClaimLevel.LEVEL_0, admissible=True, checks=[])
    )
    # Seed the adapter
    adapter._claims[claim.claim_id] = claim
    
    cm = ConflictManager(adapter)
    b1 = Belief(id="b1", confidence=0.9, source_inference_ids=[])
    b2 = Belief(id="b2", confidence=0.9, source_inference_ids=[])
    
    conflict = cm.resolve_contradiction(b1, b2)
    
    pr = ProvenanceRecord(claim_id=claim.claim_id, source_belief_ids=["b1"])
    
    cm.downgrade_dependent_claims(conflict, [pr])
    
    assert adapter.get_claim_status(claim.claim_id) == "UNCLEAR"
    assert pr.status == "UNCLEAR"
