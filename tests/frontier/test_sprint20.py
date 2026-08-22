import pytest
from frontier.provenance_pipeline import ProvenanceManager, Belief
from frontier.conflict_resolution import ConflictManager
from frontier.identity_graph import PrivateIdentityGraph

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

def test_provenance_explain_retrofitted():
    """S1720.7: Validates explain_retrofitted uses actual store."""
    class RealFakeStore(ClaimsStoreProvider):
        def get_claim(self, claim_id):
            if claim_id == "real_claim":
                class FakeClaimData:
                    confidence = 0.88
                    inference = "real_inf"
                    observation = "real_obs"
                return FakeClaimData()
            return None
        def update_claim_status(self, c, s): pass
        def iter_claims(self): yield None

    pm = ProvenanceManager(claims_store=RealFakeStore())

    # Unknown claim -> error
    res_err = pm.explain_retrofitted("unknown_claim")
    assert "error" in res_err

    # Real claim -> real provenance
    res_real = pm.explain_retrofitted("real_claim")
    assert res_real["belief_confidence"] == 0.88
    assert res_real["inference"] == "real_inf"

    # Arbitrary caller data cannot fabricate provenance (API doesn't accept it)
    import inspect
    sig = inspect.signature(pm.explain_retrofitted)
    assert "claim_data" not in sig.parameters

from frontier.interfaces.claims_store import ClaimsStoreProvider

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

def test_conflict_resolution_targeted():
    """S1720.6: Validates targeted apply_teach_correction."""
    store = MockClaimsStore()
    cm = ConflictManager(store)
    b1 = Belief(id="b1", confidence=0.9, source_inference_ids=[])
    b2 = Belief(id="b2", confidence=0.9, source_inference_ids=[])
    b3 = Belief(id="b3", confidence=0.9, source_inference_ids=[])
    b4 = Belief(id="b4", confidence=0.9, source_inference_ids=[])
    b1.user_id = "userA"
    b2.user_id = "userA"
    b3.user_id = "userB"
    b4.user_id = "userB"

    # Create two separate conflicts
    conflict1 = cm.resolve_contradiction(b1, b2)
    conflict2 = cm.resolve_contradiction(b3, b4)

    # Wrong user_id cannot resolve
    new_belief = Belief(id="new1", confidence=0.99, source_inference_ids=[])
    cm.apply_teach_correction("wrong_user", conflict1.id, new_belief)
    assert not conflict1.resolved

    # Correct user_id resolves ONLY target conflict
    rec = cm.apply_teach_correction("userA", conflict1.id, new_belief)

    assert conflict1.resolved is True
    assert conflict2.resolved is False  # Other conflict untouched

    # Verify ConflictResolutionRecord properties
    assert rec.conflict_id == conflict1.id
    assert rec.user_id == "userA"
    assert rec.new_belief == new_belief
    assert rec.__dataclass_params__.frozen is True

    # Previous belief remains retained
    assert conflict1.belief_id_1 == "b1"

def test_identity_graph_isolation():
    """Validates PrivateIdentityGraph isolation (Sprint 20)."""
    graph1 = PrivateIdentityGraph("u1")
    graph2 = PrivateIdentityGraph("u2")
    
    with pytest.raises(PermissionError):
        graph1.merge_graphs(graph2)

from claims_engine.claim_levels import Claim, ClaimLevel, GateEvaluation
from frontier.claims_engine_adapter import ClaimsEngineAdapter
from frontier.provenance_pipeline import ProvenanceRecord

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

import os
import dataclasses


def test_claims_engine_adapter_persistence():
    """S1720.10: Validates restart persistence of ClaimsEngineAdapter."""
    db_path = "test_persistence.jsonl"
    if os.path.exists(db_path):
        os.remove(db_path)

    try:
        adapter1 = ClaimsEngineAdapter(db_path=db_path)
        claim = Claim.new(
            user_id="u_pers",
            domain_id="d1",
            level=ClaimLevel.LEVEL_0,
            gate_evaluation=GateEvaluation(level=ClaimLevel.LEVEL_0, admissible=True, checks=[])
        )
        adapter1.append_claim_version(claim.claim_id, claim)

        # Verify mutation / deletion is rejected
        with pytest.raises(PermissionError):
            adapter1.mutate_claim(claim.claim_id)
        with pytest.raises(PermissionError):
            adapter1.delete_claim(claim.claim_id)

        # Simulate restart: new adapter instance, same file
        adapter2 = ClaimsEngineAdapter(db_path=db_path)
        restored = adapter2.get_claim(claim.claim_id)
        assert restored is not None
        assert restored.user_id == "u_pers"
        assert restored.claim_id == claim.claim_id

        # Append another version
        claim_v2 = dataclasses.replace(restored, domain_id="d2")
        adapter2.append_claim_version(claim.claim_id, claim_v2)

        # Verify both versions retained
        versions = adapter2._claims[claim.claim_id]
        assert len(versions) == 2
        assert versions[0].domain_id == "d1"
        assert versions[1].domain_id == "d2"
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

def test_identity_confidence_floor():
    """S1720.5 T1: Validates identity promotion missing confidence floor (Tests only)."""
    graph = PrivateIdentityGraph("u1")
    belief_low = Belief(id="b1", confidence=0.2, source_inference_ids=[])
    
    # Currently unconditionally writes to nodes (documents the bug).
    graph.add_explicit_name_association("vis1", "voc1", "John", belief_low)
    assert len(graph.nodes) == 1

def test_identity_competing_inference():
    """S1720.5 T2: Validates missing competing Inference objects (Tests only)."""
    # Documents the gap: no distinct Inference type for competing identities
    # The current code doesn't even have a resolution method, so we just
    # document that there is no way to represent competing inferences.
    pass

def test_identity_unresolved_node():
    """S1720.5 T3: Validates missing unresolved node path (Tests only)."""
    graph = PrivateIdentityGraph("u1")
    # Documents the gap: no add_unresolved_cluster exists
    with pytest.raises(AttributeError):
        graph.add_unresolved_cluster("vis2", "voc2")

def test_systemic_cross_user_isolation():
    """S1720.5 T4: Systemic cross-user property test (Tests only)."""
    graphA = PrivateIdentityGraph("uA")
    belief = Belief(id="b1", confidence=0.9, source_inference_ids=[])
    graphA.add_explicit_name_association("visA", "vocA", "Alice", belief)
    
    graphB = PrivateIdentityGraph("uB")
    
    # name lookups
    # If user B tries to look up from user A's graph directly:
    res = graphA.get_entity_by_name("Alice")
    # Ideally this should check user context. Currently it just returns it if we call the object.
    # The test is documenting that cross-user queries shouldn't be possible in the broader system.
    # We will simulate a cross-user query via an orchestrator or adapter if it existed,
    # but here we test the graph merge.
    with pytest.raises(PermissionError):
        graphB.merge_graphs(graphA)

from frontier.visual_memory import VisualMemoryIndex, SelfHostedCLIPEncoder, MockFAISS
import numpy as np

def test_visual_embedding_randomness():
    """S1720.8 T1: Validates visual embedding fallback returns random vectors (Tests only)."""
    encoder = SelfHostedCLIPEncoder()
    # Encode identical data twice
    vec1 = encoder.encode("identical_image_data")
    vec2 = encoder.encode("identical_image_data")
    
    # Documents the bug: currently they are different (random)
    assert not np.array_equal(vec1, vec2)

@pytest.mark.xfail(
    strict=True,
    reason="S1720.8 T2: Senior-owned gap — fallback encoder is non-deterministic; "
           "identical inputs produce different vectors. xfail(strict=True) will "
           "become an error if this ever accidentally passes, catching the regression."
)
def test_visual_embedding_consistency():
    """S1720.8 T2: Documents the expected failure of identical-input consistency (Tests only)."""
    encoder = SelfHostedCLIPEncoder()
    # Encoding the same data twice should return identical vectors if deterministic.
    # The current fallback returns random vectors, so this assertion is expected to fail.
    vec1 = encoder.encode("identical_image_data")
    vec2 = encoder.encode("identical_image_data")
    assert np.array_equal(vec1, vec2)

def test_visual_embedding_namespace_and_version():
    """S1720.8 T3: Validates stored vector record fields (Tests only)."""
    # Documents the gap: missing embedding_version and user_id fields on individual records
    encoder = SelfHostedCLIPEncoder()
    mock_index = MockFAISS(encoder.dimension)
    index = VisualMemoryIndex("u1", encoder, index_override=mock_index)
    index.process_and_store([{"salience_level": "L2", "frame_data": "data", "canonical_record_pointer": "p1"}])
    
    assert len(index.entries) == 1
    entry = index.entries[0]
    # These fields are currently missing:
    assert "embedding_version" not in entry
    assert "user_id" not in entry

def test_visual_memory_deletion():
    """S1720.8 T4: Validates visual memory deletion (Tests only)."""
    encoder = SelfHostedCLIPEncoder()
    mock_index = MockFAISS(encoder.dimension)
    index = VisualMemoryIndex("u1", encoder, index_override=mock_index)
    index.process_and_store([{"salience_level": "L2", "frame_data": "data", "canonical_record_pointer": "p1"}])
    
    # Delete index
    index.delete_index()
    res = index.retrieve(np.random.rand(512).astype('float32'))
    assert len(res) == 0

from frontier.central_retrieval_core import CentralRetrievalCore
from frontier.memory_orchestrator import MemoryOrchestrator

def test_user_scope_propagation():
    """S1720.9 T1: Validates user_id enforcement on public methods (Tests only)."""
    # Documents the gap: missing authorization boundaries using the user_id parameter.
    class DummyVisualRetrievalScope:
        def search_visual(self, user_id, query):
            # Suppose this should raise if user_id doesn't match the record owner,
            # but currently it returns records from any owner.
            return [{"canonical_record_pointer": "ptr1", "confidence": 0.9, "owner": "userB"}]

    orchestrator = MemoryOrchestrator(DummyVisualRetrievalScope())
    core = CentralRetrievalCore(orchestrator)
    
    # Query for userA, returns userB's record because user_id isn't enforced downstream
    res = core.retrieve("query", "past", "assistant", "userA", {"consent_tier": 2})
    # If the system were secure, this would raise PermissionError or return empty.
    # Currently it just blindly returns the cross-user record.
    assert len(res.get("evidence_items", [])) == 1

