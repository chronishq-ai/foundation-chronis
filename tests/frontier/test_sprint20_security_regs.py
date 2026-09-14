import pytest
import numpy as np

from frontier.visual_memory import VisualMemoryIndex, DeterministicTestEncoder, MockFAISS
from frontier.multimodal_assistant import MultimodalAssistant
from frontier.explainability import ExplainabilityAPI
from frontier.identity_graph import PrivateIdentityGraph
from frontier.provenance_pipeline import Belief
from frontier.interfaces.policy import MockPolicyEngine

# --- RT-02: FAISS Sentinel Negative Index ---
def test_faiss_sentinel_negative_index():
    encoder = DeterministicTestEncoder()
    # A mock that returns negative indices to simulate FAISS -1 sentinel
    class NegativeMockFAISS(MockFAISS):
        def search(self, query: np.ndarray, k: int):
            # return distance and indices
            return np.array([[0.1, 0.2]]), np.array([[-1, -2]])
            
    vmi = VisualMemoryIndex("u1", encoder, index_override=NegativeMockFAISS(encoder.dimension))
    vmi.entries = [{"user_id": "u1", "canonical_record_pointer": "p1"}, {"user_id": "u1", "canonical_record_pointer": "p2"}]
    
    results = vmi.retrieve(np.zeros(512), k=2)
    assert len(results) == 0, "Retrieve should drop negative indices, not wrap around via Python list indexing."

# --- RT-03: Cosine Similarity Analytic Cases ---
def test_cosine_similarity_analytic_cases():
    # We can test this through MultimodalAssistant's live context
    class DummyIndex:
        def __init__(self, dist):
            self.dist = dist
        def retrieve(self, emb, k):
            return [{"user_id": "u1", "ann_distance": self.dist}]
            
    assistant = MultimodalAssistant(
        retrieval_core=None, 
        llm_provider=None, 
        policy_engine=MockPolicyEngine(),
        visual_index_provider=lambda uid: DummyIndex(0.0) # will override inside test
    )
    
    test_cases = [
        (0.0, 1.0),
        (0.5, 0.75),
        (1.0, 0.5),
        (2.0, 0.0)
    ]
    
    for dist, expected_sim in test_cases:
        assistant.visual_index_provider = lambda uid: DummyIndex(dist)
        res = assistant._handle_live_context("u1", np.zeros(512))
        
        # If similarity < 0.75, it returns no_confident_match but includes the computed similarity
        if expected_sim < 0.75:
            assert res["status"] == "no_confident_match"
            assert np.isclose(res["similarity"], expected_sim), f"Failed for dist {dist}"
        else:
            assert res["status"] == "match_found"
            assert np.isclose(res["similarity"], expected_sim), f"Failed for dist {dist}"

# --- RT-04: Invalid Input No Embedding ---
def test_invalid_input_no_embedding():
    encoder = DeterministicTestEncoder()
    # The fix block invalid types by raising TypeError, not returning zero vector
    for bad_input in [None, 42, {}, object()]:
        with pytest.raises(TypeError):
            encoder.encode(bad_input)

# --- RT-09: Explainability None/Empty Owner ---
def test_explainability_none_owner_denied():
    class MockClaimsStore:
        def get_claim(self, cid):
            # Claim with None owner
            return Belief(id=cid, confidence=0.9, source_inference_ids=[], user_id=None)
            
    api = ExplainabilityAPI(layer0=None, mirror=None, claims_store=MockClaimsStore())
    res = api.explain("c1", "u1")
    assert "error" in res
    assert "belongs to a different user" in res["error"] or "Access denied" in res["error"]

def test_explainability_empty_owner_denied():
    class MockClaimsStore:
        def get_claim(self, cid):
            # Claim with "" owner
            return Belief(id=cid, confidence=0.9, source_inference_ids=[], user_id="")
            
    api = ExplainabilityAPI(layer0=None, mirror=None, claims_store=MockClaimsStore())
    res = api.explain("c1", "u1")
    assert "error" in res
    assert "belongs to a different user" in res["error"] or "Access denied" in res["error"]

# --- RT-10: Identity Graph None/Empty Requester ---
def test_identity_graph_none_requester_denied():
    graph = PrivateIdentityGraph("u1")
    with pytest.raises(PermissionError):
        graph.get_entity_by_name("Alice", requesting_user_id=None)
    with pytest.raises(PermissionError):
        graph.get_unresolved_nodes(requesting_user_id=None)

def test_identity_graph_empty_requester_denied():
    graph = PrivateIdentityGraph("u1")
    with pytest.raises(PermissionError):
        graph.get_entity_by_name("Alice", requesting_user_id="")
    with pytest.raises(PermissionError):
        graph.get_unresolved_nodes(requesting_user_id="")

# --- CRC Bypass: Live Context Enforces Ownership ---
def test_live_context_enforces_ownership():
    # Misconfigured visual index returns another user's data
    class LeakyIndex:
        def retrieve(self, emb, k):
            return [{"user_id": "u2", "ann_distance": 0.1, "canonical_record_pointer": "p1"}]
            
    assistant = MultimodalAssistant(
        retrieval_core=None, 
        llm_provider=None, 
        policy_engine=MockPolicyEngine(),
        visual_index_provider=lambda uid: LeakyIndex()
    )
    
    res = assistant._handle_live_context("u1", np.zeros(512))
    assert res.get("status") == "error"
    assert "CROSS-USER VIOLATION" in res.get("error", "")
