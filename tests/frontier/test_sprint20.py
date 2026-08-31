"""
tests/frontier/test_sprint20.py

Sprint 20 tests — S1720.5, S1720.6, S1720.7, S1720.8, S1720.9, S1720.10,
                  R2-F20.1 through R2-F20.7
"""

import dataclasses
import inspect
import os

import numpy as np
import pytest

from frontier.provenance_pipeline import (
    ProvenanceManager, ProvenanceStore,
    Belief, Observation, Feature, Inference, ClaimLink,
)
from frontier.conflict_resolution import ConflictManager, ConflictNotFoundError
from frontier.identity_graph import PrivateIdentityGraph, CONFIDENCE_FLOOR
from frontier.interfaces.claims_store import ClaimsStoreProvider
from frontier.interfaces.layer0 import MockLayer0Storage
from frontier.interfaces.mirror import MockMirrorProvider
from frontier.visual_memory import VisualMemoryIndex, SelfHostedCLIPEncoder, MockFAISS, DeterministicTestEncoder


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class MockClaimsStore(ClaimsStoreProvider):
    def __init__(self):
        self.claims = {}
        self.statuses = {}

    def get_claim(self, claim_id):
        return self.claims.get(claim_id)

    def get_claim_status(self, claim_id):
        if claim_id not in self.claims:
            return None
        return self.statuses.get(claim_id, "SURFACED")

    def update_claim_status(self, claim_id, new_status):
        if claim_id not in self.claims:
            raise KeyError(f"Claim '{claim_id}' not found")
        self.statuses[claim_id] = new_status

    def apply_status_updates_batch(self, updates):
        missing = [cid for cid, _ in updates if cid not in self.claims]
        if missing:
            raise KeyError(f"Missing claims: {missing}")
        for cid, st in updates:
            self.statuses[cid] = st

    def append_claim_version(self, claim_id, obj):
        self.claims[claim_id] = obj

    def iter_claims(self):
        yield from self.claims.values()


# ---------------------------------------------------------------------------
# S1720.7 / R2-F20.2 — Provenance pipeline (canonical IDs)
# ---------------------------------------------------------------------------

def test_provenance_pipeline_confidence_floor():
    """Below-floor belief must not promote to claim."""
    pm = ProvenanceManager()
    belief_low = Belief(id="b1", confidence=0.5, source_inference_ids=["i1"], user_id="u1")
    assert pm.promote_to_claim(belief_low, "identity_match", 1, "test") is None


def test_provenance_pipeline_promotes_above_floor():
    """Above-floor belief must promote and return a ProvenanceRecord."""
    pm = ProvenanceManager()
    belief_high = Belief(id="b2", confidence=0.9, source_inference_ids=["i2"], user_id="u1")
    claim = pm.promote_to_claim(belief_high, "identity_match", 1, "test")
    assert claim is not None
    assert claim.claim_id == "claim_b2"


def test_provenance_canonical_walk():
    """R2-F20.2: explain_retrofitted must reconstruct the full chain from store."""
    store = ProvenanceStore()
    from datetime import datetime

    # Build a full chain: obs → feat → inf → belief → claim
    obs = Observation(observation_id="obs1", user_id="u1", raw_data="raw", timestamp=datetime.now())
    feat = Feature(feature_id="feat1", user_id="u1", source_observation_ids=["obs1"], derived_representation="dr")
    inf = Inference(inference_id="inf1", user_id="u1", source_feature_ids=["feat1"], candidate_match="Alice")
    belief = Belief(id="bel1", confidence=0.9, source_inference_ids=["inf1"], user_id="u1")

    store.store_observation(obs)
    store.store_feature(feat)
    store.store_inference(inf)
    store.store_belief(belief)
    store.link_claim("claim_bel1", ["bel1"], "u1")

    chain = store.reconstruct_chain("claim_bel1", "u1")
    assert chain.get("claim_id") == "claim_bel1"
    beliefs = chain.get("beliefs", [])
    assert len(beliefs) == 1
    assert beliefs[0]["belief_id"] == "bel1"


def test_provenance_cross_user_edge_rejected():
    """R2-F20.2: cross-user provenance edge must raise PermissionError."""
    store = ProvenanceStore()
    from datetime import datetime

    obs = Observation(observation_id="obs_x", user_id="user_B", raw_data="r", timestamp=datetime.now())
    feat = Feature(feature_id="feat_x", user_id="user_A", source_observation_ids=["obs_x"], derived_representation="dr")
    inf = Inference(inference_id="inf_x", user_id="user_A", source_feature_ids=["feat_x"], candidate_match=None)
    belief = Belief(id="bel_x", confidence=0.9, source_inference_ids=["inf_x"], user_id="user_A")

    store.store_observation(obs)
    store.store_feature(feat)
    store.store_inference(inf)
    store.store_belief(belief)
    store.link_claim("claim_x", ["bel_x"], "user_A")

    # Requesting as user_A → should hit user_B's observation and raise
    with pytest.raises(PermissionError):
        store.reconstruct_chain("claim_x", "user_A")


def test_provenance_missing_node_structured_error():
    """R2-F20.2: missing node returns structured error dict with defined schema."""
    store = ProvenanceStore()
    belief = Belief(id="bel_orphan", confidence=0.9, source_inference_ids=["inf_missing"], user_id="u1")
    store.store_belief(belief)
    store.link_claim("claim_orphan", ["bel_orphan"], "u1")

    chain = store.reconstruct_chain("claim_orphan", "u1")
    beliefs = chain.get("beliefs", [])
    assert len(beliefs) == 1
    inf_entries = beliefs[0].get("inferences", [])
    assert len(inf_entries) == 1
    # Should be a structured error dict
    err = inf_entries[0]
    assert err.get("status") == "error"
    assert err.get("error_type") == "MISSING_PROVENANCE_NODE"
    assert err.get("node_type") == "Inference"
    assert err.get("node_id") == "inf_missing"


def test_provenance_explain_no_caller_data():
    """R2-F20.1: explain_retrofitted must not accept claim_data parameter."""
    pm = ProvenanceManager()
    sig = inspect.signature(pm.explain_retrofitted)
    assert "claim_data" not in sig.parameters


# ---------------------------------------------------------------------------
# S1720.6 / R2-F20.3 — Conflict resolution
# ---------------------------------------------------------------------------

def test_conflict_resolution_creates_record():
    """Validates conflicts are retained, not deleted (Sprint 20)."""
    store = MockClaimsStore()
    cm = ConflictManager(store, provenance_store=ProvenanceStore())
    b1 = Belief(id="b1", confidence=0.9, source_inference_ids=[], user_id="userA")
    b2 = Belief(id="b2", confidence=0.9, source_inference_ids=[], user_id="userA")

    conflict = cm.resolve_contradiction(b1, b2)
    assert conflict.belief_id_1 == "b1"
    assert conflict.belief_id_2 == "b2"


def test_conflict_nonexistent_id_raises_typed_error():
    """R2-F20.3: nonexistent conflict_id → ConflictNotFoundError (not silent no-op)."""
    store = MockClaimsStore()
    cm = ConflictManager(store, provenance_store=ProvenanceStore())
    new_belief = Belief(id="new1", confidence=0.99, source_inference_ids=[], user_id="userA")

    with pytest.raises(ConflictNotFoundError):
        cm.apply_teach_correction("userA", "nonexistent_conflict_id", new_belief)


def test_conflict_wrong_user_raises():
    """R2-F20.3: cross-user correction → PermissionError."""
    store = MockClaimsStore()
    cm = ConflictManager(store, provenance_store=ProvenanceStore())
    b1 = Belief(id="b1", confidence=0.9, source_inference_ids=[], user_id="userA")
    b2 = Belief(id="b2", confidence=0.9, source_inference_ids=[], user_id="userA")
    conflict = cm.resolve_contradiction(b1, b2)

    new_belief = Belief(id="new1", confidence=0.99, source_inference_ids=[], user_id="userB")
    with pytest.raises(PermissionError):
        cm.apply_teach_correction("userB", conflict.id, new_belief)


def test_conflict_wrong_new_belief_user_raises():
    """R2-F20.3: new_belief.user_id mismatch → PermissionError."""
    store = MockClaimsStore()
    cm = ConflictManager(store, provenance_store=ProvenanceStore())
    b1 = Belief(id="b1", confidence=0.9, source_inference_ids=[], user_id="userA")
    b2 = Belief(id="b2", confidence=0.9, source_inference_ids=[], user_id="userA")
    conflict = cm.resolve_contradiction(b1, b2)

    # correct caller user_id but new_belief.user_id is different
    new_belief = Belief(id="new1", confidence=0.99, source_inference_ids=[], user_id="userB")
    with pytest.raises(PermissionError):
        cm.apply_teach_correction("userA", conflict.id, new_belief)


def test_conflict_resolution_targeted():
    """S1720.6: targeted apply_teach_correction resolves only target conflict."""
    store = MockClaimsStore()
    cm = ConflictManager(store, provenance_store=ProvenanceStore())
    b1 = Belief(id="b1", confidence=0.9, source_inference_ids=[], user_id="userA")
    b2 = Belief(id="b2", confidence=0.9, source_inference_ids=[], user_id="userA")
    b3 = Belief(id="b3", confidence=0.9, source_inference_ids=[], user_id="userB")
    b4 = Belief(id="b4", confidence=0.9, source_inference_ids=[], user_id="userB")

    conflict1 = cm.resolve_contradiction(b1, b2)
    conflict2 = cm.resolve_contradiction(b3, b4)

    # Wrong user cannot resolve conflict1
    new_belief = Belief(id="new1", confidence=0.99, source_inference_ids=[], user_id="userA")
    with pytest.raises(PermissionError):
        cm.apply_teach_correction("wrong_user", conflict1.id, new_belief)
    assert not conflict1.resolved

    # Correct user resolves ONLY conflict1
    rec = cm.apply_teach_correction("userA", conflict1.id, new_belief)
    assert conflict1.resolved is True
    assert conflict2.resolved is False

    assert rec.conflict_id == conflict1.id
    assert rec.user_id == "userA"
    assert rec.new_belief == new_belief
    assert rec.__dataclass_params__.frozen is True
    assert conflict1.belief_id_1 == "b1"   # old belief preserved


def test_conflict_idempotency():
    """R2-F20.3: repeated correction with same belief returns existing record."""
    store = MockClaimsStore()
    cm = ConflictManager(store, provenance_store=ProvenanceStore())
    b1 = Belief(id="b1", confidence=0.9, source_inference_ids=[], user_id="u1")
    b2 = Belief(id="b2", confidence=0.9, source_inference_ids=[], user_id="u1")
    conflict = cm.resolve_contradiction(b1, b2)

    new_b = Belief(id="new1", confidence=0.99, source_inference_ids=[], user_id="u1")
    rec1 = cm.apply_teach_correction("u1", conflict.id, new_b)
    rec2 = cm.apply_teach_correction("u1", conflict.id, new_b)
    assert rec1 is rec2   # idempotent — returns same record


# ---------------------------------------------------------------------------
# S1720.5 / R2-F20.4 — Identity Graph
# ---------------------------------------------------------------------------

def test_identity_graph_isolation():
    """Validates PrivateIdentityGraph cross-user isolation."""
    graph1 = PrivateIdentityGraph("u1")
    graph2 = PrivateIdentityGraph("u2")

    with pytest.raises(PermissionError):
        graph1.merge_graphs(graph2)


def test_identity_confidence_floor_enforced():
    """S1720.5 T1: Low-confidence belief must NOT create an IdentityNode."""
    graph = PrivateIdentityGraph("u1")
    belief_low = Belief(id="b1", confidence=0.2, source_inference_ids=[], user_id="u1")

    with pytest.raises(ValueError, match="below the required floor"):
        graph.add_explicit_name_association("vis1", "voc1", "John", belief_low)

    assert len(graph.nodes) == 0, "No IdentityNode should have been created"


def test_identity_unresolved_node_exists():
    """S1720.5 T3: add_unresolved_cluster now works and creates node with name=None."""
    graph = PrivateIdentityGraph("u1")
    node = graph.add_unresolved_cluster("vis2", "voc2")
    assert node is not None
    assert node.name is None
    assert len(graph.nodes) == 1


def test_identity_competing_candidates_preserved():
    """S1720.5 T2: Competing inference candidates are preserved, not collapsed."""
    graph = PrivateIdentityGraph("u1")
    inf = graph.add_inference(
        visual_cluster_id="vis1",
        voice_cluster_id="voc1",
        candidates=["Alice", "Bob"],
        confidence=0.50,   # below floor → inference only
        user_id="u1",
    )
    assert len(inf.candidates) == 2
    assert "Alice" in inf.candidates
    assert "Bob" in inf.candidates
    # Not promoted to IdentityNode
    assert len(graph.nodes) == 0


def test_identity_get_entity_cross_user_rejected():
    """S1720.5 T4: Cross-user identity read raises PermissionError."""
    graphA = PrivateIdentityGraph("uA")
    belief = Belief(id="b1", confidence=0.9, source_inference_ids=[], user_id="uA")
    graphA.add_explicit_name_association("visA", "vocA", "Alice", belief)

    with pytest.raises(PermissionError):
        graphA.get_entity_by_name("Alice", requesting_user_id="uB")


def test_systemic_cross_user_isolation():
    """S1720.5 T4: merge_graphs across users raises PermissionError."""
    graphA = PrivateIdentityGraph("uA")
    graphB = PrivateIdentityGraph("uB")

    with pytest.raises(PermissionError):
        graphB.merge_graphs(graphA)


# ---------------------------------------------------------------------------
# S1720.8 / R2-F20.5 — Visual encoder (BLOCKED ticket)
# ---------------------------------------------------------------------------




def test_visual_embedding_consistency():
    """S1720.8: Identical inputs must produce identical vectors."""
    encoder = SelfHostedCLIPEncoder()
    vec1 = encoder.encode("identical_image_data")
    vec2 = encoder.encode("identical_image_data")
    assert np.array_equal(vec1, vec2)


def test_deterministic_test_encoder_is_deterministic():
    """DeterministicTestEncoder (unit tests only) must be deterministic."""
    encoder = DeterministicTestEncoder()
    vec1 = encoder.encode("same_data")
    vec2 = encoder.encode("same_data")
    assert np.array_equal(vec1, vec2)

    vec3 = encoder.encode("different_data")
    assert not np.array_equal(vec1, vec3)


def test_visual_embedding_namespace_and_version():
    """S1720.8 T3: Stored entries must now include user_id and embedding_version."""
    encoder = DeterministicTestEncoder()
    mock_index = MockFAISS(encoder.dimension)
    index = VisualMemoryIndex("u1", encoder, index_override=mock_index)
    index.process_and_store([{
        "salience_level": "L2",
        "frame_data": "data",
        "canonical_record_pointer": "p1",
    }])

    assert len(index.entries) == 1
    entry = index.entries[0]
    assert entry.get("user_id") == "u1"
    assert "embedding_version" in entry
    assert "encoder_model_id" in entry


def test_visual_memory_deletion():
    """S1720.8 T4: Delete index removes all entries."""
    encoder = DeterministicTestEncoder()
    mock_index = MockFAISS(encoder.dimension)
    index = VisualMemoryIndex("u1", encoder, index_override=mock_index)
    index.process_and_store([{
        "salience_level": "L2",
        "frame_data": "data",
        "canonical_record_pointer": "p1",
    }])

    index.delete_index()
    res = index.retrieve(np.zeros(512, dtype="float32"))
    assert len(res) == 0


# ---------------------------------------------------------------------------
# S1720.9 — user_id propagation through CRC (cross-user rejected)
# ---------------------------------------------------------------------------

from frontier.central_retrieval_core import CentralRetrievalCore, CrossUserEvidenceError
from frontier.memory_orchestrator import MemoryOrchestrator
from frontier.interfaces.policy import MockPolicyEngine


def test_user_scope_cross_user_raises():
    """S1720.9: CRC raises CrossUserEvidenceError on cross-user evidence (not silent pass)."""
    class CrossUserScope:
        def search_visual(self, user_id, query):
            return [{"canonical_record_pointer": "ptr1", "confidence": 0.9, "owner_user_id": "userB"}]

    orchestrator = MemoryOrchestrator(CrossUserScope())
    core = CentralRetrievalCore(orchestrator, MockPolicyEngine())

    with pytest.raises(CrossUserEvidenceError):
        core.retrieve("query", "past", "assistant", "userA", {})


# ---------------------------------------------------------------------------
# S1720.10 / R2-F20.6 / R2-F20.7 — Claims persistence
# ---------------------------------------------------------------------------

from frontier.claims_engine_adapter import ClaimsEngineAdapter
from claims_engine.claim_levels import Claim, ClaimLevel, GateEvaluation


def test_claims_engine_adapter_persistence():
    """S1720.10: Claims survive restart (JSON, not pickle)."""
    db_path = "test_persistence_s1720.jsonl"
    if os.path.exists(db_path):
        os.remove(db_path)

    try:
        adapter1 = ClaimsEngineAdapter(db_path=db_path)
        claim = Claim.new(
            user_id="u_pers",
            domain_id="d1",
            level=ClaimLevel.LEVEL_0,
            gate_evaluation=GateEvaluation(level=ClaimLevel.LEVEL_0, admissible=True, checks=[]),
        )
        adapter1.append_claim_version(claim.claim_id, claim)

        with pytest.raises(PermissionError):
            adapter1.mutate_claim(claim.claim_id)
        with pytest.raises(PermissionError):
            adapter1.delete_claim(claim.claim_id)

        # Simulate restart
        adapter2 = ClaimsEngineAdapter(db_path=db_path)
        restored = adapter2.get_claim(claim.claim_id)
        assert restored is not None
        assert restored.user_id == "u_pers"
        assert restored.claim_id == claim.claim_id

        claim_v2 = dataclasses.replace(restored, domain_id="d2")
        adapter2.append_claim_version(claim.claim_id, claim_v2)

        versions = adapter2._claims[claim.claim_id]
        assert len(versions) == 2
        assert versions[0].domain_id == "d1"
        assert versions[1].domain_id == "d2"
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_claims_no_pickle_in_store():
    """R2-F20.6: JSONL store must contain no pickle/base64 binary."""
    db_path = "test_no_pickle.jsonl"
    if os.path.exists(db_path):
        os.remove(db_path)

    try:
        adapter = ClaimsEngineAdapter(db_path=db_path)
        claim = Claim.new(
            user_id="u1",
            domain_id="d1",
            level=ClaimLevel.LEVEL_0,
            gate_evaluation=GateEvaluation(level=ClaimLevel.LEVEL_0, admissible=True, checks=[]),
        )
        adapter.append_claim_version(claim.claim_id, claim)

        content = open(db_path, "r", encoding="utf-8").read()
        assert "pickle" not in content.lower()
        # base64 strings (gAAS... pattern) shouldn't appear
        assert "gAAS" not in content
        # All lines must be valid JSON
        import json
        for line in content.splitlines():
            if line.strip():
                json.loads(line)  # must not raise
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_claims_status_unknown_returns_none():
    """R2-F20.7: get_claim_status for unknown ID must return None, not 'SURFACED'."""
    adapter = ClaimsEngineAdapter(db_path="nonexistent_status_test.jsonl")
    result = adapter.get_claim_status("totally_unknown_claim_id_xyz")
    assert result is None, f"Expected None for unknown claim, got '{result}'"


def test_claims_update_nonexistent_raises():
    """R2-F20.7: update_claim_status on nonexistent ID must raise KeyError."""
    adapter = ClaimsEngineAdapter(db_path="nonexistent_update_test.jsonl")
    with pytest.raises(KeyError):
        adapter.update_claim_status("nonexistent_id", "UNCLEAR")


# ---------------------------------------------------------------------------
# Integration: ClaimsEngine → ConflictManager
# ---------------------------------------------------------------------------

def test_claims_engine_integration():
    """Integration: ClaimsEngine → ClaimsStoreProvider → ConflictManager."""
    from frontier.provenance_pipeline import ProvenanceRecord

    adapter = ClaimsEngineAdapter()
    claim = Claim.new(
        user_id="u1",
        domain_id="d1",
        level=ClaimLevel.LEVEL_0,
        gate_evaluation=GateEvaluation(level=ClaimLevel.LEVEL_0, admissible=True, checks=[]),
    )
    adapter._claims[claim.claim_id] = [claim]
    adapter._claim_statuses[claim.claim_id] = "SURFACED"

    cm = ConflictManager(adapter)
    b1 = Belief(id="b1", confidence=0.9, source_inference_ids=[], user_id="u1")
    b2 = Belief(id="b2", confidence=0.9, source_inference_ids=[], user_id="u1")

    conflict = cm.resolve_contradiction(b1, b2)
    pr = ProvenanceRecord(claim_id=claim.claim_id, source_belief_ids=["b1"])
    cm.downgrade_dependent_claims(conflict, [pr])

    assert adapter.get_claim_status(claim.claim_id) == "UNCLEAR"
    assert pr.status == "UNCLEAR"


# ---------------------------------------------------------------------------
# Slow benchmark: 10k cross-user conflict attempts
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cross_user_conflict_10k():
    """R2-F20.3 (slow): 10,000 cross-user correction attempts → 100% rejection."""
    failures = 0
    store = MockClaimsStore()
    prov_store = ProvenanceStore()
    cm = ConflictManager(store, provenance_store=prov_store)
    for i in range(10_000):
        b1 = Belief(id=f"b{i}a", confidence=0.9, source_inference_ids=[], user_id="userA")
        b2 = Belief(id=f"b{i}b", confidence=0.9, source_inference_ids=[], user_id="userA")
        conflict = cm.resolve_contradiction(b1, b2)

        bad_belief = Belief(id=f"bad{i}", confidence=0.99, source_inference_ids=[], user_id="userB")
        try:
            cm.apply_teach_correction("userB", conflict.id, bad_belief)
            failures += 1   # should have raised
        except PermissionError:
            pass

    assert failures == 0, (
        f"SECURITY FAILURE: {failures}/10,000 cross-user corrections were not rejected"
    )


# ---------------------------------------------------------------------------
# Conflict resolution atomicity — rollback test
# ---------------------------------------------------------------------------

def test_conflict_resolution_atomic_rollback():
    """
    R2-F20.3 / Feedback #4: If dependent-claim update fails, conflict must
    NOT be marked as resolved (consistent state over partial progress).
    """
    class FailingClaimsStore(MockClaimsStore):
        def apply_status_updates_batch(self, updates):
            raise RuntimeError("Simulated claim-store write failure")
        def update_claim_status(self, claim_id, new_status):
            raise RuntimeError("Simulated claim-store write failure")

    failing_store = FailingClaimsStore()
    prov_store = ProvenanceStore()
    prov_store.link_claim("claim_dep_1", ["ba"], "u1")

    cm = ConflictManager(failing_store, provenance_store=prov_store)

    b1 = Belief(id="ba", confidence=0.9, source_inference_ids=[], user_id="u1")
    b2 = Belief(id="bb", confidence=0.9, source_inference_ids=[], user_id="u1")
    conflict = cm.resolve_contradiction(b1, b2)

    new_belief = Belief(id="new_ba", confidence=0.99, source_inference_ids=[], user_id="u1")
    with pytest.raises(RuntimeError, match="UNRESOLVED"):
        cm.apply_teach_correction("u1", conflict.id, new_belief)

    # Conflict must remain unresolved because batch claim update failed
    assert conflict.resolved is False


# ---------------------------------------------------------------------------
# Temporal retrieval boundary tests
# ---------------------------------------------------------------------------

from frontier.retrieval import RetrievalAPI
from frontier.visual_memory import VisualMemoryIndex, MockFAISS, DeterministicTestEncoder
from frontier.interfaces.layer0 import MockLayer0Storage
from datetime import datetime, timedelta


def _make_temporal_api(entries):
    encoder = DeterministicTestEncoder()
    mock_index = MockFAISS(encoder.dimension)
    index = VisualMemoryIndex("u1", encoder, index_override=mock_index)
    index.entries = entries
    return RetrievalAPI(
        visual_indexes={"u1": index},
        layer0=MockLayer0Storage(),
        encoder=encoder,
    )


def test_temporal_boundary_inclusive_start():
    """Timestamp == start → included."""
    start = datetime(2024, 6, 1, 10, 0, 0)
    end = datetime(2024, 6, 1, 12, 0, 0)
    api = _make_temporal_api([{
        "user_id": "u1", "canonical_record_pointer": "p_start",
        "timestamp_ntp": start, "salience_level": "L2",
        "embedding_version": 1, "encoder_model_id": "test",
    }])
    result = api.get_context("u1", (start, end), "past")
    assert any(r["canonical_record_pointer"] == "p_start" for r in result)


def test_temporal_boundary_inclusive_end():
    """Timestamp == end → included."""
    start = datetime(2024, 6, 1, 10, 0, 0)
    end = datetime(2024, 6, 1, 12, 0, 0)
    api = _make_temporal_api([{
        "user_id": "u1", "canonical_record_pointer": "p_end",
        "timestamp_ntp": end, "salience_level": "L2",
        "embedding_version": 1, "encoder_model_id": "test",
    }])
    result = api.get_context("u1", (start, end), "past")
    assert any(r["canonical_record_pointer"] == "p_end" for r in result)


def test_temporal_boundary_before_start_excluded():
    """Timestamp < start → excluded."""
    start = datetime(2024, 6, 1, 10, 0, 0)
    end = datetime(2024, 6, 1, 12, 0, 0)
    api = _make_temporal_api([{
        "user_id": "u1", "canonical_record_pointer": "p_before",
        "timestamp_ntp": start - timedelta(seconds=1), "salience_level": "L2",
        "embedding_version": 1, "encoder_model_id": "test",
    }])
    result = api.get_context("u1", (start, end), "past")
    assert len(result) == 0


def test_temporal_boundary_after_end_excluded():
    """Timestamp > end → excluded."""
    start = datetime(2024, 6, 1, 10, 0, 0)
    end = datetime(2024, 6, 1, 12, 0, 0)
    api = _make_temporal_api([{
        "user_id": "u1", "canonical_record_pointer": "p_after",
        "timestamp_ntp": end + timedelta(seconds=1), "salience_level": "L2",
        "embedding_version": 1, "encoder_model_id": "test",
    }])
    result = api.get_context("u1", (start, end), "past")
    assert len(result) == 0


def test_temporal_invalid_query_type():
    """Invalid query_type → ValueError."""
    api = _make_temporal_api([])
    with pytest.raises(ValueError, match="query_type must be"):
        api.get_context("u1", (datetime.now(), datetime.now()), "sideways")


# ---------------------------------------------------------------------------
# Contradiction detection: false-positive tests
# ---------------------------------------------------------------------------

from frontier.memory_orchestrator import MemoryOrchestrator


def test_contradiction_same_fact_different_modality_not_contradicted():
    """Same (subject, predicate, object) from different modalities → NOT a contradiction."""
    class SameFact:
        def search_visual(self, user_id, query):
            return [
                {"canonical_record_pointer": "p1", "confidence": 0.9,
                 "owner_user_id": user_id,
                 "assertion": {"subject": "person_1", "predicate": "wearing", "object": "red_shirt"}},
                {"canonical_record_pointer": "p2", "confidence": 0.85,
                 "owner_user_id": user_id,
                 "assertion": {"subject": "person_1", "predicate": "wearing", "object": "red_shirt"}},
            ]
    orch = MemoryOrchestrator(SameFact())
    res = orch.orchestrate("u1", "q", "past")
    assert len(res["contradictions"]) == 0


def test_contradiction_unrelated_facts_not_contradicted():
    """Different (subject, predicate) pairs → NOT a contradiction."""
    class UnrelatedFacts:
        def search_visual(self, user_id, query):
            return [
                {"canonical_record_pointer": "p1", "confidence": 0.9,
                 "owner_user_id": user_id,
                 "assertion": {"subject": "person_1", "predicate": "at_location", "object": "desk"}},
                {"canonical_record_pointer": "p2", "confidence": 0.8,
                 "owner_user_id": user_id,
                 "assertion": {"subject": "person_1", "predicate": "purchased", "object": "laptop"}},
            ]
    orch = MemoryOrchestrator(UnrelatedFacts())
    res = orch.orchestrate("u1", "q", "past")
    assert len(res["contradictions"]) == 0


def test_contradiction_same_source_duplicated_not_contradicted():
    """Same content_pointer and modality (duplicate) → NOT a contradiction."""
    class Duplicated:
        def search_visual(self, user_id, query):
            return [
                {"canonical_record_pointer": "p1", "confidence": 0.9,
                 "owner_user_id": user_id,
                 "assertion": {"subject": "s", "predicate": "p", "object": "o"}},
                {"canonical_record_pointer": "p1", "confidence": 0.9,  # same ptr
                 "owner_user_id": user_id,
                 "assertion": {"subject": "s", "predicate": "p", "object": "different"}},
            ]
    orch = MemoryOrchestrator(Duplicated())
    res = orch.orchestrate("u1", "q", "past")
    # Same content_pointer + no modality difference → not contradicted
    assert len(res["contradictions"]) == 0


# ---------------------------------------------------------------------------
# Overall confidence with all-low-confidence evidence
# ---------------------------------------------------------------------------

def test_overall_confidence_all_low():
    """When all items are below 0.5, overall_confidence must still be correct avg."""
    class AllLow:
        def search_visual(self, user_id, query):
            return [
                {"canonical_record_pointer": "p1", "confidence": 0.3, "owner_user_id": user_id},
                {"canonical_record_pointer": "p2", "confidence": 0.4, "owner_user_id": user_id},
            ]
    orch = MemoryOrchestrator(AllLow())
    res = orch.orchestrate("u1", "q", "past")
    # Both items flagged as low_confidence
    for item in res["evidence_items"]:
        assert item.get("low_confidence") is True
    # Overall = (0.3 + 0.4) / 2 = 0.35
    assert abs(res["overall_confidence"] - 0.35) < 0.01


def test_overall_confidence_empty_evidence():
    """Empty evidence → overall_confidence = 0.0, episode_window = (None, None)."""
    class NoResults:
        def search_visual(self, user_id, query):
            return []
    orch = MemoryOrchestrator(NoResults())
    res = orch.orchestrate("u1", "q", "past")
    assert res["overall_confidence"] == 0.0
    assert res["episode_window"] == (None, None)
    assert res["evidence_items"] == []


# ---------------------------------------------------------------------------
# Import consistency test
# ---------------------------------------------------------------------------

def test_import_consistency():
    """
    All production frontier files must use consistent import paths for
    interfaces (frontier.interfaces.X, not bare absolute 'interfaces.X').
    """
    import ast
    from pathlib import Path

    root = Path(__file__).parent.parent.parent / "src" / "frontier"
    violations = []

    for f in root.rglob("*.py"):
        if "__pycache__" in str(f):
            continue
        src = f.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                # node.level > 0 means it's a relative import (from .interfaces.X)
                # which is valid within the frontier package.
                # Only flag ABSOLUTE imports of bare 'interfaces.X'.
                if node.level == 0 and (
                    module.startswith("interfaces.") or module == "interfaces"
                ):
                    violations.append(
                        f"{f.name}:{node.lineno} — absolute bare import '{module}' "
                        f"should be 'frontier.interfaces...'"
                    )

    assert not violations, (
        "Import consistency violation(s):\n" + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Additional Sprint 20 tests: provenance scoping, batch atomicity, corruption
# ---------------------------------------------------------------------------

def test_provenance_find_claims_referencing():
    """Phase 2a: ProvenanceStore.find_claims_referencing returns only claims linking specified beliefs."""
    store = ProvenanceStore()
    store.link_claim(claim_id="c1", source_belief_ids=["b1", "b2"], user_id="u1")
    store.link_claim(claim_id="c2", source_belief_ids=["b3"], user_id="u1")
    store.link_claim(claim_id="c3", source_belief_ids=["b2", "b4"], user_id="u1")

    # Referencing b1 should return c1
    assert store.find_claims_referencing({"b1"}) == ["c1"]
    # Referencing b2 should return c1 and c3
    res_b2 = store.find_claims_referencing({"b2"})
    assert set(res_b2) == {"c1", "c3"}
    # Referencing unlinked b99 should return empty
    assert store.find_claims_referencing({"b99"}) == []


def test_conflict_only_dependent_claims_updated():
    """
    Phase 4b: Resolving a conflict must ONLY update claims whose provenance
    references the conflicting beliefs, preserving unrelated claims in the store.
    """
    db_path = "test_conflict_dependent_only.jsonl"
    if os.path.exists(db_path):
        os.remove(db_path)

    try:
        claims_store = ClaimsEngineAdapter(db_path=db_path)
        c1 = Claim.new(user_id="u1", domain_id="d1", level=ClaimLevel.LEVEL_0, gate_evaluation=GateEvaluation(level=ClaimLevel.LEVEL_0, admissible=True, checks=[]))
        c2_unrelated = Claim.new(user_id="u1", domain_id="d2", level=ClaimLevel.LEVEL_0, gate_evaluation=GateEvaluation(level=ClaimLevel.LEVEL_0, admissible=True, checks=[]))

        claims_store.append_claim_version(c1.claim_id, c1)
        claims_store.append_claim_version(c2_unrelated.claim_id, c2_unrelated)

        prov_store = ProvenanceStore()
        # c1 depends on b1 (conflicted); c2 depends on b_unrelated
        prov_store.link_claim(claim_id=c1.claim_id, source_belief_ids=["b_conflict_1"], user_id="u1")
        prov_store.link_claim(claim_id=c2_unrelated.claim_id, source_belief_ids=["b_unrelated"], user_id="u1")

        cm = ConflictManager(claims_store=claims_store)
        b1 = Belief(id="b_conflict_1", confidence=0.9, source_inference_ids=[], user_id="u1")
        b2 = Belief(id="b_conflict_2", confidence=0.9, source_inference_ids=[], user_id="u1")
        conflict = cm.resolve_contradiction(b1, b2)

        new_belief = Belief(id="b_resolved", confidence=0.99, source_inference_ids=[], user_id="u1")
        cm.apply_teach_correction("u1", conflict.id, new_belief, provenance_store=prov_store)

        # c1 should be UNCLEAR, but c2_unrelated must remain SURFACED
        assert claims_store.get_claim_status(c1.claim_id) == "UNCLEAR"
        assert claims_store.get_claim_status(c2_unrelated.claim_id) == "SURFACED"
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_conflict_atomic_batch_failure_rollback():
    """
    Phase 4b: If batch claim update fails, conflict must NOT be marked resolved.
    """
    prov_store = ProvenanceStore()
    prov_store.link_claim(claim_id="c_failing", source_belief_ids=["b_bad1"], user_id="u1")

    class FailingBatchStore(ClaimsStoreProvider):
        def get_claim(self, cid): return None
        def get_claim_status(self, cid): return None
        def update_claim_status(self, cid, status): pass
        def append_claim_version(self, cid, obj): pass
        def iter_claims(self): yield from []
        def apply_status_updates_batch(self, updates):
            raise RuntimeError("Database write error during batch update")

    cm = ConflictManager(claims_store=FailingBatchStore())
    b1 = Belief(id="b_bad1", confidence=0.9, source_inference_ids=[], user_id="u1")
    b2 = Belief(id="b_bad2", confidence=0.9, source_inference_ids=[], user_id="u1")
    conflict = cm.resolve_contradiction(b1, b2)

    new_belief = Belief(id="b_new", confidence=0.99, source_inference_ids=[], user_id="u1")
    with pytest.raises(RuntimeError, match="UNRESOLVED"):
        cm.apply_teach_correction("u1", conflict.id, new_belief, provenance_store=prov_store)

    assert not conflict.resolved, "Conflict must NOT be marked resolved when claim update fails"


def test_conflict_requires_provenance_store():
    """Phase 4b: ConflictManager fails closed if provenance_store is not provided."""
    claims_store = ClaimsEngineAdapter(db_path="test_no_prov.jsonl")
    cm = ConflictManager(claims_store=claims_store)
    b1 = Belief(id="b1", confidence=0.9, source_inference_ids=[], user_id="u1")
    b2 = Belief(id="b2", confidence=0.9, source_inference_ids=[], user_id="u1")
    conflict = cm.resolve_contradiction(b1, b2)

    new_belief = Belief(id="b_new", confidence=0.99, source_inference_ids=[], user_id="u1")
    with pytest.raises(RuntimeError, match="requires a provenance_store"):
        cm.apply_teach_correction("u1", conflict.id, new_belief, provenance_store=None)


def test_provenance_interior_corruption_raises():
    """Phase 2a: Malformed interior JSONL line raises an exception, not silently skipped."""
    db_path = "test_interior_corruption.jsonl"
    try:
        with open(db_path, "w", encoding="utf-8") as f:
            f.write('{"node_type": "Observation", "data": {"observation_id": "o1", "user_id": "u1"}}\n')
            f.write('{ INVALID_JSON_MIDDLE_LINE\n')
            f.write('{"node_type": "Observation", "data": {"observation_id": "o2", "user_id": "u1"}}\n')

        with pytest.raises(Exception):
            ProvenanceStore(store_path=db_path)
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_provenance_tail_truncation_recovers():
    """Phase 2a: Malformed final line in JSONL is recovered gracefully."""
    db_path = "test_tail_truncation.jsonl"
    try:
        with open(db_path, "w", encoding="utf-8") as f:
            f.write('{"node_type": "Observation", "data": {"observation_id": "o1", "user_id": "u1"}}\n')
            f.write('{ TRUNCATED_LAST_LINE')

        store = ProvenanceStore(store_path=db_path)
        assert "o1" in store._observations
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_claims_batch_update_validates_before_mutating():
    """Phase 2b: Batch status update validates all IDs and fails before mutating any."""
    db_path = "test_batch_validate.jsonl"
    if os.path.exists(db_path):
        os.remove(db_path)

    try:
        adapter = ClaimsEngineAdapter(db_path=db_path)
        c1 = Claim.new(user_id="u1", domain_id="d1", level=ClaimLevel.LEVEL_0, gate_evaluation=GateEvaluation(level=ClaimLevel.LEVEL_0, admissible=True, checks=[]))
        adapter.append_claim_version(c1.claim_id, c1)

        # Batch update with c1 and an invalid ID
        with pytest.raises(KeyError, match="Cannot batch-update"):
            adapter.apply_status_updates_batch([(c1.claim_id, "UNCLEAR"), ("missing_claim_xyz", "UNCLEAR")])

        # c1 status must not have been mutated
        assert adapter.get_claim_status(c1.claim_id) == "SURFACED"
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_claims_json_round_trip_complex_types():
    """Phase 2b: Datetime, ClaimLevel enum, nested checks round-trip cleanly."""
    db_path = "test_complex_types.jsonl"
    if os.path.exists(db_path):
        os.remove(db_path)

    try:
        adapter = ClaimsEngineAdapter(db_path=db_path)
        from claims_engine.claim_levels import GateCheck
        now = datetime(2024, 5, 1, 12, 30, 45)
        claim = Claim(
            claim_id="claim_complex_1",
            user_id="u_complex",
            domain_id="domain_1",
            level=ClaimLevel.LEVEL_2,
            dominant_divergence_type="TEMPORAL",
            gate_evaluation=GateEvaluation(
                level=ClaimLevel.LEVEL_2,
                admissible=True,
                checks=[GateCheck(name="temporal_consistency", passed=True, detail="OK")],
            ),
            created_at=now,
            is_dual_structured=False,
            dual_structure_components=None,
        )
        adapter.append_claim_version(claim.claim_id, claim)

        # Reload in new instance
        adapter_reloaded = ClaimsEngineAdapter(db_path=db_path)
        loaded_claim = adapter_reloaded.get_claim("claim_complex_1")
        assert loaded_claim is not None
        assert loaded_claim.level == ClaimLevel.LEVEL_2
        assert loaded_claim.created_at == now
        assert loaded_claim.gate_evaluation.checks[0].name == "temporal_consistency"
        assert loaded_claim.gate_evaluation.checks[0].passed is True
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_identity_optional_voice_cluster():
    """Phase 4a: IdentityInference accepts voice_cluster_id=None without type or runtime errors."""
    from frontier.identity_graph import PrivateIdentityGraph
    graph = PrivateIdentityGraph(user_id="user_v")
    inf = graph.add_inference(
        visual_cluster_id="vis_001",
        voice_cluster_id=None,
        candidates=["Alice", "Alicia"],
        confidence=0.80,
        user_id="user_v",
    )
    assert inf.voice_cluster_id is None
    assert inf.visual_cluster_id == "vis_001"


def test_b4_canonical_provenance_not_caller_data():
    """
    Phase 5a / B4: ExplainabilityAPI uses canonical stored provenance confidence (e.g. 0.4)
    even when caller attempts to supply fabricated confidence.
    """
    from frontier.explainability import ExplainabilityAPI

    class _StoredClaim:
        user_id = "user1"
        confidence = 0.4
        class GateEval:
            admissible = 0.4
        gate_evaluation = GateEval()

    class _Store:
        def get_claim(self, claim_id):
            return _StoredClaim() if claim_id == "canonical_c1" else None

    api = ExplainabilityAPI(
        layer0=MockLayer0Storage(),
        mirror=MockMirrorProvider(),
        claims_store=_Store(),
    )
    res = api.explain("canonical_c1", "user1")
    assert res.get("confidence") == 0.4, "Must return canonical stored confidence"
