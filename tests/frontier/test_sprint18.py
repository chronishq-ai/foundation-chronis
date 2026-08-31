"""
tests/frontier/test_sprint18.py

Sprint 18 tests — S1720.1, S1720.2, R2-F18.1, R2-F18.2
"""

import random
import pytest

from frontier.central_retrieval_core import CentralRetrievalCore, CrossUserEvidenceError
from frontier.memory_orchestrator import MemoryOrchestrator
from frontier.multimodal_assistant import MultimodalAssistant
from frontier.interfaces.policy import MockPolicyEngine
from frontier.interfaces.llm import LLMProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyVisualRetrieval:
    def search_visual(self, user_id, query):
        return [{"canonical_record_pointer": "ptr1", "confidence": 0.9, "owner_user_id": user_id}]


class MockLLMProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        return "Mock answer for general knowledge query."


# ---------------------------------------------------------------------------
# S1720.1 / R2-F18.1 — Central Retrieval Core
# ---------------------------------------------------------------------------

def test_central_retrieval_policy_engine_enforced():
    """
    R2-F18.1: CRC uses policy_engine, not caller-supplied consent_tier.
    A policy engine that denies access must cause retrieval to fail.
    """
    class DenyAllPolicy(MockPolicyEngine):
        def check_access(self, user_id, resource_type, required_tier):
            return False   # deny everything

    orchestrator = MemoryOrchestrator(DummyVisualRetrieval())
    core = CentralRetrievalCore(orchestrator, DenyAllPolicy())

    # Even if caller supplies consent_tier=2, policy engine denies → error
    res = core.retrieve("query", "past", "assistant", "user1", {"consent_tier": 2})
    assert "error" in res


def test_central_retrieval_valid():
    """CRC grants access when policy engine approves."""
    policy = MockPolicyEngine()   # default allows all
    orchestrator = MemoryOrchestrator(DummyVisualRetrieval())
    core = CentralRetrievalCore(orchestrator, policy)

    res = core.retrieve("query", "past", "assistant", "user1", {})
    assert len(res.get("evidence_items", [])) >= 1


def test_central_retrieval_fails_closed_cross_user():
    """
    R2-F18.1 / S1720.9: CRC raises CrossUserEvidenceError when orchestrator
    returns evidence whose owner_user_id != requesting_user_id.
    Must FAIL CLOSED — not strip-and-continue.
    """
    class CrossUserReturningOrchestrator:
        def orchestrate(self, user_id, query, query_type):
            return {
                "episode_window": (None, None),
                "evidence_items": [
                    {
                        "modality": "visual",
                        "content_pointer": "ptr_b",
                        "confidence": 0.9,
                        "source": "visual_index",
                        "owner_user_id": "user_B",   # ← wrong user
                    }
                ],
                "contradictions": [],
                "overall_confidence": 0.9,
            }

    policy = MockPolicyEngine()
    core = CentralRetrievalCore(CrossUserReturningOrchestrator(), policy)

    with pytest.raises(CrossUserEvidenceError):
        core.retrieve("query", "past", "assistant", "user_A", {})


def test_no_bypass_of_central_retrieval_core():
    """Validates that interface modules do not bypass CentralRetrievalCore."""
    import ast
    from pathlib import Path

    banned_imports = ["visual_memory", "transcript_search", "claims_engine_adapter"]
    root = Path(__file__).parent.parent.parent / "src" / "frontier"
    interface_files = ["voice_assistant.py", "multimodal_assistant.py", "explainability.py"]

    for filename in interface_files:
        filepath = root / filename
        if not filepath.exists():
            continue
        tree = ast.parse(filepath.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for banned in banned_imports:
                        assert banned not in alias.name, (
                            f"Bypass detected in {filename}: imports {banned}"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for banned in banned_imports:
                        assert banned not in node.module, (
                            f"Bypass detected in {filename}: imports from {banned}"
                        )


# ---------------------------------------------------------------------------
# S1720.2 / R2-F18.2 — EvidencePackage + Memory Orchestrator
# ---------------------------------------------------------------------------

def test_evidence_package_contract_low_confidence():
    """S1720.2 T1: <0.5 confidence items retained, not silently dropped."""
    class LowConfRetrieval:
        def search_visual(self, user_id, query):
            return [{"canonical_record_pointer": "ptr_low", "confidence": 0.2, "owner_user_id": user_id}]

    orch = MemoryOrchestrator(LowConfRetrieval())
    res = orch.orchestrate("u1", "q", "past")
    assert len(res["evidence_items"]) == 1
    assert res["evidence_items"][0]["confidence"] == 0.2
    assert res["evidence_items"][0].get("low_confidence") is True


def test_evidence_package_contract_single_item_no_contradiction():
    """S1720.2 T1b: A single evidence item must never produce a contradiction."""
    class SingleItemRetrieval:
        def search_visual(self, user_id, query):
            return [{"canonical_record_pointer": "ptr1", "confidence": 0.9, "owner_user_id": user_id}]

    orch = MemoryOrchestrator(SingleItemRetrieval())
    res = orch.orchestrate("u1", "q", "past")
    assert len(res["contradictions"]) == 0


def test_orchestrator_no_fake_contradiction():
    """
    R2-F18.2: Two agreeing items from same source must NOT produce a
    contradiction.  The old len > 1 heuristic was wrong.
    """
    class AgreeingRetrieval:
        def search_visual(self, user_id, query):
            return [
                {"canonical_record_pointer": "ptr1", "confidence": 0.9, "owner_user_id": user_id},
                {"canonical_record_pointer": "ptr2", "confidence": 0.8, "owner_user_id": user_id},
            ]

    orch = MemoryOrchestrator(AgreeingRetrieval())
    res = orch.orchestrate("u1", "q", "past")
    assert len(res["evidence_items"]) == 2
    # Two items with no assertion dict → no contradiction
    assert len(res["contradictions"]) == 0
    # overall_confidence should be weighted average, not reduced by false contradiction
    expected_avg = (0.9 + 0.8) / 2
    assert abs(res["overall_confidence"] - expected_avg) < 0.01


def test_orchestrator_semantic_contradiction_detected():
    """
    R2-F18.2: Two items with conflicting assertions (same subject+predicate,
    different object) ARE a contradiction.
    """
    class ConflictingRetrieval:
        def search_visual(self, user_id, query):
            return [
                {
                    "canonical_record_pointer": "ptr1",
                    "confidence": 0.9,
                    "owner_user_id": user_id,
                    "assertion": {"subject": "person_1", "predicate": "wearing", "object": "red_shirt"},
                },
                {
                    "canonical_record_pointer": "ptr2",
                    "confidence": 0.8,
                    "owner_user_id": user_id,
                    "assertion": {"subject": "person_1", "predicate": "wearing", "object": "blue_shirt"},
                },
            ]

    orch = MemoryOrchestrator(ConflictingRetrieval())
    res = orch.orchestrate("u1", "q", "past")
    assert len(res["contradictions"]) == 1
    assert res["overall_confidence"] < (0.9 + 0.8) / 2   # reduced by contradiction


def test_orchestrator_episode_window_from_evidence():
    """R2-F18.2: episode_window must be derived from evidence timestamps, not datetime.now()."""
    from datetime import datetime, timedelta

    base = datetime(2024, 3, 15, 10, 0, 0)

    class TimestampedRetrieval:
        def search_visual(self, user_id, query):
            return [
                {"canonical_record_pointer": "p1", "confidence": 0.9,
                 "owner_user_id": user_id, "timestamp": base},
                {"canonical_record_pointer": "p2", "confidence": 0.8,
                 "owner_user_id": user_id, "timestamp": base + timedelta(hours=2)},
            ]

    orch = MemoryOrchestrator(TimestampedRetrieval())
    res = orch.orchestrate("u1", "q", "past")
    start, end = res["episode_window"]
    assert start == base
    assert end == base + timedelta(hours=2)


# ---------------------------------------------------------------------------
# R2-F18.1 — 10k cross-user benchmark (slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cross_user_10k_property():
    """
    R2-F18.1 / XINT.3 (slow): 10,000 cross-user requests must ALL raise
    CrossUserEvidenceError — zero leaks, no silent returns.
    """
    policy = MockPolicyEngine()
    violations = 0

    for i in range(10_000):
        user_a = f"user_a_{i % 100}"
        user_b = f"user_b_{i % 100}"

        class _CrossOrch:
            def orchestrate(self, user_id, query, query_type):
                return {
                    "episode_window": (None, None),
                    "evidence_items": [{
                        "modality": "visual",
                        "content_pointer": f"ptr_{i}",
                        "confidence": 0.9,
                        "source": "visual_index",
                        "owner_user_id": user_b,
                    }],
                    "contradictions": [],
                    "overall_confidence": 0.9,
                }

        core = CentralRetrievalCore(_CrossOrch(), policy)
        try:
            core.retrieve("q", "past", "test", user_a, {})
            violations += 1   # should never reach here
        except CrossUserEvidenceError:
            pass   # expected

    assert violations == 0, (
        f"SECURITY FAILURE: {violations}/10,000 cross-user requests leaked evidence"
    )


# ---------------------------------------------------------------------------
# R2-F17.3 — Multimodal assistant no fabricated consent / context
# ---------------------------------------------------------------------------

def test_mixed_query_composition():
    """Validates personal/general knowledge tagging (Sprint 18)."""
    policy = MockPolicyEngine()
    orchestrator = MemoryOrchestrator(DummyVisualRetrieval())
    core = CentralRetrievalCore(orchestrator, policy)
    llm = MockLLMProvider()
    assistant = MultimodalAssistant(core, llm, policy_engine=policy)

    res = assistant.resolve_query("user1", "where are my keys?", "mixed_query")
    assert "[PERSONAL_EVIDENCE]" in res
    assert "[GENERAL_KNOWLEDGE]" in res


def test_live_context_no_fabricated_string():
    """R2-F17.3: live_context must not return 'Object identified as X'."""
    import numpy as np
    policy = MockPolicyEngine()
    orchestrator = MemoryOrchestrator(DummyVisualRetrieval())
    core = CentralRetrievalCore(orchestrator, policy)
    llm = MockLLMProvider()
    assistant = MultimodalAssistant(core, llm, policy_engine=policy)

    fake_embedding = np.zeros(512, dtype="float32")
    res = assistant.resolve_query("user1", "what is this?", "live_context",
                                  live_camera_embedding=fake_embedding)
    assert "Object identified as" not in str(res), (
        "live_context must not fabricate identity strings"
    )


# ---------------------------------------------------------------------------
# Additional Sprint 18 tests: status, false-positives, forged principal, metadata
# ---------------------------------------------------------------------------

from frontier.interfaces.policy import DenyForPrincipalMismatchPolicy
from frontier.visual_memory import VisualMemoryIndex, MockFAISS, DeterministicTestEncoder, VisualEncoderMetadata


def test_orchestrator_all_low_confidence_status():
    """S1720.2: All evidence items < 0.5 results in status='low_confidence'."""
    class AllLowRetrieval:
        def search_visual(self, user_id, query):
            return [
                {"canonical_record_pointer": "p1", "confidence": 0.3, "owner_user_id": user_id},
                {"canonical_record_pointer": "p2", "confidence": 0.2, "owner_user_id": user_id},
            ]
    orch = MemoryOrchestrator(AllLowRetrieval())
    res = orch.orchestrate("user1", "q", "past")
    assert res.get("status") == "low_confidence"
    assert res.get("overall_confidence") == 0.25


def test_orchestrator_empty_evidence_explicit_contract():
    """S1720.2: Empty evidence produces explicit (None, None), 0.0 confidence, and 'low_confidence' status."""
    class EmptyRetrieval:
        def search_visual(self, user_id, query):
            return []
    orch = MemoryOrchestrator(EmptyRetrieval())
    res = orch.orchestrate("user1", "q", "past")
    assert res["episode_window"] == (None, None)
    assert res["overall_confidence"] == 0.0
    assert res["evidence_items"] == []
    assert res["status"] == "low_confidence"


def test_orchestrator_contradiction_false_positives():
    """R2-F18.2: Same fact observed across different modalities must NOT contradict."""
    class MultiModalSameFactRetrieval:
        def search_visual(self, user_id, query):
            return [
                {
                    "canonical_record_pointer": "p1",
                    "confidence": 0.9,
                    "owner_user_id": user_id,
                    "modality": "visual",
                    "assertion": {"subject": "person_1", "predicate": "location", "object": "kitchen"},
                },
                {
                    "canonical_record_pointer": "p2",
                    "confidence": 0.85,
                    "owner_user_id": user_id,
                    "modality": "audio",
                    "assertion": {"subject": "person_1", "predicate": "location", "object": "kitchen"},
                },
            ]
    orch = MemoryOrchestrator(MultiModalSameFactRetrieval())
    res = orch.orchestrate("user1", "q", "past")
    assert len(res["contradictions"]) == 0, "Same fact across modalities is not a contradiction"


def test_orchestrator_diff_predicate_not_contradiction():
    """R2-F18.2: Different predicates about the same subject must NOT contradict."""
    class DiffPredicateRetrieval:
        def search_visual(self, user_id, query):
            return [
                {
                    "canonical_record_pointer": "p1",
                    "confidence": 0.9,
                    "owner_user_id": user_id,
                    "assertion": {"subject": "person_1", "predicate": "wearing", "object": "blue_shirt"},
                },
                {
                    "canonical_record_pointer": "p2",
                    "confidence": 0.85,
                    "owner_user_id": user_id,
                    "assertion": {"subject": "person_1", "predicate": "name", "object": "John"},
                },
            ]
    orch = MemoryOrchestrator(DiffPredicateRetrieval())
    res = orch.orchestrate("user1", "q", "past")
    assert len(res["contradictions"]) == 0, "Different predicates must not be considered a contradiction"


def test_central_retrieval_forged_principal_denied():
    """
    S1720.1 / B8: When an authenticated principal tries to request data for another user_id,
    the PolicyEngine rejects access at the principal boundary.
    """
    deny_policy = DenyForPrincipalMismatchPolicy(authenticated_principal="attacker")
    orchestrator = MemoryOrchestrator(DummyVisualRetrieval())
    core = CentralRetrievalCore(orchestrator, deny_policy)

    res = core.retrieve(query="where was I?", query_type="past", requesting_interface="test", user_id="victim", consent_context={})
    assert "error" in res
    assert res["error"] == "Access denied by policy engine"


def test_multimodal_match_includes_encoder_metadata():
    """R2-F17.3 / S1720.8: MultimodalAssistant match response includes VisualEncoderMetadata."""
    encoder = DeterministicTestEncoder()
    mock_index = MockFAISS(encoder.dimension)
    index = VisualMemoryIndex("user1", encoder, index_override=mock_index)
    # Add an entry with distance 0.0 (exact match)
    index.entries = [{
        "canonical_record_pointer": "ptr_exact",
        "ann_distance": 0.0,
        "user_id": "user1",
    }]

    custom_meta = VisualEncoderMetadata(
        model_id="custom-clip-v1",
        version=1,
        dimension=512,
        dtype="float32",
        normalization="L2",
        similarity_metric="cosine",
    )

    core = CentralRetrievalCore(MemoryOrchestrator(DummyVisualRetrieval()), MockPolicyEngine())
    assistant = MultimodalAssistant(
        retrieval_core=core,
        llm_provider=MockLLMProvider(),
        policy_engine=MockPolicyEngine(),
        visual_index_provider=lambda uid: index if uid == "user1" else None,
        encoder_metadata=custom_meta,
    )

    query_vec = encoder.encode("test frame")
    res = assistant.resolve_query("user1", "identify this", "live_context", live_camera_embedding=query_vec)
    assert res.get("status") == "match_found"
    assert res.get("encoder_model_id") == "custom-clip-v1"
    assert res.get("encoder_version") == 1
    assert res.get("similarity_metric") == "cosine"
    assert res.get("threshold_used") == 0.75
