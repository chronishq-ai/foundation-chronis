import pytest
from frontier.central_retrieval_core import CentralRetrievalCore
from frontier.memory_orchestrator import MemoryOrchestrator
from frontier.multimodal_assistant import MultimodalAssistant

class DummyVisualRetrieval:
    def search_visual(self, user_id, query):
        return [{"canonical_record_pointer": "ptr1", "confidence": 0.9}]

def test_central_retrieval_routing():
    """Validates CentralRetrievalCore and bypass protection (Sprint 18)."""
    orchestrator = MemoryOrchestrator(DummyVisualRetrieval())
    core = CentralRetrievalCore(orchestrator)
    
    # Must fail without consent tier 2
    res = core.retrieve("query", "past", "assistant", "user1", {"consent_tier": 1})
    assert "error" in res
    
    # Valid retrieval
    res = core.retrieve("query", "past", "assistant", "user1", {"consent_tier": 2})
    assert len(res.get("evidence_items", [])) == 1

from frontier.interfaces.llm import LLMProvider

class MockLLMProvider(LLMProvider):
    def generate(self, prompt: str) -> str:
        return "Mock answer for general knowledge query."

def test_mixed_query_composition():
    """Validates personal/general knowledge tagging (Sprint 18)."""
    orchestrator = MemoryOrchestrator(DummyVisualRetrieval())
    core = CentralRetrievalCore(orchestrator)
    llm = MockLLMProvider()
    assistant = MultimodalAssistant(core, llm)
    
    res = assistant.resolve_query("user1", "where are my keys?", "mixed_query")
    assert "[PERSONAL_EVIDENCE]" in res
    assert "[GENERAL_KNOWLEDGE]" in res

import ast
from pathlib import Path

def test_no_bypass_of_central_retrieval_core():
    """Validates that interface modules do not bypass CentralRetrievalCore."""
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
                        assert banned not in alias.name, f"Bypass detected in {filename}: imports {banned}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for banned in banned_imports:
                        assert banned not in node.module, f"Bypass detected in {filename}: imports from {banned}"

def test_evidence_package_contract_low_confidence():
    """S1720.2 T1: Validates <0.5 confidence items are not silently dropped."""
    class DummyVisualRetrievalLow:
        def search_visual(self, user_id, query):
            return [{"canonical_record_pointer": "ptr_low", "confidence": 0.2}]
            
    orchestrator = MemoryOrchestrator(DummyVisualRetrievalLow())
    res = orchestrator.orchestrate("u1", "q", "past")
    assert len(res["evidence_items"]) == 1
    assert res["evidence_items"][0]["confidence"] == 0.2
    assert res["overall_confidence"] == 1.0

def test_evidence_package_contract_contradictions():
    """S1720.2 T2: Validates contradictory evidence lowers overall_confidence."""
    class DummyVisualRetrievalContradict:
        def search_visual(self, user_id, query):
            return [
                {"canonical_record_pointer": "ptr1", "confidence": 0.9},
                {"canonical_record_pointer": "ptr2", "confidence": 0.8}
            ]
            
    orchestrator = MemoryOrchestrator(DummyVisualRetrievalContradict())
    res = orchestrator.orchestrate("u1", "q", "past")
    assert len(res["evidence_items"]) == 2
    assert len(res["contradictions"]) == 1
    assert res["overall_confidence"] < 1.0

