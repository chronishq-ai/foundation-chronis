import pytest
from src.frontier.central_retrieval_core import CentralRetrievalCore
from src.frontier.memory_orchestrator import MemoryOrchestrator
from src.frontier.multimodal_assistant import MultimodalAssistant

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

from src.frontier.interfaces.llm import LLMProvider

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
