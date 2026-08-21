
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analyzer  
from llm_client import LLMProvider  
from prompt import build_prompt  
from schemas import EventSignals 


class FakeProvider(LLMProvider):
    """Returns a scripted sequence of responses, one per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self._responses:
            raise RuntimeError("FakeProvider ran out of scripted responses")
        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def reset_provider():
    """Ensure each test starts with a clean injected provider."""
    yield
    analyzer.set_provider(None)



def test_build_prompt_inserts_event_text():
    prompt = build_prompt("I got promoted today.")
    assert "I got promoted today." in prompt
    assert "{{event}}" not in prompt


def test_build_prompt_rejects_empty_event():
    with pytest.raises(ValueError):
        build_prompt("   ")


def test_event_signals_accepts_valid_payload():
    payload = {"signals": {"mood": {"value": 8, "confidence": 0.9}}}
    result = EventSignals(**payload)
    assert result.signals["mood"].value == 8


def test_event_signals_rejects_out_of_range_value():
    with pytest.raises(Exception):
        EventSignals(signals={"mood": {"value": 15, "confidence": 0.9}})


def test_event_signals_rejects_unknown_variable():
    with pytest.raises(Exception):
        EventSignals(signals={"not_a_real_variable": {"value": 5, "confidence": 0.5}})


def test_event_signals_allows_empty_signals():
    result = EventSignals(signals={})
    assert result.signals == {}



def test_analyze_event_happy_path():
    fake = FakeProvider([
        json.dumps({"signals": {"mood": {"value": 8, "confidence": 0.95}}})
    ])
    analyzer.set_provider(fake)

    result = analyzer.analyze_event("I got promoted today.")

    assert result == {"signals": {"mood": {"value": 8.0, "confidence": 0.95}}}
    assert len(fake.calls) == 1


def test_analyze_event_strips_markdown_fences():
    fake = FakeProvider([
        "```json\n" + json.dumps({"signals": {"mood": {"value": 3, "confidence": 0.5}}}) + "\n```"
    ])
    analyzer.set_provider(fake)

    result = analyzer.analyze_event("I had a bad day.")

    assert result["signals"]["mood"]["value"] == 3.0


def test_analyze_event_retries_on_malformed_json_then_succeeds():
    fake = FakeProvider([
        "not valid json at all",
        json.dumps({"signals": {"stress": {"value": 7, "confidence": 0.8}}}),
    ])
    analyzer.set_provider(fake)

    result = analyzer.analyze_event("Deadline moved up with no warning.", max_retries=3)

    assert result["signals"]["stress"]["value"] == 7.0
    assert len(fake.calls) == 2
 
    assert "previous response was invalid" in fake.calls[1]


def test_analyze_event_retries_on_schema_violation_then_succeeds():
    fake = FakeProvider([
        json.dumps({"signals": {"mood": {"value": 99, "confidence": 0.9}}}),  
        json.dumps({"signals": {"mood": {"value": 6, "confidence": 0.6}}}),
    ])
    analyzer.set_provider(fake)

    result = analyzer.analyze_event("Something happened.", max_retries=3)

    assert result["signals"]["mood"]["value"] == 6.0
    assert len(fake.calls) == 2


def test_analyze_event_raises_after_exhausting_retries():
    fake = FakeProvider(["still not json", "still not json"])
    analyzer.set_provider(fake)

    with pytest.raises(RuntimeError):
        analyzer.analyze_event("Ambiguous thing occurred.", max_retries=2)

    assert len(fake.calls) == 2


def test_analyze_event_accepts_empty_signals_for_neutral_event():
    fake = FakeProvider([json.dumps({"signals": {}})])
    analyzer.set_provider(fake)

    result = analyzer.analyze_event("I bought milk and eggs.")

    assert result == {"signals": {}}
