import pytest
from datetime import datetime
from src.frontier.model_boundary import BoundaryValidator
from src.frontier.interfaces.layer0 import MockLayer0Storage
from src.frontier.interfaces.mirror import MockMirrorProvider
from src.frontier.voice_assistant import VoiceAssistant

def test_class_a_b_boundary_enforcement():
    """Validates Class A/B boundaries (Sprint 17)."""
    # Registration requirement
    with pytest.raises(ValueError, match="'model_class' tag is required"):
        BoundaryValidator.validate_registration({"name": "my_model"})

    assert BoundaryValidator.validate_registration({"model_class": "A"}) == True
    
    # CI Isolation (A cannot load B)
    all_artifacts = {
        "path/to/base": {"model_class": "A"},
        "path/to/personal": {"model_class": "B"}
    }
    
    with pytest.raises(PermissionError, match="CI ISOLATION FAILURE"):
        BoundaryValidator.check_ci_isolation(
            {"target_class": "A", "artifact_dependencies": ["path/to/personal"]},
            all_artifacts
        )

def test_voice_routing():
    """Validates voice assistant routing (Sprint 17)."""
    mirror = MockMirrorProvider()
    assistant = VoiceAssistant(mirror)
    
    assert "visual/temporal" in assistant.process_query("user1", "have I been here before?").lower()
    assert "explain" in assistant.process_query("user1", "explain this to me").lower()
    assert "cannot help with that general question" in assistant.process_query("user1", "who is the president?").lower()
