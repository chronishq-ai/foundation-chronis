import pytest

from silence_map import (
    SilenceInput,
    SilenceMap,
    SilenceMapError,
    SilenceResult,
)


def sample(**overrides):
    values = {
        "user_id": "user_001",
        "silence_seconds": 10.0,
        "turn_expected": False,
        "typing_active": True,
        "physiological_delta": 0.1,
    }
    values.update(overrides)
    return SilenceInput(**values)


def test_attentive_silence_from_typing():
    result = SilenceMap().classify(sample())

    assert isinstance(result, SilenceResult)
    assert result.classification == "attentive"


def test_attentive_silence_when_typing_continues():
    result = SilenceMap().classify(
        sample(
            turn_expected=True,
            typing_active=True,
        )
    )

    assert result.classification == "attentive"


def test_avoidant_silence_from_expected_turn_and_activation():
    result = SilenceMap().classify(
        sample(
            turn_expected=True,
            typing_active=False,
            physiological_delta=0.8,
        )
    )

    assert result.classification == "avoidant"


def test_conversational_silence_without_activation():
    result = SilenceMap().classify(
        sample(
            turn_expected=True,
            typing_active=False,
            physiological_delta=0.1,
        )
    )

    assert result.classification == "conversational"


def test_confidence_is_bounded():
    result = SilenceMap().classify(sample())

    assert 0.0 <= result.confidence <= 1.0


def test_explanation_is_present():
    result = SilenceMap().classify(sample())

    assert result.explanation


def test_classify_many():
    results = SilenceMap.classify_many(
        [
            sample(),
            sample(
                turn_expected=True,
                typing_active=False,
                physiological_delta=0.8,
            ),
        ]
    )

    assert len(results) == 2
    assert results[0].classification == "attentive"
    assert results[1].classification == "avoidant"


def test_rejects_empty_user():
    with pytest.raises(SilenceMapError):
        SilenceMap().classify(
            sample(user_id="")
        )


def test_rejects_negative_silence():
    with pytest.raises(SilenceMapError):
        SilenceMap().classify(
            sample(silence_seconds=-1)
        )


def test_rejects_non_finite_physiology():
    with pytest.raises(SilenceMapError):
        SilenceMap().classify(
            sample(physiological_delta=float("nan"))
        )
def test_zero_silence_is_rejected():
    with pytest.raises(SilenceMapError):
        SilenceMap().classify(sample(silence_seconds=0))
