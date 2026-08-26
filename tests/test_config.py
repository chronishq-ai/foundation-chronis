"""Tests for synthetic.config — GenConfig and roster scaffolding."""

from datetime import date

import pytest
from synthetic.config import CorruptionConfig, GenConfig, Participant, build_roster


def base_config(**overrides: object) -> GenConfig:
    defaults: dict[str, object] = {
        "seed": 42,
        "participant_count": 5,
        "study_start_date": date(2026, 1, 1),
        "study_length_days": 14,
    }
    defaults.update(overrides)
    return GenConfig(**defaults)  # type: ignore[arg-type]


# --- CorruptionConfig -------------------------------------------------


def test_corruption_config_defaults_are_low_but_nonzero() -> None:
    config = CorruptionConfig()

    assert 0.0 < config.missing_block < 0.10
    assert 0.0 < config.participant_dropout < 0.10


def test_corruption_config_rejects_out_of_range_probability() -> None:
    with pytest.raises(ValueError, match="within \\[0.0, 1.0\\]"):
        CorruptionConfig(missing_block=1.5)


def test_stress_config_has_elevated_rates() -> None:
    stress = CorruptionConfig.stress_config()
    normal = CorruptionConfig()

    assert stress.missing_block > normal.missing_block
    assert stress.participant_dropout > normal.participant_dropout


# --- GenConfig ----------------------------------------------------------


def test_genconfig_rejects_zero_participants() -> None:
    with pytest.raises(ValueError, match="participant_count"):
        base_config(participant_count=0)


def test_genconfig_rejects_zero_length_study() -> None:
    with pytest.raises(ValueError, match="study_length_days"):
        base_config(study_length_days=0)


def test_genconfig_to_dict_is_fully_serializable() -> None:
    config = base_config()
    result = config.to_dict()

    assert result["seed"] == 42
    assert result["study_start_date"] == "2026-01-01"
    assert isinstance(result["corruption"], dict)
    # every value must be a JSON-primitive type (no dates, no nested objects)
    import json

    json.dumps(result)  # raises if not serializable


# --- build_roster: determinism -------------------------------------------


def test_roster_is_deterministic_given_same_seed() -> None:
    config = base_config(seed=123)

    roster_a = build_roster(config)
    roster_b = build_roster(config)

    assert roster_a == roster_b


def test_roster_differs_across_different_seeds() -> None:
    roster_a = build_roster(base_config(seed=1))
    roster_b = build_roster(base_config(seed=2))

    assert roster_a != roster_b


def test_roster_has_requested_participant_count() -> None:
    config = base_config(participant_count=10)

    roster = build_roster(config)

    assert len(roster) == 10


def test_participant_ids_are_unique() -> None:
    roster = build_roster(base_config(participant_count=20))

    ids = [p.participant_id for p in roster]

    assert len(ids) == len(set(ids))


# --- build_roster: resting HR baseline ------------------------------------


def test_resting_heart_rate_is_within_clipped_range() -> None:
    roster = build_roster(base_config(participant_count=50, seed=7))

    for participant in roster:
        assert 50.0 <= participant.resting_heart_rate <= 95.0


def test_resting_heart_rate_varies_across_participants() -> None:
    roster = build_roster(base_config(participant_count=20, seed=7))

    rates = {p.resting_heart_rate for p in roster}

    assert len(rates) > 1  # not all identical


# --- build_roster: active days / dropout -----------------------------------


def test_active_days_never_exceed_study_length() -> None:
    config = base_config(study_length_days=30, participant_count=30)

    roster = build_roster(config)

    for participant in roster:
        assert len(participant.active_days) <= 30


def test_active_days_are_a_contiguous_prefix_of_the_study_window() -> None:
    config = base_config(
        study_start_date=date(2026, 3, 1),
        study_length_days=10,
        participant_count=30,
    )

    roster = build_roster(config)

    for participant in roster:
        assert participant.active_days[0] == date(2026, 3, 1)
        # contiguous, no gaps
        days = participant.active_days
        for earlier, later in zip(days, days[1:], strict=False):
            assert (later - earlier).days == 1


def test_zero_dropout_probability_means_full_study_window_for_everyone() -> None:
    config = base_config(
        study_length_days=14,
        participant_count=25,
        corruption=CorruptionConfig(participant_dropout=0.0),
    )

    roster = build_roster(config)

    assert all(len(p.active_days) == 14 for p in roster)


def test_high_dropout_probability_produces_some_truncated_participants() -> None:
    config = base_config(
        study_length_days=14,
        participant_count=50,
        corruption=CorruptionConfig(participant_dropout=1.0),
    )

    roster = build_roster(config)

    # With dropout probability 1.0, every participant should be truncated
    # (though at least 1 day always remains).
    assert all(len(p.active_days) < 14 for p in roster)
    assert all(len(p.active_days) >= 1 for p in roster)


def test_participant_is_immutable() -> None:
    participant = Participant(
        participant_id="synthetic_p0001",
        enrollment_date=date(2026, 1, 1),
        active_days=(date(2026, 1, 1),),
        resting_heart_rate=68.0,
    )

    with pytest.raises(AttributeError):
        participant.resting_heart_rate = 100.0  # type: ignore[misc]
