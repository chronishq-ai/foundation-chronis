"""Tests for Step 3: the remaining 6 stream generators, plus a full
8-stream registry smoke test."""

import json
from datetime import date, datetime
from random import Random

from synthetic.config import Participant
from synthetic.generators.audio_features import AudioFeatureGenerator
from synthetic.generators.chest_ecg import ChestEcgGenerator
from synthetic.generators.environment import EnvironmentGenerator
from synthetic.generators.fitbit_sleep import FitbitSleepGenerator
from synthetic.generators.fitbit_steps import FitbitStepsGenerator
from synthetic.generators.proximity import ProximityGenerator
from synthetic.generators.surveys import EmaSurveyGenerator
from synthetic.registry import REGISTRY

TEST_DAY = date(2026, 3, 1)


def make_participant(
    active_days: tuple[date, ...] = (TEST_DAY,),
    resting_heart_rate: float = 68.0,
) -> Participant:
    return Participant(
        participant_id="synthetic_p0001",
        enrollment_date=active_days[0],
        active_days=active_days,
        resting_heart_rate=resting_heart_rate,
    )


# --- Registry: full 8-stream check ------------------------------------------


def test_registry_has_all_9_registered_streams() -> None:
    """The spec's "8 modalities" refers to 8 top-level folders
    (fitbit, chest_ecg, audio_features, proximity, environment,
    phone_events, surveys, metadata) — but fitbit alone contributes 3
    separately-registered streams (heart_rate, steps, sleep), so the
    registry itself correctly has 9 entries, not 8. metadata (the
    roster) is not a per-participant-day stream and is intentionally
    not in this registry at all."""
    assert len(REGISTRY) == 9


def test_every_registered_generator_runs_without_error() -> None:
    participant = make_participant()

    for stream_name, generator in REGISTRY.items():
        records = generator.generate(participant, TEST_DAY, Random(1))
        assert isinstance(records, list), f"{stream_name} did not return a list"


# --- FitbitStepsGenerator ----------------------------------------------------


def test_steps_produces_one_bucket_per_minute() -> None:
    participant = make_participant()
    generator = FitbitStepsGenerator()

    records = generator.generate(participant, TEST_DAY, Random(1))

    assert len(records) == 24 * 60


def test_steps_capped_at_configured_maximum() -> None:
    participant = make_participant()
    generator = FitbitStepsGenerator(idle_bucket_probability=0.0, max_steps_per_bucket=130)

    records = generator.generate(participant, TEST_DAY, Random(1))

    assert all(int(r["step_count"]) <= 130 for r in records)


def test_steps_never_negative() -> None:
    participant = make_participant()
    generator = FitbitStepsGenerator()

    records = generator.generate(participant, TEST_DAY, Random(1))

    assert all(int(r["step_count"]) >= 0 for r in records)


def test_steps_is_deterministic() -> None:
    participant = make_participant()
    generator = FitbitStepsGenerator()

    a = generator.generate(participant, TEST_DAY, Random(5))
    b = generator.generate(participant, TEST_DAY, Random(5))

    assert a == b


# --- FitbitSleepGenerator -----------------------------------------------------


def test_sleep_segments_are_contiguous() -> None:
    participant = make_participant()
    generator = FitbitSleepGenerator()

    records = generator.generate(participant, TEST_DAY, Random(1))

    for earlier, later in zip(records, records[1:], strict=False):
        assert earlier["end_time"] == later["start_time"]


def test_sleep_total_duration_within_bounds() -> None:
    participant = make_participant()
    generator = FitbitSleepGenerator()

    records = generator.generate(participant, TEST_DAY, Random(1))

    total_seconds = sum(float(r["stage_duration_seconds"]) for r in records)
    total_minutes = total_seconds / 60.0

    assert 180.0 <= total_minutes <= 600.0 + 20  # small tolerance for segment rounding


def test_sleep_all_records_share_session_id() -> None:
    participant = make_participant()
    generator = FitbitSleepGenerator()

    records = generator.generate(participant, TEST_DAY, Random(1))
    session_ids = {r["sleep_session_id"] for r in records}

    assert len(session_ids) == 1


def test_sleep_stages_are_from_known_set() -> None:
    participant = make_participant()
    generator = FitbitSleepGenerator()

    records = generator.generate(participant, TEST_DAY, Random(3))

    assert all(r["stage"] in {"light", "deep", "rem", "awake"} for r in records)


# --- ChestEcgGenerator ---------------------------------------------------------


def test_ecg_sample_count_matches_duration_and_rate() -> None:
    participant = make_participant()
    generator = ChestEcgGenerator(duration_seconds=15.0, sampling_rate_hz=250.0)

    records = generator.generate(participant, TEST_DAY, Random(1))

    assert records  # at least one snippet
    assert all(int(r["sample_count"]) == 3750 for r in records)


def test_ecg_samples_field_is_valid_json_of_correct_length() -> None:
    participant = make_participant()
    generator = ChestEcgGenerator(duration_seconds=1.0, sampling_rate_hz=50.0)  # small for speed

    records = generator.generate(participant, TEST_DAY, Random(1))

    for record in records[:3]:  # spot-check a few, full waveform gen is not free
        samples = json.loads(str(record["samples"]))
        assert len(samples) == int(record["sample_count"])
        assert all(isinstance(s, int | float) for s in samples)


def test_ecg_snippets_carry_participant_id_and_derived_hr() -> None:
    participant = make_participant(resting_heart_rate=70.0)
    generator = ChestEcgGenerator(duration_seconds=1.0, sampling_rate_hz=50.0)

    records = generator.generate(participant, TEST_DAY, Random(1))

    assert all(r["participant_id"] == participant.participant_id for r in records)
    assert all(30.0 <= float(r["derived_hr_bpm"]) <= 150.0 for r in records)


# --- AudioFeatureGenerator ------------------------------------------------------


def test_audio_features_restricted_to_work_hours() -> None:
    participant = make_participant()
    generator = AudioFeatureGenerator(work_start_hour=9, work_end_hour=17)

    records = generator.generate(participant, TEST_DAY, Random(1))

    for record in records:
        ts = datetime.fromisoformat(str(record["timestamp"]))
        assert 9 <= ts.hour < 17


def test_audio_features_score_within_bounds() -> None:
    participant = make_participant()
    generator = AudioFeatureGenerator()

    records = generator.generate(participant, TEST_DAY, Random(1))

    assert all(0.0 <= float(r["voice_activity_score"]) <= 1.0 for r in records)
    assert all(float(r["pitch_variance"]) >= 0.0 for r in records)


# --- ProximityGenerator -------------------------------------------------------


def test_proximity_rssi_within_clipped_bounds() -> None:
    participant = make_participant()
    generator = ProximityGenerator()

    records = generator.generate(participant, TEST_DAY, Random(1))

    assert all(-95.0 <= float(r["rssi_dbm"]) <= -30.0 for r in records)


def test_proximity_source_is_participant_target_is_beacon() -> None:
    participant = make_participant()
    generator = ProximityGenerator()

    records = generator.generate(participant, TEST_DAY, Random(1))

    assert all(r["source_type"] == "participant" for r in records)
    assert all(r["target_type"] == "beacon" for r in records)
    assert all(r["source_id"] == participant.participant_id for r in records)


def test_proximity_events_are_bursty_not_evenly_spaced() -> None:
    participant = make_participant()
    generator = ProximityGenerator(burst_count_range=(4, 4), readings_per_burst_range=(10, 10))

    records = generator.generate(participant, TEST_DAY, Random(2))
    timestamps = [datetime.fromisoformat(str(r["timestamp"])) for r in records]

    gaps = [(timestamps[i + 1] - timestamps[i]).total_seconds() for i in range(len(timestamps) - 1)]

    # Bursty means a wide spread between smallest and largest gaps.
    assert max(gaps) > min(gaps) * 5


# --- EnvironmentGenerator ------------------------------------------------------


def test_environment_daytime_light_higher_than_night() -> None:
    participant = make_participant()
    generator = EnvironmentGenerator(min_gap_seconds=30, max_gap_seconds=60)  # dense sampling

    records = generator.generate(participant, TEST_DAY, Random(1))

    day_lux = [
        float(r["light_lux"])
        for r in records
        if generator.day_start_hour
        <= datetime.fromisoformat(str(r["timestamp"])).hour
        < generator.day_end_hour
        and float(r["light_lux"]) > 0
    ]
    night_lux = [
        float(r["light_lux"])
        for r in records
        if not (
            generator.day_start_hour
            <= datetime.fromisoformat(str(r["timestamp"])).hour
            < generator.day_end_hour
        )
    ]

    assert day_lux and night_lux
    assert (sum(day_lux) / len(day_lux)) > (sum(night_lux) / len(night_lux))


def test_environment_humidity_clipped_to_bounds() -> None:
    participant = make_participant()
    generator = EnvironmentGenerator()

    records = generator.generate(participant, TEST_DAY, Random(1))

    assert all(15.0 <= float(r["humidity_pct"]) <= 80.0 for r in records)


def test_environment_device_id_derived_from_participant() -> None:
    participant = make_participant()
    generator = EnvironmentGenerator()

    records = generator.generate(participant, TEST_DAY, Random(1))

    assert all(r["device_id"] == f"{participant.participant_id}_env_device" for r in records)


# --- EmaSurveyGenerator --------------------------------------------------------


def test_ema_response_count_within_spec_range() -> None:
    participant = make_participant()
    generator = EmaSurveyGenerator()

    records = generator.generate(participant, TEST_DAY, Random(1))
    periodic = [r for r in records if r["survey_type"] == "periodic"]

    assert 3 <= len(periodic) <= 5


def test_ema_stress_score_within_1_to_5() -> None:
    participant = make_participant()
    generator = EmaSurveyGenerator()

    records = generator.generate(participant, TEST_DAY, Random(1))

    assert all(1 <= int(r["stress_1_to_5"]) <= 5 for r in records)


def test_ema_onboarding_survey_on_first_active_day() -> None:
    active_days = (date(2026, 3, 1), date(2026, 3, 2), date(2026, 3, 3))
    participant = make_participant(active_days=active_days)
    generator = EmaSurveyGenerator()

    records = generator.generate(participant, active_days[0], Random(1))

    assert any(r["survey_type"] == "onboarding" for r in records)


def test_ema_closing_survey_on_last_active_day() -> None:
    active_days = (date(2026, 3, 1), date(2026, 3, 2), date(2026, 3, 3))
    participant = make_participant(active_days=active_days)
    generator = EmaSurveyGenerator()

    records = generator.generate(participant, active_days[-1], Random(1))

    assert any(r["survey_type"] == "closing" for r in records)


def test_ema_no_boundary_survey_on_middle_day() -> None:
    active_days = (date(2026, 3, 1), date(2026, 3, 2), date(2026, 3, 3))
    participant = make_participant(active_days=active_days)
    generator = EmaSurveyGenerator()

    records = generator.generate(participant, active_days[1], Random(1))

    assert not any(r["survey_type"] in {"onboarding", "closing"} for r in records)


def test_ema_stress_score_correlates_with_shared_daily_intensity() -> None:
    """Two participants on days with different daily_stress_intensity
    values should show different average stress scores on average,
    proving the correlation actually has an effect (not perfectly
    derived, but not independent either)."""

    from synthetic.dailystate import daily_stress_intensity

    generator = EmaSurveyGenerator(responses_per_day_range=(20, 20))  # many samples to reduce noise

    # Find two days with meaningfully different intensity for the same participant.
    participant = make_participant(active_days=tuple(date(2026, 1, d) for d in range(1, 15)))

    intensities = {
        day: daily_stress_intensity(participant.participant_id, day)
        for day in participant.active_days
    }
    low_day = min(intensities, key=lambda d: intensities[d])
    high_day = max(intensities, key=lambda d: intensities[d])

    assert intensities[high_day] - intensities[low_day] > 0.3  # ensure a meaningful gap exists

    low_records = generator.generate(participant, low_day, Random(1))
    high_records = generator.generate(participant, high_day, Random(1))

    low_avg = sum(int(r["stress_1_to_5"]) for r in low_records) / len(low_records)
    high_avg = sum(int(r["stress_1_to_5"]) for r in high_records) / len(high_records)

    assert high_avg > low_avg
