"""Tests for Step 2: the two reference exemplar stream generators
(fitbit.heart_rate, phone_events.interaction) and the registry."""

from datetime import date
from random import Random

from synthetic.config import Participant
from synthetic.generators.fitbit import FitbitHRGenerator
from synthetic.generators.phone_events import PhoneEventGenerator
from synthetic.registry import REGISTRY

TEST_DAY = date(2026, 3, 1)


def make_participant(resting_heart_rate: float = 68.0) -> Participant:
    return Participant(
        participant_id="synthetic_p0001",
        enrollment_date=TEST_DAY,
        active_days=(TEST_DAY,),
        resting_heart_rate=resting_heart_rate,
    )


# --- Registry --------------------------------------------------------------


def test_registry_contains_both_reference_exemplars() -> None:
    assert "fitbit.heart_rate" in REGISTRY
    assert "phone_events.interaction" in REGISTRY


def test_registry_entries_conform_to_generate_signature() -> None:
    participant = make_participant()

    for stream_name, generator in REGISTRY.items():
        rng = Random(1)
        records = generator.generate(participant, TEST_DAY, rng)
        assert isinstance(records, list), f"{stream_name} did not return a list"


# --- FitbitHRGenerator: determinism -----------------------------------------


def test_fitbit_hr_is_deterministic_given_same_rng_seed() -> None:
    participant = make_participant()
    generator = FitbitHRGenerator()

    records_a = generator.generate(participant, TEST_DAY, Random(42))
    records_b = generator.generate(participant, TEST_DAY, Random(42))

    assert records_a == records_b


def test_fitbit_hr_differs_across_seeds() -> None:
    participant = make_participant()
    generator = FitbitHRGenerator()

    records_a = generator.generate(participant, TEST_DAY, Random(1))
    records_b = generator.generate(participant, TEST_DAY, Random(2))

    assert records_a != records_b


# --- FitbitHRGenerator: irregular sampling ----------------------------------


def test_fitbit_hr_sampling_is_irregular_not_uniform() -> None:
    participant = make_participant()
    generator = FitbitHRGenerator()

    records = generator.generate(participant, TEST_DAY, Random(7))

    from datetime import datetime

    timestamps = [datetime.fromisoformat(str(r["timestamp"])) for r in records]
    gaps = {(timestamps[i + 1] - timestamps[i]).total_seconds() for i in range(len(timestamps) - 1)}

    # A uniform grid would produce (near) one single gap value repeated.
    assert len(gaps) > 5


def test_fitbit_hr_gaps_within_configured_bounds() -> None:
    participant = make_participant()
    generator = FitbitHRGenerator(min_gap_seconds=5.0, max_gap_seconds=120.0)

    records = generator.generate(participant, TEST_DAY, Random(7))

    from datetime import datetime

    timestamps = sorted(datetime.fromisoformat(str(r["timestamp"])) for r in records)
    for earlier, later in zip(timestamps, timestamps[1:], strict=False):
        gap = (later - earlier).total_seconds()
        assert 5.0 <= gap <= 120.0 + 1e-6


# --- FitbitHRGenerator: physiological plausibility --------------------------


def test_fitbit_hr_never_below_physiological_floor() -> None:
    participant = make_participant(resting_heart_rate=50.0)
    generator = FitbitHRGenerator()

    records = generator.generate(participant, TEST_DAY, Random(99))

    assert all(float(r["heart_rate_bpm"]) >= generator.minimum_physiological_hr for r in records)


def test_fitbit_hr_stress_and_active_windows_raise_values_above_baseline() -> None:
    """With enough active/stress windows forced, at least some readings
    should be meaningfully above the flat resting baseline — proves the
    window-offset logic actually fires, not just returns the baseline."""

    participant = make_participant(resting_heart_rate=68.0)
    generator = FitbitHRGenerator(
        active_window_count_range=(3, 3),
        active_window_minutes_range=(60, 60),
        stress_window_count_range=(2, 2),
        stress_window_minutes_range=(30, 30),
    )

    records = generator.generate(participant, TEST_DAY, Random(5))
    values = [float(r["heart_rate_bpm"]) for r in records]

    assert max(values) > 68.0 + 15.0  # meaningfully above baseline


def test_fitbit_hr_records_carry_participant_id() -> None:
    participant = make_participant()
    generator = FitbitHRGenerator()

    records = generator.generate(participant, TEST_DAY, Random(3))

    assert all(r["participant_id"] == participant.participant_id for r in records)


# --- FitbitHRGenerator: flat-line injection ----------------------------------


def test_fitbit_hr_flat_line_injection_produces_repeated_run() -> None:
    """With flat_line_probability=1.0, every run must contain an actual
    flat run of identical consecutive values."""

    participant = make_participant()
    generator = FitbitHRGenerator(
        flat_line_probability=1.0,
        flat_line_run_length_range=(30, 30),
    )

    records = generator.generate(participant, TEST_DAY, Random(11))
    values = [float(r["heart_rate_bpm"]) for r in records]

    # find the longest run of identical consecutive values
    longest_run = 1
    current_run = 1
    for earlier, later in zip(values, values[1:], strict=False):
        if earlier == later:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1

    assert longest_run >= 30


def test_fitbit_hr_zero_flat_line_probability_produces_no_forced_flat_run() -> None:
    participant = make_participant()
    generator = FitbitHRGenerator(flat_line_probability=0.0)

    records = generator.generate(participant, TEST_DAY, Random(11))
    values = [float(r["heart_rate_bpm"]) for r in records]

    longest_run = 1
    current_run = 1
    for earlier, later in zip(values, values[1:], strict=False):
        if earlier == later:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1

    assert longest_run < 30


# --- PhoneEventGenerator: determinism + bursty clustering -------------------


def test_phone_events_is_deterministic_given_same_seed() -> None:
    participant = make_participant()
    generator = PhoneEventGenerator()

    records_a = generator.generate(participant, TEST_DAY, Random(42))
    records_b = generator.generate(participant, TEST_DAY, Random(42))

    assert records_a == records_b


def test_phone_events_are_sorted_by_timestamp() -> None:
    participant = make_participant()
    generator = PhoneEventGenerator()

    records = generator.generate(participant, TEST_DAY, Random(3))
    timestamps = [r["timestamp"] for r in records]

    assert timestamps == sorted(timestamps)


def test_phone_events_cluster_into_bursts_not_evenly_spaced() -> None:
    """Events within a burst window should be much closer together in
    time than the average gap across the whole day — proves clustering,
    not uniform spacing."""

    participant = make_participant()
    generator = PhoneEventGenerator(
        burst_count_range=(5, 5),
        burst_event_count_range=(10, 10),
        burst_window_minutes=10,
        overnight_event_probability=0.0,
    )

    from datetime import datetime

    records = generator.generate(participant, TEST_DAY, Random(4))
    timestamps = [datetime.fromisoformat(str(r["timestamp"])) for r in records]

    gaps = [(timestamps[i + 1] - timestamps[i]).total_seconds() for i in range(len(timestamps) - 1)]

    small_gaps = [g for g in gaps if g < 600]  # within a burst window
    large_gaps = [g for g in gaps if g >= 600]  # between bursts

    assert len(small_gaps) > 0
    assert len(large_gaps) > 0  # proves bursts are actually separated, not one giant burst


def test_phone_events_duration_matches_lognormal_shape() -> None:
    """Most durations should be short (5-60s per spec), with an
    occasional long tail, not a flat/uniform distribution."""

    participant = make_participant()
    generator = PhoneEventGenerator(
        burst_count_range=(8, 8),
        burst_event_count_range=(15, 15),
    )

    records = generator.generate(participant, TEST_DAY, Random(6))
    durations = [float(r["duration_seconds"]) for r in records]

    short_durations = [d for d in durations if d <= 60]

    # Majority should be short, per spec ("most interactions 5-60s")
    assert len(short_durations) / len(durations) > 0.5


def test_phone_events_duration_never_exceeds_cap() -> None:
    participant = make_participant()
    generator = PhoneEventGenerator(duration_cap_seconds=1200.0)

    records = generator.generate(participant, TEST_DAY, Random(6))

    assert all(float(r["duration_seconds"]) <= 1200.0 for r in records)


def test_phone_events_event_type_is_one_of_the_defined_set() -> None:
    from synthetic.generators.phone_events import EVENT_TYPES

    participant = make_participant()
    generator = PhoneEventGenerator()

    records = generator.generate(participant, TEST_DAY, Random(6))

    assert all(r["event_type"] in EVENT_TYPES for r in records)


def test_phone_events_records_carry_participant_id() -> None:
    participant = make_participant()
    generator = PhoneEventGenerator()

    records = generator.generate(participant, TEST_DAY, Random(6))

    assert all(r["participant_id"] == participant.participant_id for r in records)
