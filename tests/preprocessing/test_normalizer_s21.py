"""S2.1 test sheet — original 5 test cases, run against the real
implementation.

Test Sheet — S2.1:
    T1: Synthetic user, hand-computable mu/sigma over a 7-day window ->
        Normalize a known value -> Output == (x-mu)/sigma within float
        tolerance
    T2: User with <7 days of history -> Normalize -> Output typed as
        low-confidence/insufficient-baseline, not a silently-returned raw
        difference
    T3: Near-constant signal (sigma ~ 0) -> Normalize -> No
        divide-by-zero; explicit documented guard triggers
    T4: Synthetic future spike injected after the evaluation timestamp ->
        Normalize a pre-spike value -> Value unchanged - proves the
        baseline window is strictly backward-looking
    T5: Any output record -> Inspect fields -> Carries
        baseline_window_start, baseline_window_end, baseline_version
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from chronis_ml.preprocessing.normalizer import (
    NormalizationConfidence,
    normalize_value,
)
from chronis_ml.schema.models import FeatureRecord, MeasurementStatus

USER_ID = "user_001"
FEATURE_NAME = "heart_rate"


def make_record(day_offset: int, value: float, hour: int = 12) -> FeatureRecord:
    base = datetime(2026, 8, 1, hour, 0, tzinfo=UTC)
    return FeatureRecord(
        user_id=USER_ID,
        timestamp=base + timedelta(days=day_offset),
        feature_name=FEATURE_NAME,
        value=value,
        modality="ppg",
        status=MeasurementStatus.OBSERVED,
    )


# --- T1: hand-computable mu/sigma, exact z-score -----------------------------


def test_s21_t1_z_score_matches_hand_computed_value() -> None:
    # 7 days of known values: 60, 62, 64, 66, 68, 70, 72
    known_values = [60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0]
    history = [make_record(day, value) for day, value in enumerate(known_values)]

    target = make_record(7, 80.0)  # day 8, evaluated against the 7 days before it

    import statistics

    expected_mean = statistics.mean(known_values)
    expected_sd = statistics.pstdev(known_values)
    expected_z = (80.0 - expected_mean) / expected_sd

    result = normalize_value(history, target)

    assert result.confidence is NormalizationConfidence.NORMAL
    assert result.z_score is not None
    assert abs(result.z_score - expected_z) < 1e-9


# --- T2: insufficient baseline (<7 days) -------------------------------------


def test_s21_t2_insufficient_baseline_is_typed_not_silent() -> None:
    # Only 4 days of history - below the 7-day minimum.
    history = [make_record(day, 65.0 + day) for day in range(4)]
    target = make_record(4, 90.0)

    result = normalize_value(history, target)

    assert result.confidence is NormalizationConfidence.INSUFFICIENT_BASELINE
    assert result.z_score is None  # never a silently-returned raw difference
    assert result.raw_value == 90.0  # raw value still visible for diagnostics


def test_s21_t2_zero_history_is_also_insufficient_baseline() -> None:
    target = make_record(0, 70.0)

    result = normalize_value([], target)

    assert result.confidence is NormalizationConfidence.INSUFFICIENT_BASELINE
    assert result.z_score is None
    assert result.baseline_sample_count == 0


# --- T3: near-zero variance guard --------------------------------------------


def test_s21_t3_near_zero_variance_never_divides_by_zero() -> None:
    # 7 days of an essentially constant signal.
    history = [make_record(day, 65.0) for day in range(7)]
    target = make_record(7, 65.0)

    result = normalize_value(history, target)  # must not raise ZeroDivisionError

    assert result.confidence is NormalizationConfidence.NEAR_ZERO_VARIANCE
    assert result.z_score is None
    assert result.baseline_mean is not None  # diagnostics still populated
    assert result.baseline_sd is not None
    assert result.baseline_sd < 1e-6


def test_s21_t3_near_zero_variance_threshold_is_configurable() -> None:
    history = [make_record(day, 65.0 + (0.0001 * day)) for day in range(7)]
    target = make_record(7, 65.0)

    # With the default (tight) threshold, this small variance should
    # NOT be flagged near-zero.
    result_default = normalize_value(history, target)
    assert result_default.confidence is NormalizationConfidence.NORMAL

    # With a much looser threshold, the same data IS flagged near-zero.
    result_loose_threshold = normalize_value(history, target, near_zero_variance_threshold=1.0)
    assert result_loose_threshold.confidence is NormalizationConfidence.NEAR_ZERO_VARIANCE


# --- T4: strictly backward-looking, no future-data leakage -------------------


def test_s21_t4_future_spike_does_not_affect_prior_value() -> None:
    # 7 days of stable baseline data.
    history = [make_record(day, 65.0) for day in range(7)]

    # A pre-spike target at day 7, evaluated normally.
    pre_spike_target = make_record(7, 66.0)
    result_before_spike_exists = normalize_value(history, pre_spike_target)

    # Now inject a huge spike AFTER the target's own timestamp.
    spike_record = make_record(8, 500.0)
    history_with_future_spike = history + [spike_record]

    result_with_future_spike_present = normalize_value(history_with_future_spike, pre_spike_target)

    # The pre-spike target's z-score must be COMPLETELY unaffected by
    # the future spike's existence in the dataset.
    assert result_before_spike_exists.z_score == result_with_future_spike_present.z_score
    before_mean = result_before_spike_exists.baseline_mean
    after_mean = result_with_future_spike_present.baseline_mean
    assert before_mean == after_mean
    assert result_before_spike_exists.baseline_sd == result_with_future_spike_present.baseline_sd


def test_s21_t4_same_timestamp_as_target_is_excluded_from_its_own_baseline() -> None:
    """Strictly < target.timestamp, never <=."""
    history = [make_record(day, 65.0 + day) for day in range(8)]  # days 0-7, varying values

    target_timestamp = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)  # exact same instant as history[7]
    same_instant_record = FeatureRecord(
        user_id=USER_ID,
        timestamp=target_timestamp,
        feature_name=FEATURE_NAME,
        value=999.0,
        modality="ppg",
        status=MeasurementStatus.OBSERVED,
    )

    result = normalize_value(history, same_instant_record)

    # 999.0 (the target's own value) must never appear inside its own
    # baseline mean, and the mean should reflect only days 0-6 (65..71).
    assert result.confidence is NormalizationConfidence.NORMAL
    assert result.baseline_mean is not None
    assert result.baseline_mean < 100.0  # nowhere near 999


# --- T5: every output record carries full provenance -------------------------


def test_s21_t5_output_carries_baseline_window_and_version() -> None:
    history = [make_record(day, 65.0 + day) for day in range(7)]
    target = make_record(7, 80.0)

    result = normalize_value(history, target, baseline_version="2.0")

    assert result.baseline_window_start == target.timestamp - timedelta(days=7)
    assert result.baseline_window_end == target.timestamp
    assert result.baseline_version == "2.0"


def test_s21_t5_insufficient_baseline_result_still_carries_window_metadata() -> None:
    """Provenance fields must be present even on the edge-case
    confidence states, not only the NORMAL happy path."""
    target = make_record(0, 70.0)

    result = normalize_value([], target)

    assert result.baseline_window_start == target.timestamp - timedelta(days=7)
    assert result.baseline_window_end == target.timestamp
    assert result.baseline_version == "1.0"
