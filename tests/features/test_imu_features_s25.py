"""S2.5 test sheet - original 4 test cases.

Test Sheet - S2.5:
  T1: Synthetic flat/still signal -> Extract -> stillness_duration ==
      full window; movement intensity ~ 0
  T2: Synthetic walking-cadence fixture at a known step frequency ->
      Extract -> Cadence estimate within tolerance of the known
      frequency
  T3: Any output -> Inspect -> Per-axis (x/y/z) values present, not
      only a scalar magnitude
  T4: Synthetic corrupted/clipped segment -> Extract -> SQI flags the
      segment; downstream features marked degraded, not computed as if
      clean
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from chronis_ml.features.imu_features import (
    IMUSample,
    PostureState,
    SignalQuality,
    extract_imu_features,
)

USER_ID = "user_001"
START = datetime(2026, 8, 16, 9, 0, tzinfo=UTC)


def build_flat_signal(duration_seconds: float, hz: float = 10.0) -> list[IMUSample]:
    """A still device with tiny, realistic sensor noise (magnitude
    hovers around 1.0g). Deliberately NOT bit-identical across samples
    -- a real accelerometer always has a small noise floor even at
    rest; bit-identical consecutive readings are themselves a
    stuck-sensor signature (see T4), not a realistic "still" signal."""

    sample_count = int(duration_seconds * hz)
    return [
        IMUSample(
            timestamp=START + timedelta(seconds=i / hz),
            x=0.0,
            y=0.0,
            z=1.0 + (0.001 if i % 2 == 0 else -0.001),  # tiny deterministic jitter
        )
        for i in range(sample_count)
    ]


def build_walking_signal(
    duration_seconds: float,
    cadence_hz: float,
    sample_rate_hz: float = 20.0,
) -> list[IMUSample]:
    """A clean sinusoidal magnitude oscillation at a known cadence,
    riding on top of the 1g gravity baseline -- simulates a walking
    accelerometer signal."""

    sample_count = int(duration_seconds * sample_rate_hz)
    samples = []
    for i in range(sample_count):
        t = i / sample_rate_hz
        oscillation = 0.3 * math.sin(2 * math.pi * cadence_hz * t)
        magnitude = GRAVITY_G_FOR_TEST + oscillation
        # Put the full oscillation on the z-axis for simplicity; x/y stay at 0.
        samples.append(IMUSample(timestamp=START + timedelta(seconds=t), x=0.0, y=0.0, z=magnitude))
    return samples


GRAVITY_G_FOR_TEST = 1.0


# --- T1: flat/still signal ----------------------------------------------------


def test_s25_t1_flat_signal_stillness_duration_is_full_window() -> None:
    samples = build_flat_signal(duration_seconds=10.0, hz=10.0)

    result = extract_imu_features(USER_ID, samples)

    window_duration = (samples[-1].timestamp - samples[0].timestamp).total_seconds()

    assert result.posture_state is PostureState.STILL
    assert abs(result.stillness_duration_seconds - window_duration) < 1e-6
    assert result.movement_intensity < 0.01  # ~= 0, allowing for realistic sensor noise
    assert result.quality is SignalQuality.CLEAN


# --- T2: walking-cadence fixture at a known frequency ------------------------


def test_s25_t2_cadence_estimate_matches_known_frequency() -> None:
    known_cadence_hz = 1.5  # ~90 steps/min, a typical walking pace

    samples = build_walking_signal(duration_seconds=20.0, cadence_hz=known_cadence_hz)

    result = extract_imu_features(USER_ID, samples)

    assert result.cadence_hz is not None
    assert abs(result.cadence_hz - known_cadence_hz) < 0.15  # within tolerance
    assert result.quality is SignalQuality.CLEAN


def test_s25_t2_cadence_estimate_scales_with_a_different_known_frequency() -> None:
    """Sanity check that the estimator isn't hardcoded to one
    frequency -- a different known cadence should be recovered too."""

    known_cadence_hz = 2.0

    samples = build_walking_signal(duration_seconds=20.0, cadence_hz=known_cadence_hz)

    result = extract_imu_features(USER_ID, samples)

    assert result.cadence_hz is not None
    assert abs(result.cadence_hz - known_cadence_hz) < 0.15


# --- T3: per-axis values present, not only a scalar magnitude ---------------


def test_s25_t3_output_carries_per_axis_values_not_just_magnitude() -> None:
    samples = build_walking_signal(duration_seconds=5.0, cadence_hz=1.5)

    result = extract_imu_features(USER_ID, samples)

    # Per-axis fields must genuinely exist and be independently inspectable.
    assert hasattr(result, "x_mean")
    assert hasattr(result, "y_mean")
    assert hasattr(result, "z_mean")
    assert hasattr(result, "x_variance")
    assert hasattr(result, "y_variance")
    assert hasattr(result, "z_variance")

    # The per-axis values must be genuinely distinguishable, not all
    # collapsed to the same scalar (our test fixture puts all motion on
    # z, so x/y should differ meaningfully from z).
    assert result.z_variance > result.x_variance
    assert result.z_variance > result.y_variance


# --- T4: corrupted/clipped segment -> SQI flags, features degraded --------


def test_s25_t4_clipped_segment_is_flagged_degraded() -> None:
    """A sample with an implausibly high magnitude (sensor saturation)
    must flag the whole window as DEGRADED."""

    samples = build_flat_signal(duration_seconds=5.0, hz=10.0)
    clipped_samples = list(samples)
    clipped_samples[25] = IMUSample(
        timestamp=clipped_samples[25].timestamp,
        x=0.0,
        y=0.0,
        z=50.0,  # implausible spike
    )

    result = extract_imu_features(USER_ID, clipped_samples)

    assert result.quality is SignalQuality.DEGRADED
    assert result.quality_reason is not None
    assert "clip threshold" in result.quality_reason


def test_s25_t4_degraded_signal_does_not_confidently_assert_posture_or_cadence() -> None:
    samples = build_flat_signal(duration_seconds=5.0, hz=10.0)
    clipped_samples = list(samples)
    clipped_samples[25] = IMUSample(timestamp=clipped_samples[25].timestamp, x=0.0, y=0.0, z=50.0)

    result = extract_imu_features(USER_ID, clipped_samples)

    assert result.posture_state is PostureState.UNKNOWN  # never STILL/ACTIVE on bad data
    assert result.cadence_hz is None  # never a fabricated cadence number


def test_s25_t4_stuck_sensor_run_is_also_flagged_degraded() -> None:
    """A long run of bit-identical samples (a frozen/stuck sensor) is a
    distinct corruption mode from clipping, and must also be caught."""

    samples = [
        IMUSample(timestamp=START + timedelta(seconds=i * 0.1), x=0.3, y=0.3, z=0.7)
        for i in range(10)
    ]  # 10 bit-identical samples

    result = extract_imu_features(USER_ID, samples)

    assert result.quality is SignalQuality.DEGRADED
    assert "stuck" in result.quality_reason.lower() or "frozen" in result.quality_reason.lower()


def test_s25_t4_clean_signal_is_never_flagged_degraded() -> None:
    """Positive control: a genuinely clean signal must not be
    false-positive flagged."""

    samples = build_walking_signal(duration_seconds=5.0, cadence_hz=1.5)

    result = extract_imu_features(USER_ID, samples)

    assert result.quality is SignalQuality.CLEAN
    assert result.quality_reason is None
