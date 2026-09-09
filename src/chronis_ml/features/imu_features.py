"""S2.5 — IMU feature extraction.

Fixes the audit-flagged gap: the original IMU pipeline computed only a
scalar mean/variance, with no per-axis detail, no movement semantics,
and no signal-quality gating.

Required outputs, each with a corresponding test:
  - Per-axis (x/y/z) values present, not only a scalar magnitude.
  - Stillness duration and movement intensity, correctly ~0/full-window
    for a genuinely flat/still signal.
  - A cadence estimate for a walking-pattern fixture, within tolerance
    of the known stepping frequency.
  - Signal-quality (SQI) gating: a corrupted/clipped segment must be
    FLAGGED, and downstream interpretive features (posture, cadence)
    must be marked degraded rather than confidently computed as if the
    signal were clean.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

GRAVITY_G = 1.0
"""Baseline magnitude (in g) expected for a stationary device under
gravity alone."""


class SignalQuality(StrEnum):
    CLEAN = "clean"
    DEGRADED = "degraded"


class PostureState(StrEnum):
    STILL = "still"
    ACTIVE = "active"
    UNKNOWN = "unknown"
    """Used whenever the signal is DEGRADED - posture is never
    confidently asserted on a corrupted signal."""


@dataclass(frozen=True, slots=True)
class IMUSample:
    timestamp: datetime
    x: float
    y: float
    z: float

    @property
    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)


@dataclass(frozen=True, slots=True)
class IMUFeatureSet:
    user_id: str
    window_start: datetime
    window_end: datetime

    x_mean: float
    y_mean: float
    z_mean: float
    x_variance: float
    y_variance: float
    z_variance: float

    magnitude_mean: float
    posture_state: PostureState
    stillness_duration_seconds: float
    movement_intensity: float
    cadence_hz: float | None
    """None whenever no reliable cadence estimate is available (empty
    window or a DEGRADED signal) - never a fabricated number."""

    quality: SignalQuality
    quality_reason: str | None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _variance(values: Sequence[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return sum((v - mean) ** 2 for v in values) / len(values)


def _assess_signal_quality(
    samples: Sequence[IMUSample],
    *,
    clip_threshold_g: float,
    stuck_run_threshold: int,
) -> tuple[SignalQuality, str | None]:
    """Detects two realistic corruption modes: sensor saturation
    (magnitude implausibly high for human movement) and a stuck/frozen
    sensor (a long run of bit-identical consecutive samples)."""

    for sample in samples:
        if sample.magnitude >= clip_threshold_g:
            return (
                SignalQuality.DEGRADED,
                f"magnitude {sample.magnitude:.2f}g at or beyond "
                f"clip threshold {clip_threshold_g}g",
            )

    run_length = 1
    for earlier, later in zip(samples, samples[1:], strict=False):
        if (earlier.x, earlier.y, earlier.z) == (later.x, later.y, later.z):
            run_length += 1
            if run_length >= stuck_run_threshold:
                return (
                    SignalQuality.DEGRADED,
                    f"{run_length} consecutive identical samples (stuck/frozen sensor)",
                )
        else:
            run_length = 1

    return SignalQuality.CLEAN, None


def _count_positive_zero_crossings(values: Sequence[float], baseline: float) -> int:
    """Counts how many times the signal crosses from at-or-below
    baseline to above baseline - each crossing corresponds to one
    stepping cycle for a clean, periodic walking signal."""

    crossings = 0
    previous_above = values[0] > baseline

    for value in values[1:]:
        current_above = value > baseline
        if current_above and not previous_above:
            crossings += 1
        previous_above = current_above

    return crossings


def extract_imu_features(
    user_id: str,
    samples: Sequence[IMUSample],
    *,
    stillness_tolerance_g: float = 0.05,
    clip_threshold_g: float = 8.0,
    stuck_run_threshold: int = 5,
) -> IMUFeatureSet:
    """Extract axis-aware IMU features from one window of raw samples."""

    if not samples:
        raise ValueError("samples must not be empty")

    x_values = [s.x for s in samples]
    y_values = [s.y for s in samples]
    z_values = [s.z for s in samples]
    magnitudes = [s.magnitude for s in samples]

    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    z_mean = _mean(z_values)

    quality, quality_reason = _assess_signal_quality(
        samples, clip_threshold_g=clip_threshold_g, stuck_run_threshold=stuck_run_threshold
    )

    magnitude_mean = _mean(magnitudes)
    movement_intensity = _mean([abs(m - GRAVITY_G) for m in magnitudes])

    # Stillness duration: sum of time between consecutive samples where
    # BOTH endpoints are within tolerance of the gravity baseline.
    stillness_duration = 0.0
    for earlier, later in zip(samples, samples[1:], strict=False):
        earlier_still = abs(earlier.magnitude - GRAVITY_G) <= stillness_tolerance_g
        later_still = abs(later.magnitude - GRAVITY_G) <= stillness_tolerance_g
        if earlier_still and later_still:
            stillness_duration += (later.timestamp - earlier.timestamp).total_seconds()

    window_duration = (samples[-1].timestamp - samples[0].timestamp).total_seconds()

    if quality is SignalQuality.DEGRADED:
        # Never confidently assert posture or cadence on a corrupted signal.
        posture_state = PostureState.UNKNOWN
        cadence_hz = None
    else:
        if window_duration > 0 and stillness_duration >= window_duration - 1e-9:
            posture_state = PostureState.STILL
        elif movement_intensity > stillness_tolerance_g:
            posture_state = PostureState.ACTIVE
        else:
            posture_state = PostureState.STILL

        if window_duration > 0 and len(samples) >= 3:
            crossings = _count_positive_zero_crossings(magnitudes, GRAVITY_G)
            cadence_hz = crossings / window_duration if crossings > 0 else None
        else:
            cadence_hz = None

    return IMUFeatureSet(
        user_id=user_id,
        window_start=samples[0].timestamp,
        window_end=samples[-1].timestamp,
        x_mean=x_mean,
        y_mean=y_mean,
        z_mean=z_mean,
        x_variance=_variance(x_values, x_mean),
        y_variance=_variance(y_values, y_mean),
        z_variance=_variance(z_values, z_mean),
        magnitude_mean=magnitude_mean,
        posture_state=posture_state,
        stillness_duration_seconds=stillness_duration,
        movement_intensity=movement_intensity,
        cadence_hz=cadence_hz,
        quality=quality,
        quality_reason=quality_reason,
    )
