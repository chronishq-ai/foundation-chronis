"""fitbit.heart_rate stream generator — spec Section 5.1.

Reference exemplar #1 for the generator registry pattern (spec Section
11, Step 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from random import Random

from synthetic.config import Participant
from synthetic.generators.base import Record
from synthetic.timeutils import irregular_timestamps, is_within_any_window, random_windows


@dataclass(frozen=True, slots=True)
class FitbitHRGenerator:
    """Produces irregular heart-rate readings across one full day.

    Behavior, per spec Section 5.1:
      - Irregular sampling every 5-120 seconds, never a perfect grid.
      - A per-participant resting baseline (already drawn once at roster
        build time — see `synthetic.config.build_roster`) is the default
        value outside any special window.
      - Randomly placed "active" windows add a temporary offset drawn
        from Normal(mean=35, sd=15).
      - Randomly placed "stress event" windows add a temporary offset
        drawn from Normal(mean=20, sd=8), with extra faster
        micro-fluctuation layered on top.
      - Occasional flat-line runs (30-80 identical consecutive values)
        are injected at a configurable rate, simulating a sensor stall
        that downstream code must learn to treat as "repeated value ->
        likely stalled," not as 80 independent genuine readings.
    """

    min_gap_seconds: float = 5.0
    max_gap_seconds: float = 120.0

    active_window_count_range: tuple[int, int] = (1, 3)
    active_window_minutes_range: tuple[int, int] = (15, 60)

    stress_window_count_range: tuple[int, int] = (0, 2)
    stress_window_minutes_range: tuple[int, int] = (10, 30)

    flat_line_probability: float = 0.05
    """Probability that any given day's output contains one injected
    flat-line run. Kept separate from the general corruption-injection
    system (spec Section 6) because this is a sensor-specific realism
    detail described directly in Section 5.1, not a generic cross-stream
    corruption mode."""

    flat_line_run_length_range: tuple[int, int] = (30, 80)

    minimum_physiological_hr: float = 30.0
    """Floor applied after adding window offsets, so noise can never
    push a value into an impossible (near-zero or negative) heart rate."""

    baseline_noise_sd: float = 2.5
    """Small always-on Gaussian noise applied even outside any active or
    stress window. Without this, resting readings would be the exact
    same constant value every time, which is not physiologically
    realistic and would produce accidental long flat-line runs
    indistinguishable from the deliberate `flat_line_probability`
    injection below."""

    def generate(self, participant: Participant, day: date, rng: Random) -> list[Record]:
        timestamps = irregular_timestamps(
            day,
            min_gap_seconds=self.min_gap_seconds,
            max_gap_seconds=self.max_gap_seconds,
            rng=rng,
        )

        active_windows = random_windows(
            day,
            count_range=self.active_window_count_range,
            duration_minutes_range=self.active_window_minutes_range,
            rng=rng,
        )
        stress_windows = random_windows(
            day,
            count_range=self.stress_window_count_range,
            duration_minutes_range=self.stress_window_minutes_range,
            rng=rng,
        )

        values = [
            self._value_at(
                timestamp=timestamp,
                baseline=participant.resting_heart_rate,
                active_windows=active_windows,
                stress_windows=stress_windows,
                rng=rng,
            )
            for timestamp in timestamps
        ]

        values = self._maybe_inject_flat_line(values, rng)

        return [
            {
                "timestamp": timestamp.isoformat(),
                "participant_id": participant.participant_id,
                "heart_rate_bpm": round(value, 1),
            }
            for timestamp, value in zip(timestamps, values, strict=True)
        ]

    def _value_at(
        self,
        *,
        timestamp: datetime,
        baseline: float,
        active_windows: tuple[tuple[datetime, datetime], ...],
        stress_windows: tuple[tuple[datetime, datetime], ...],
        rng: Random,
    ) -> float:
        value = baseline + rng.gauss(0.0, self.baseline_noise_sd)

        if is_within_any_window(timestamp, stress_windows):
            value += rng.gauss(20.0, 8.0)
            value += rng.gauss(0.0, 3.0)  # extra micro-fluctuation during stress
        elif is_within_any_window(timestamp, active_windows):
            value += rng.gauss(35.0, 15.0)

        return max(self.minimum_physiological_hr, value)

    def _maybe_inject_flat_line(self, values: list[float], rng: Random) -> list[float]:
        if not values or rng.random() >= self.flat_line_probability:
            return values

        run_length = rng.randint(*self.flat_line_run_length_range)
        run_length = min(run_length, len(values))

        if run_length < 1:
            return values

        start_index = rng.randint(0, len(values) - run_length)
        flat_value = values[start_index]

        result = list(values)
        for offset in range(run_length):
            result[start_index + offset] = flat_value

        return result
