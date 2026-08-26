"""fitbit.sleep stream generator.

The spec document lists "sleep" as part of the fitbit stream in its
folder/modality table (Section 3) but does not give it a numbered
Section-5 sub-spec with explicit ranges, unlike heart_rate/steps/etc.
This generator's stage-cycle structure and durations are therefore our
own reasonable design, not derived from any numbered spec section —
documented here explicitly so nobody mistakes these specific numbers
for a spec requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from random import Random

from synthetic.config import Participant
from synthetic.generators.base import Record
from synthetic.timeutils import day_start

STAGE_CYCLE: tuple[str, ...] = ("light", "deep", "rem", "light")
"""One simplified sleep cycle. Real sleep architecture is more complex;
this is a plausible-enough approximation for a test fixture, not a
sleep-science claim."""


@dataclass(frozen=True, slots=True)
class FitbitSleepGenerator:
    """Produces one sleep session per day, starting the previous night.

    The session begins between roughly 22:00-23:59 on the given `day`
    and extends into the following calendar day, matching how a night's
    sleep is naturally anchored to the day it started on.
    """

    sleep_start_hour_range: tuple[float, float] = (22.0, 23.99)
    total_sleep_minutes_mean: float = 420.0  # 7 hours
    total_sleep_minutes_sd: float = 60.0
    total_sleep_minutes_bounds: tuple[float, float] = (180.0, 600.0)  # 3-10 hours

    cycle_minutes_mean: float = 90.0
    cycle_minutes_sd: float = 15.0

    awake_interruption_probability: float = 0.15
    """Probability of a brief awake interruption between cycles."""
    awake_interruption_minutes_range: tuple[float, float] = (2.0, 15.0)

    def generate(self, participant: Participant, day: date, rng: Random) -> list[Record]:
        start = day_start(day) + timedelta(hours=rng.uniform(*self.sleep_start_hour_range))

        total_minutes = min(
            max(
                rng.gauss(self.total_sleep_minutes_mean, self.total_sleep_minutes_sd),
                self.total_sleep_minutes_bounds[0],
            ),
            self.total_sleep_minutes_bounds[1],
        )

        session_id = f"{participant.participant_id}_{day.isoformat()}_sleep"

        segments = self._build_stage_segments(total_minutes, rng)

        records: list[Record] = []
        cursor = start

        for stage, duration_minutes in segments:
            segment_start = cursor
            segment_end = segment_start + timedelta(minutes=duration_minutes)

            records.append(
                {
                    "sleep_session_id": session_id,
                    "participant_id": participant.participant_id,
                    "start_time": segment_start.isoformat(),
                    "end_time": segment_end.isoformat(),
                    "stage": stage,
                    "stage_duration_seconds": round(duration_minutes * 60.0, 1),
                }
            )

            cursor = segment_end

        return records

    def _build_stage_segments(self, total_minutes: float, rng: Random) -> list[tuple[str, float]]:
        segments: list[tuple[str, float]] = []
        remaining = total_minutes
        cycle_index = 0

        while remaining > 0:
            for stage in STAGE_CYCLE:
                if remaining <= 0:
                    break

                stage_minutes = min(
                    max(
                        rng.gauss(
                            self.cycle_minutes_mean / len(STAGE_CYCLE), self.cycle_minutes_sd / 2
                        ),
                        1.0,
                    ),
                    remaining,
                )
                segments.append((stage, stage_minutes))
                remaining -= stage_minutes

            if remaining > 0 and rng.random() < self.awake_interruption_probability:
                awake_minutes = min(rng.uniform(*self.awake_interruption_minutes_range), remaining)
                segments.append(("awake", awake_minutes))
                remaining -= awake_minutes

            cycle_index += 1
            if cycle_index > 20:  # safety valve against pathological configs
                break

        return segments
