"""surveys.ema stream generator — spec Section 5.8."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from random import Random

from synthetic.config import Participant
from synthetic.dailystate import daily_stress_intensity
from synthetic.generators.base import Record
from synthetic.timeutils import day_start


@dataclass(frozen=True, slots=True)
class EmaSurveyGenerator:
    """Produces periodic self-report stress surveys, plus onboarding/
    closing surveys on a participant's first/last active day.

    Per spec Section 5.8:
      - stress_1_to_5 delivered 3-5 times per simulated day.
      - The signal loosely correlates with, but is not perfectly
        derived from, the day's physiological stress signal — achieved
        here via `daily_stress_intensity`, a value shared with (but not
        directly read from) other generators, plus independent
        per-response noise.
    """

    responses_per_day_range: tuple[int, int] = (3, 5)
    day_start_hour: int = 8
    day_end_hour: int = 21

    def generate(self, participant: Participant, day: date, rng: Random) -> list[Record]:
        records: list[Record] = []

        intensity = daily_stress_intensity(participant.participant_id, day)

        response_count = rng.randint(*self.responses_per_day_range)

        window_start = day_start(day) + timedelta(hours=self.day_start_hour)
        window_end = day_start(day) + timedelta(hours=self.day_end_hour)
        window_seconds = (window_end - window_start).total_seconds()

        for index in range(response_count):
            offset_seconds = rng.uniform(0, window_seconds)
            timestamp = window_start + timedelta(seconds=offset_seconds)

            stress_score = self._sample_stress_score(intensity, rng)

            records.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "participant_id": participant.participant_id,
                    "survey_type": "periodic",
                    "stress_1_to_5": stress_score,
                    "response_id": (
                        f"{participant.participant_id}_{day.isoformat()}_ema_{index:02d}"
                    ),
                }
            )

        if participant.active_days and day == participant.active_days[0]:
            records.append(self._boundary_survey(participant, day, "onboarding", intensity, rng))

        if participant.active_days and day == participant.active_days[-1]:
            records.append(self._boundary_survey(participant, day, "closing", intensity, rng))

        records.sort(key=lambda record: record["timestamp"])  # type: ignore[arg-type,return-value]

        return records

    def _sample_stress_score(self, intensity: float, rng: Random) -> int:
        """Map the shared daily intensity to a 1-5 score, with
        independent noise so the self-report is loosely correlated with
        — but not perfectly derived from — the underlying intensity
        (spec Section 5.8's explicit requirement)."""

        base = 1.0 + intensity * 4.0  # maps [0,1] -> [1,5]
        noisy = base + rng.gauss(0.0, 0.9)  # independent per-response noise
        clamped = min(5.0, max(1.0, noisy))
        return round(clamped)

    def _boundary_survey(
        self,
        participant: Participant,
        day: date,
        survey_type: str,
        intensity: float,
        rng: Random,
    ) -> Record:
        timestamp = day_start(day) + timedelta(hours=self.day_start_hour)
        stress_score = self._sample_stress_score(intensity, rng)

        return {
            "timestamp": timestamp.isoformat(),
            "participant_id": participant.participant_id,
            "survey_type": survey_type,
            "stress_1_to_5": stress_score,
            "response_id": f"{participant.participant_id}_{day.isoformat()}_ema_{survey_type}",
        }
