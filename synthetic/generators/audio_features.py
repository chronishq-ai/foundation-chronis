"""audio_features.summary stream generator — spec Section 5.4.

Per the spec's hard rules: derived stats only, never actual audio or
transcripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from random import Random

from synthetic.config import Participant
from synthetic.generators.base import Record
from synthetic.timeutils import day_start, irregular_timestamps_between


@dataclass(frozen=True, slots=True)
class AudioFeatureGenerator:
    """Produces derived audio-statistic rows during simulated "at work"
    hours only.

    Per spec Section 5.4: a 0-1 "voice activity" score and a rough
    pitch-variance number, one row roughly every 1-5 minutes, restricted
    to at-work hours — never actual audio or transcript content.
    """

    work_start_hour: int = 9
    work_end_hour: int = 17

    min_gap_minutes: float = 1.0
    max_gap_minutes: float = 5.0

    def generate(self, participant: Participant, day: date, rng: Random) -> list[Record]:
        work_start = day_start(day) + timedelta(hours=self.work_start_hour)
        work_end = day_start(day) + timedelta(hours=self.work_end_hour)

        timestamps = irregular_timestamps_between(
            work_start,
            work_end,
            min_gap_seconds=self.min_gap_minutes * 60,
            max_gap_seconds=self.max_gap_minutes * 60,
            rng=rng,
        )

        records: list[Record] = []
        for timestamp in timestamps:
            voice_activity_score = min(1.0, max(0.0, rng.betavariate(2, 3)))
            pitch_variance = max(0.0, rng.gauss(15.0, 6.0))

            records.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "participant_id": participant.participant_id,
                    "voice_activity_score": round(voice_activity_score, 3),
                    "pitch_variance": round(pitch_variance, 2),
                }
            )

        return records
