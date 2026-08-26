"""fitbit.steps stream generator — spec Section 5.2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from random import Random

from synthetic.config import Participant
from synthetic.generators.base import Record
from synthetic.timeutils import day_start


@dataclass(frozen=True, slots=True)
class FitbitStepsGenerator:
    """1-minute step-count buckets across a full day.

    Per spec Section 5.2: 0 steps for ~70% of buckets (idle/sleep),
    otherwise Poisson(lambda=40) capped at 130.
    """

    idle_bucket_probability: float = 0.70
    poisson_lambda: float = 40.0
    max_steps_per_bucket: int = 130

    def generate(self, participant: Participant, day: date, rng: Random) -> list[Record]:
        start = day_start(day)
        bucket_count = 24 * 60  # 1-minute buckets across the full day

        records: list[Record] = []

        for bucket_index in range(bucket_count):
            window_start = start + timedelta(minutes=bucket_index)
            window_end = window_start + timedelta(minutes=1)

            if rng.random() < self.idle_bucket_probability:
                step_count = 0
            else:
                step_count = min(self._poisson(self.poisson_lambda, rng), self.max_steps_per_bucket)

            records.append(
                {
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "participant_id": participant.participant_id,
                    "step_count": step_count,
                }
            )

        return records

    @staticmethod
    def _poisson(lam: float, rng: Random) -> int:
        """Knuth's algorithm — stdlib `random` has no built-in Poisson
        sampler."""
        import math

        limit = math.exp(-lam)
        k = 0
        p = 1.0

        while True:
            k += 1
            p *= rng.random()
            if p <= limit:
                return k - 1
