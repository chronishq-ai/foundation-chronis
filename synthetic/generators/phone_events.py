"""phone_events.interaction stream generator — spec Section 5.7.

Reference exemplar #2 for the generator registry pattern (spec Section
11, Step 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from random import Random

from synthetic.config import Participant
from synthetic.generators.base import Record
from synthetic.timeutils import day_start

EVENT_TYPES: tuple[str, ...] = ("screen_on", "app_open", "app_switch", "notification_view")


@dataclass(frozen=True, slots=True)
class PhoneEventGenerator:
    """Produces a bursty, clustered log of phone-interaction events.

    Behavior, per spec Section 5.7:
      - Each event is (timestamp, duration_seconds).
      - Duration is drawn from a log-normal distribution: most
        interactions are 5-60 seconds, with occasional long sessions of
        5-20 minutes.
      - Frequency is deliberately NOT uniform across the day: fewer
        events overnight, clustered bursts during simulated breaks —
        matching the spec's explicit warning that phone/proximity-style
        events must be bursty and clustered, not evenly spaced.
    """

    burst_count_range: tuple[int, int] = (4, 8)
    burst_event_count_range: tuple[int, int] = (3, 15)
    burst_window_minutes: int = 15
    """A burst's events are clustered within this many minutes."""

    overnight_event_probability: float = 0.15
    """Probability of a handful of isolated, non-bursty events occurring
    overnight (00:00-06:00), representing occasional real overnight
    phone checks rather than complete silence."""

    overnight_event_count_range: tuple[int, int] = (0, 3)

    duration_lognormal_mu: float = 3.0
    """Underlying normal-distribution mean for log-normal duration
    sampling. exp(3.0) ~= 20s median duration, matching "most
    interactions 5-60s"."""

    duration_lognormal_sigma: float = 0.9
    """Controls the long tail — higher sigma produces more of the
    occasional 5-20 minute long sessions the spec calls for."""

    duration_cap_seconds: float = 1200.0  # 20 minutes, per spec's stated upper bound

    def generate(self, participant: Participant, day: date, rng: Random) -> list[Record]:
        records: list[Record] = []

        start = day_start(day)

        # Daytime bursts, clustered.
        burst_count = rng.randint(*self.burst_count_range)
        for _ in range(burst_count):
            burst_start_offset = rng.uniform(6 * 3600, 23 * 3600)  # 06:00-23:00
            burst_start = start + timedelta(seconds=burst_start_offset)

            event_count = rng.randint(*self.burst_event_count_range)
            for _ in range(event_count):
                event_offset = rng.uniform(0, self.burst_window_minutes * 60)
                timestamp = burst_start + timedelta(seconds=event_offset)

                records.append(self._build_event(participant, timestamp, rng))

        # Sparse overnight events.
        if rng.random() < self.overnight_event_probability:
            overnight_count = rng.randint(*self.overnight_event_count_range)
            for _ in range(overnight_count):
                overnight_offset = rng.uniform(0, 6 * 3600)  # 00:00-06:00
                timestamp = start + timedelta(seconds=overnight_offset)

                records.append(self._build_event(participant, timestamp, rng))

        records.sort(key=lambda record: record["timestamp"])  # type: ignore[arg-type,return-value]

        return records

    def _build_event(self, participant: Participant, timestamp: datetime, rng: Random) -> Record:
        duration = rng.lognormvariate(self.duration_lognormal_mu, self.duration_lognormal_sigma)
        duration = min(duration, self.duration_cap_seconds)

        return {
            "timestamp": timestamp.isoformat(),
            "participant_id": participant.participant_id,
            "event_type": rng.choice(EVENT_TYPES),
            "duration_seconds": round(duration, 1),
        }
