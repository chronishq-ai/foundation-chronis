"""proximity stream generator — spec Section 5.5.

SCOPE NOTE: the spec describes three proximity variants — participant-
to-beacon, beacon-to-beacon, and device-to-beacon. Only participant-to-
beacon is naturally scoped to a single participant; the other two are
not tied to any one participant at all, so they don't fit the
per-participant-per-day `generate(participant, day, rng)` signature
every other generator in this registry uses. Implementing this
generator for the participant-beacon variant only, and leaving the
other two as an explicitly open follow-up requiring a separate,
non-participant-scoped generation entry point, rather than forcing a
bad fit here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from random import Random

from synthetic.config import Participant
from synthetic.generators.base import Record
from synthetic.timeutils import random_windows

DEFAULT_BEACON_IDS: tuple[str, ...] = tuple(f"beacon_{i:03d}" for i in range(1, 6))


@dataclass(frozen=True, slots=True)
class ProximityGenerator:
    """Produces sparse, bursty participant-to-beacon RSSI readings.

    Per spec Section 5.5: RSSI in dBm, roughly Normal(mean=-65, sd=15),
    clipped to [-95, -30]. Sparse and bursty by design — proximity
    events cluster in short windows rather than spreading evenly across
    the day.
    """

    burst_count_range: tuple[int, int] = (2, 6)
    burst_duration_minutes_range: tuple[int, int] = (2, 8)
    readings_per_burst_range: tuple[int, int] = (3, 20)

    rssi_mean: float = -65.0
    rssi_sd: float = 15.0
    rssi_bounds: tuple[float, float] = (-95.0, -30.0)

    beacon_ids: tuple[str, ...] = DEFAULT_BEACON_IDS

    def generate(self, participant: Participant, day: date, rng: Random) -> list[Record]:
        windows = random_windows(
            day,
            count_range=self.burst_count_range,
            duration_minutes_range=self.burst_duration_minutes_range,
            rng=rng,
        )

        records: list[Record] = []

        for window_start, window_end in windows:
            beacon_id = rng.choice(self.beacon_ids)
            reading_count = rng.randint(*self.readings_per_burst_range)

            window_seconds = (window_end - window_start).total_seconds()

            for _ in range(reading_count):
                offset_seconds = rng.uniform(0, window_seconds)
                timestamp = window_start + timedelta(seconds=offset_seconds)

                rssi = rng.gauss(self.rssi_mean, self.rssi_sd)
                rssi = min(self.rssi_bounds[1], max(self.rssi_bounds[0], rssi))

                records.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "source_id": participant.participant_id,
                        "source_type": "participant",
                        "target_id": beacon_id,
                        "target_type": "beacon",
                        "rssi_dbm": round(rssi, 1),
                    }
                )

        records.sort(key=lambda record: record["timestamp"])  # type: ignore[arg-type,return-value]

        return records
