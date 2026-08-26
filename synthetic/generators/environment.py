"""environment.device stream generator — spec Section 5.6.

SCOPE NOTE: the spec's granularity table says this stream is "per
device," not per participant. This generator assumes one environment
device per participant (their own badge/sensor) so it fits the
per-participant-per-day `generate()` signature — a documented
simplification, not a spec requirement. A future multi-device-per-
participant or shared-device model would need a different, non-1:1
device_id scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from random import Random

from synthetic.config import Participant
from synthetic.generators.base import Record
from synthetic.timeutils import irregular_timestamps


@dataclass(frozen=True, slots=True)
class EnvironmentGenerator:
    """Produces per-device ambient-environment readings across a day.

    Per spec Section 5.6:
      - Light: 0-50 lux at night, 200-900 lux during simulated day
        hours, occasional 0 (device covered/off).
      - Motion (3-axis): resting near 0 with small noise; short bursts
        of larger magnitude to simulate nearby movement.
      - Temperature: Normal(mean=22C, sd=1.5).
      - Humidity: Normal(mean=45%, sd=8), clipped [15, 80].
    """

    min_gap_seconds: float = 60.0
    max_gap_seconds: float = 300.0

    day_start_hour: int = 7
    day_end_hour: int = 19

    device_off_probability: float = 0.02

    motion_burst_probability: float = 0.05
    motion_burst_magnitude_sd: float = 0.8
    motion_resting_sd: float = 0.05

    temperature_mean_c: float = 22.0
    temperature_sd_c: float = 1.5

    humidity_mean_pct: float = 45.0
    humidity_sd_pct: float = 8.0
    humidity_bounds_pct: tuple[float, float] = (15.0, 80.0)

    def generate(self, participant: Participant, day: date, rng: Random) -> list[Record]:
        device_id = f"{participant.participant_id}_env_device"

        timestamps = irregular_timestamps(
            day,
            min_gap_seconds=self.min_gap_seconds,
            max_gap_seconds=self.max_gap_seconds,
            rng=rng,
        )

        records: list[Record] = []

        for timestamp in timestamps:
            is_daytime = self.day_start_hour <= timestamp.hour < self.day_end_hour

            if rng.random() < self.device_off_probability:
                light_lux = 0.0
            elif is_daytime:
                light_lux = rng.uniform(200.0, 900.0)
            else:
                light_lux = rng.uniform(0.0, 50.0)

            if rng.random() < self.motion_burst_probability:
                motion = tuple(rng.gauss(0.0, self.motion_burst_magnitude_sd) for _ in range(3))
            else:
                motion = tuple(rng.gauss(0.0, self.motion_resting_sd) for _ in range(3))

            temperature = rng.gauss(self.temperature_mean_c, self.temperature_sd_c)

            humidity = min(
                self.humidity_bounds_pct[1],
                max(
                    self.humidity_bounds_pct[0],
                    rng.gauss(self.humidity_mean_pct, self.humidity_sd_pct),
                ),
            )

            records.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "device_id": device_id,
                    "light_lux": round(light_lux, 1),
                    "motion_x": round(motion[0], 3),
                    "motion_y": round(motion[1], 3),
                    "motion_z": round(motion[2], 3),
                    "temperature_c": round(temperature, 2),
                    "humidity_pct": round(humidity, 1),
                }
            )

        return records
