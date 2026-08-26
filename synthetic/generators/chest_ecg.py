"""chest_ecg.snippet stream generator — spec Section 5.3 (OMSignal stand-in).

Per spec Section 1.2 / S1.1 note: OMSignal is physiological/movement
data, not audio — this generator produces a synthetic ECG-shaped
waveform, never anything audio-derived.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from random import Random

from synthetic.config import Participant
from synthetic.generators.base import Record
from synthetic.timeutils import irregular_timestamps


@dataclass(frozen=True, slots=True)
class ChestEcgGenerator:
    """Produces short synthetic ECG-shaped waveform snippets.

    Per spec Section 5.3: a 15-second waveform at 250 Hz (3,750 samples)
    generated as a simple repeating pulse-shape (Gaussian-bump template)
    at the participant's current synthetic HR, plus Gaussian noise. Not
    medically accurate — only shaped like a repeating waveform with
    realistic sample count and timing. Emitted roughly every 5 minutes.

    NOTE: "current synthetic HR" here is approximated from the
    participant's resting baseline plus small random variation, since
    this generator is decoupled from `FitbitHRGenerator`'s actual
    per-timestamp output (each generator in the registry is independent
    and does not read another generator's results).
    """

    duration_seconds: float = 15.0
    sampling_rate_hz: float = 250.0
    snippet_interval_minutes: float = 5.0
    snippet_interval_jitter_minutes: float = 1.0

    pulse_width_seconds: float = 0.08
    """Width (std dev) of each simulated heartbeat's Gaussian bump."""
    noise_sd: float = 0.05
    """Amplitude of additive Gaussian noise on top of the clean pulse
    template."""

    hr_variation_sd: float = 5.0
    """Spread of the per-snippet approximate HR around the
    participant's resting baseline."""

    def generate(self, participant: Participant, day: date, rng: Random) -> list[Record]:
        snippet_starts = self._snippet_start_times(day, rng)

        sample_count = round(self.duration_seconds * self.sampling_rate_hz)

        records: list[Record] = []

        for index, start_time in enumerate(snippet_starts):
            approx_hr = max(30.0, rng.gauss(participant.resting_heart_rate, self.hr_variation_sd))

            samples = self._generate_waveform(approx_hr, sample_count, rng)

            records.append(
                {
                    "snippet_id": f"{participant.participant_id}_{day.isoformat()}_ecg_{index:04d}",
                    "participant_id": participant.participant_id,
                    "start_time": start_time.isoformat(),
                    "sampling_rate_hz": self.sampling_rate_hz,
                    "sample_count": sample_count,
                    "samples": json.dumps([round(s, 4) for s in samples]),
                    "derived_hr_bpm": round(approx_hr, 1),
                }
            )

        return records

    def _snippet_start_times(self, day: date, rng: Random) -> list[datetime]:
        base_gap = self.snippet_interval_minutes * 60
        jitter = self.snippet_interval_jitter_minutes * 60

        return irregular_timestamps(
            day,
            min_gap_seconds=max(1.0, base_gap - jitter),
            max_gap_seconds=base_gap + jitter,
            rng=rng,
        )

    def _generate_waveform(self, hr_bpm: float, sample_count: int, rng: Random) -> list[float]:
        period_seconds = 60.0 / hr_bpm
        dt = 1.0 / self.sampling_rate_hz

        samples = []
        for sample_index in range(sample_count):
            t = sample_index * dt
            phase = t % period_seconds
            # Distance from the phase to the nearest beat center (0 or period).
            distance_to_beat = min(phase, period_seconds - phase)
            pulse = math.exp(-(distance_to_beat**2) / (2 * self.pulse_width_seconds**2))
            noisy_value = pulse + rng.gauss(0.0, self.noise_sd)
            samples.append(noisy_value)

        return samples
