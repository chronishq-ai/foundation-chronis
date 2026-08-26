"""The stream-generator contract shared by every modality generator.

Each concrete generator (FitbitHRGenerator, PhoneEventGenerator, etc.)
implements this Protocol independently and is registered under a stream
name in `synthetic.registry.REGISTRY`. This keeps each modality's logic
isolated and independently testable, mirroring the "modality registry,
not one giant if/elif" principle (spec Section 4.2).
"""

from __future__ import annotations

from datetime import date
from random import Random
from typing import Protocol

from synthetic.config import Participant

Record = dict[str, object]
"""One raw output row for a given stream, keyed by that stream's column
names (see the project's synthetic schema table). Deliberately a plain
dict, not a stream-specific dataclass — this lets `write_records` and
the corruption-injection step (added in a later build step) operate
generically across every stream without knowing its specific shape.
"""


class StreamGenerator(Protocol):
    def generate(self, participant: Participant, day: date, rng: Random) -> list[Record]:
        """Generate this stream's records for one participant, one day.

        Note: this takes the full `Participant` object, not just a bare
        `participant_id` string as in the spec's illustrative pseudocode
        (Section 4.2) — several generators (e.g. fitbit.heart_rate) need
        per-participant baseline attributes such as `resting_heart_rate`,
        which only exist on the `Participant` object built by
        `synthetic.config.build_roster`.
        """
        ...
