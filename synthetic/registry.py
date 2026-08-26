"""The stream-name -> generator registry.

All 8 modalities are registered as of Step 3. Two scope notes, carried
from the individual generator modules:

  - `proximity.participant_beacon` is the only proximity variant
    implemented. beacon-to-beacon and device-to-beacon are NOT
    participant-scoped and don't fit this registry's per-participant
    `generate(participant, day, rng)` signature — see
    `synthetic.generators.proximity` for the full explanation. They
    need a separate, non-participant-scoped generation entry point,
    not yet built.
  - `environment.device` assumes one environment device per participant
    (a documented simplification) rather than the spec's literal
    per-device granularity — see `synthetic.generators.environment`.
"""

from __future__ import annotations

from synthetic.generators.audio_features import AudioFeatureGenerator
from synthetic.generators.base import StreamGenerator
from synthetic.generators.chest_ecg import ChestEcgGenerator
from synthetic.generators.environment import EnvironmentGenerator
from synthetic.generators.fitbit import FitbitHRGenerator
from synthetic.generators.fitbit_sleep import FitbitSleepGenerator
from synthetic.generators.fitbit_steps import FitbitStepsGenerator
from synthetic.generators.phone_events import PhoneEventGenerator
from synthetic.generators.proximity import ProximityGenerator
from synthetic.generators.surveys import EmaSurveyGenerator

REGISTRY: dict[str, StreamGenerator] = {
    "fitbit.heart_rate": FitbitHRGenerator(),
    "fitbit.steps": FitbitStepsGenerator(),
    "fitbit.sleep": FitbitSleepGenerator(),
    "chest_ecg.snippet": ChestEcgGenerator(),
    "audio_features.summary": AudioFeatureGenerator(),
    "proximity.participant_beacon": ProximityGenerator(),
    "environment.device": EnvironmentGenerator(),
    "phone_events.interaction": PhoneEventGenerator(),
    "surveys.ema": EmaSurveyGenerator(),
}
