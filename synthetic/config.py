"""Configuration and participant-roster scaffolding for the synthetic
Chronis test-data generator.

This module deliberately contains NO stream-specific generation logic.
Its only job is:
  1. Define the generator's configuration surface (`GenConfig`).
  2. Build a deterministic, seeded participant roster with per-participant
     "active day" calendars.

Stream generators (fitbit.heart_rate, fitbit.steps, etc.) are built on
top of this in later steps and must not duplicate roster/day logic.

GROUND TRUTH: every value produced here is entirely invented by this
project for use as a test fixture. Nothing in this module is derived
from, or should ever be presented as, real TILES-2018 or any other
third-party dataset's actual content or schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from random import Random


@dataclass(frozen=True, slots=True)
class CorruptionConfig:
    """Per-corruption-mode injection probabilities.

    Each value is the probability (0.0-1.0) that a given corruption mode
    is applied to a given record/file during generation. Defaults are
    the spec's "normal run" rates (low, non-zero); the CI "stress config"
    overrides these with elevated rates (Section 6 / Section 4.1 of the
    spec) via `CorruptionConfig.stress_config()`.
    """

    missing_block: float = 0.03
    duplicate_rows: float = 0.02
    out_of_order: float = 0.02
    clock_drift: float = 0.02
    malformed_row: float = 0.02
    participant_dropout: float = 0.03
    schema_version_bump: float = 0.01
    # leaky_fixture is NOT a probability - it is a fixed, always-on
    # regression case (see Section 6/8). It is injected deterministically,
    # not sampled, and is controlled separately by GenConfig.include_leaky_fixture.

    @classmethod
    def stress_config(cls) -> CorruptionConfig:
        """Elevated corruption rates for the scheduled CI stress run."""
        return cls(
            missing_block=0.30,
            duplicate_rows=0.25,
            out_of_order=0.25,
            clock_drift=0.30,
            malformed_row=0.25,
            participant_dropout=0.35,
            schema_version_bump=0.10,
        )

    def __post_init__(self) -> None:
        for field_name in (
            "missing_block",
            "duplicate_rows",
            "out_of_order",
            "clock_drift",
            "malformed_row",
            "participant_dropout",
            "schema_version_bump",
        ):
            value = getattr(self, field_name)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{field_name}={value!r} must be within [0.0, 1.0]")


@dataclass(frozen=True, slots=True)
class GenConfig:
    """Top-level, fully-serializable configuration for one generation run.

    Deliberately config-driven per the spec's Section 4.1 requirement:
    number of participants, study length, and corruption rates are all
    parameters here, never hard-coded in generator logic.
    """

    seed: int
    participant_count: int
    study_start_date: date
    study_length_days: int
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    include_leaky_fixture: bool = True
    generator_version: str = "0.1.0"

    def __post_init__(self) -> None:
        if self.participant_count < 1:
            raise ValueError("participant_count must be at least 1")
        if self.study_length_days < 1:
            raise ValueError("study_length_days must be at least 1")

    def to_dict(self) -> dict[str, object]:
        """Serializable form for `_generator_manifest.json`."""
        return {
            "seed": self.seed,
            "participant_count": self.participant_count,
            "study_start_date": self.study_start_date.isoformat(),
            "study_length_days": self.study_length_days,
            "corruption": {
                "missing_block": self.corruption.missing_block,
                "duplicate_rows": self.corruption.duplicate_rows,
                "out_of_order": self.corruption.out_of_order,
                "clock_drift": self.corruption.clock_drift,
                "malformed_row": self.corruption.malformed_row,
                "participant_dropout": self.corruption.participant_dropout,
                "schema_version_bump": self.corruption.schema_version_bump,
            },
            "include_leaky_fixture": self.include_leaky_fixture,
            "generator_version": self.generator_version,
        }


@dataclass(frozen=True, slots=True)
class Participant:
    """One synthetic study participant and their active-day calendar."""

    participant_id: str
    enrollment_date: date
    active_days: tuple[date, ...]
    """Days this participant has data for. May be a strict subset of the
    full study window if `participant_dropout` corruption removed some
    trailing days for this participant (Section 6)."""

    resting_heart_rate: float
    """Per-participant baseline resting HR (bpm), drawn once at roster
    build time so all of this participant's fitbit.heart_rate records
    are internally consistent with a single physiological baseline
    (spec Section 5.1: "draw a per-participant resting HR")."""


def _build_active_days(
    *,
    study_start_date: date,
    study_length_days: int,
    dropout_probability: float,
    rng: Random,
) -> tuple[date, ...]:
    """Build one participant's active-day calendar.

    With probability `dropout_probability`, this participant "drops out"
    partway through the study: their active days are truncated at a
    randomly chosen point rather than covering the full study window.
    This directly implements the `participant_dropout` corruption mode
    (Section 6): downstream code must never assume every participant has
    every study day present.
    """

    all_days = tuple(
        study_start_date + timedelta(days=offset) for offset in range(study_length_days)
    )

    if rng.random() < dropout_probability and study_length_days > 1:
        # Truncate at a random point strictly before the end, so at
        # least one day is always present.
        cutoff = rng.randint(1, study_length_days - 1)
        return all_days[:cutoff]

    return all_days


def build_roster(config: GenConfig) -> tuple[Participant, ...]:
    """Deterministically build the full participant roster for a run.

    Given the same `config.seed` and other config values, this must
    always return byte-identical results (spec Section 4.1:
    "deterministic given a seed").
    """

    rng = Random(config.seed)

    participants = []

    for index in range(config.participant_count):
        participant_id = f"synthetic_p{index + 1:04d}"

        active_days = _build_active_days(
            study_start_date=config.study_start_date,
            study_length_days=config.study_length_days,
            dropout_probability=config.corruption.participant_dropout,
            rng=rng,
        )

        # Per Section 5.1: Normal(mean=68, sd=8), clipped to [50, 95].
        resting_hr = min(95.0, max(50.0, rng.gauss(68.0, 8.0)))

        participants.append(
            Participant(
                participant_id=participant_id,
                enrollment_date=config.study_start_date,
                active_days=active_days,
                resting_heart_rate=resting_hr,
            )
        )

    return tuple(participants)
