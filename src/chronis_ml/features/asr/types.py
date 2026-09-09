"""S2.3 — ASR/diarization interface types (intern-safe scope only).

Per the Ownership Model, this ticket is `Support: Integration tests,
Approval: Mandatory` — interns build the CI static check and the
integration-test harness. The actual ASR/diarization model wiring
(Whisper/pyannote integration) AND the wearer-beam-only routing design
are explicitly Speech ML lead-owned: "this is a privacy-critical
routing decision, not plumbing." Neither is built here.

What this module provides: the shared interface types a real,
senior-designed routing implementation would use, so the integration
harness (`asr_harness.py`) has something concrete to test against
without requiring the real implementation to exist yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class SpeakerRole(StrEnum):
    WEARER = "wearer"
    BYSTANDER = "bystander"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class DiarizedSegment:
    """One diarized audio segment. `audio_reference` is an opaque
    pointer to the actual audio (never raw bytes in this type) —
    consistent with the instruction not to use TILES or any other
    real audio corpus for this testing; all fixtures use synthetic
    reference strings only."""

    speaker_role: SpeakerRole
    audio_reference: str
    start_time: float
    end_time: float


class ASRBackend(Protocol):
    """The interface a real ASR backend (self-hosted Whisper, once
    wired by the Speech ML lead) must satisfy. This is an interface
    only — no implementation lives here."""

    def transcribe(self, audio_reference: str) -> str: ...
