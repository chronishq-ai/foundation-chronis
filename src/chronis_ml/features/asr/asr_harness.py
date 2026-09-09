"""S2.3 — Integration-test harness (intern-owned deliverable).

This is the actual intern deliverable for S2.3: a reusable test harness
that can validate ANY real wearer-beam-only routing implementation
against the required privacy invariant, once the Speech ML lead builds
one. It does not implement or decide the routing itself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from chronis_ml.features.asr.types import ASRBackend, DiarizedSegment, SpeakerRole


class RecordingASRBackend:
    """An instrumented ASR mock recording every audio_reference it was
    ever called with — the harness inspects this AFTER routing runs to
    prove bystander audio never reached the call at all, not merely
    that its transcript was discarded afterward."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def transcribe(self, audio_reference: str) -> str:
        self.calls.append(audio_reference)
        return f"transcript_for::{audio_reference}"


class PrivacyInvariantViolation(AssertionError):
    """Raised when a routing implementation calls the ASR backend with
    a bystander segment's audio."""


RoutingFunction = Callable[[Sequence[DiarizedSegment], ASRBackend], dict[str, str]]


def assert_bystander_never_reaches_asr(
    route_and_transcribe: RoutingFunction,
    segments: Sequence[DiarizedSegment],
) -> RecordingASRBackend:
    """Run a routing implementation against a diarized fixture and
    assert the S2.3 privacy invariant: the ASR backend must NEVER be
    invoked with a bystander segment's audio_reference.

    This is the reusable harness the ticket calls for — pass it any
    real (senior-designed) routing implementation, or the reference/
    negative-control implementations in this test suite, and it proves
    the same invariant against all of them.
    """

    asr = RecordingASRBackend()
    route_and_transcribe(segments, asr)

    bystander_refs = {
        segment.audio_reference
        for segment in segments
        if segment.speaker_role is SpeakerRole.BYSTANDER
    }
    called_refs = set(asr.calls)

    violating_refs = bystander_refs & called_refs
    if violating_refs:
        raise PrivacyInvariantViolation(
            f"ASR backend was called with {len(violating_refs)} bystander segment(s): "
            f"{sorted(violating_refs)} — bystander audio must never reach the ASR call at all"
        )

    return asr


def assert_wearer_segments_reach_asr(
    route_and_transcribe: RoutingFunction,
    segments: Sequence[DiarizedSegment],
) -> dict[str, str]:
    """Confirms the plumbing actually delivers wearer segments to the
    ASR backend and returns real transcript output — proving the
    pipeline works end-to-end for the allowed case, not just that it
    correctly withholds the disallowed case."""

    asr = RecordingASRBackend()
    results = route_and_transcribe(segments, asr)

    wearer_refs = {s.audio_reference for s in segments if s.speaker_role is SpeakerRole.WEARER}

    assert wearer_refs <= set(asr.calls), "not every wearer segment reached the ASR backend"
    assert wearer_refs <= set(results.keys()), "not every wearer segment has a transcript result"

    return results
