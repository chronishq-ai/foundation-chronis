"""S2.3 T2/T3 — integration harness self-tests.

Test Sheet - S2.3:
  T2: Synthetic multi-speaker fixture, ASR call mocked/instrumented ->
      Run diarization + ASR pipeline -> The ASR mock is never invoked
      with bystander-segment audio at all (not merely discarded post-
      call)
  T3: Same fixture, wearer segment -> Run pipeline -> Wearer beam is
      passed to ASR and returns real transcript text once wired

IMPORTANT: `reference_route_and_transcribe` and
`broken_route_and_transcribe` below are TEST-ONLY reference
implementations, used solely to prove the harness itself works
correctly (catches violations AND passes compliant implementations).
NEITHER is the production wearer-beam-only routing design — that
decision is Speech ML lead-owned per the Ownership Model. Do not reuse
`reference_route_and_transcribe` as a real routing implementation.
"""

from __future__ import annotations

import pytest

from chronis_ml.features.asr.asr_harness import (
    PrivacyInvariantViolation,
    assert_bystander_never_reaches_asr,
    assert_wearer_segments_reach_asr,
)
from chronis_ml.features.asr.types import ASRBackend, DiarizedSegment, SpeakerRole


def reference_route_and_transcribe(
    segments: list[DiarizedSegment], asr_backend: ASRBackend
) -> dict[str, str]:
    """TEST-ONLY reference implementation. Correctly routes only
    wearer segments to the ASR backend — used here to prove the
    harness correctly PASSES a compliant implementation. Not
    production code."""

    results = {}
    for segment in segments:
        if segment.speaker_role is SpeakerRole.WEARER:
            results[segment.audio_reference] = asr_backend.transcribe(segment.audio_reference)
    return results


def broken_route_and_transcribe(
    segments: list[DiarizedSegment], asr_backend: ASRBackend
) -> dict[str, str]:
    """TEST-ONLY deliberately broken implementation — routes EVERY
    segment to the ASR backend, including bystanders. Used to prove
    the harness actually catches a real privacy violation rather than
    trivially passing anything handed to it."""

    results = {}
    for segment in segments:
        results[segment.audio_reference] = asr_backend.transcribe(segment.audio_reference)
    return results


def build_multi_speaker_fixture() -> list[DiarizedSegment]:
    return [
        DiarizedSegment(SpeakerRole.WEARER, "audio_ref_wearer_001", 0.0, 5.0),
        DiarizedSegment(SpeakerRole.BYSTANDER, "audio_ref_bystander_001", 5.0, 9.0),
        DiarizedSegment(SpeakerRole.WEARER, "audio_ref_wearer_002", 9.0, 14.0),
        DiarizedSegment(SpeakerRole.BYSTANDER, "audio_ref_bystander_002", 14.0, 18.0),
        DiarizedSegment(SpeakerRole.UNKNOWN, "audio_ref_unknown_001", 18.0, 20.0),
    ]


# --- T2: bystander audio never reaches the ASR call ---------------------------


def test_s23_t2_compliant_routing_never_calls_asr_with_bystander_audio() -> None:
    segments = build_multi_speaker_fixture()

    asr = assert_bystander_never_reaches_asr(reference_route_and_transcribe, segments)

    # Confirm the mock genuinely recorded calls (the harness isn't
    # trivially passing because nothing was called at all).
    assert asr.calls
    assert all("bystander" not in call for call in asr.calls)


def test_s23_t2_harness_catches_a_real_violation() -> None:
    """The harness must be able to FAIL — a harness that can never
    fail proves nothing. This proves it correctly detects the broken
    implementation's privacy violation."""

    segments = build_multi_speaker_fixture()

    with pytest.raises(PrivacyInvariantViolation, match="bystander segment"):
        assert_bystander_never_reaches_asr(broken_route_and_transcribe, segments)


def test_s23_t2_unknown_speaker_segments_are_not_bystander_but_also_not_asserted_safe() -> None:
    """UNKNOWN-role segments are a distinct case from BYSTANDER: the
    harness only asserts the BYSTANDER invariant explicitly. Whether
    UNKNOWN segments should reach the ASR is itself a routing-design
    question for the Speech ML lead, not decided by this harness."""

    segments = build_multi_speaker_fixture()

    # This must not raise, regardless of how the reference
    # implementation treats UNKNOWN (it currently excludes it, same as
    # bystander, but that's the reference impl's own choice, not an
    # invariant this harness enforces).
    assert_bystander_never_reaches_asr(reference_route_and_transcribe, segments)


# --- T3: wearer beam is passed to ASR and returns transcript text -----------


def test_s23_t3_wearer_segments_reach_asr_and_return_transcripts() -> None:
    segments = build_multi_speaker_fixture()

    results = assert_wearer_segments_reach_asr(reference_route_and_transcribe, segments)

    assert results["audio_ref_wearer_001"] == "transcript_for::audio_ref_wearer_001"
    assert results["audio_ref_wearer_002"] == "transcript_for::audio_ref_wearer_002"


def test_s23_t3_harness_catches_a_routing_that_drops_wearer_segments() -> None:
    """Negative control in the other direction: an implementation that
    routes NOTHING to the ASR (over-cautious to the point of breaking
    functionality) must also be caught."""

    def drops_everything(
        segments: list[DiarizedSegment], asr_backend: ASRBackend
    ) -> dict[str, str]:
        return {}

    segments = build_multi_speaker_fixture()

    with pytest.raises(AssertionError, match="not every wearer segment"):
        assert_wearer_segments_reach_asr(drops_everything, segments)
