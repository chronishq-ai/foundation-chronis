"""
mirror/tone_calibration.py
Sprint 12, Day 35 — Tone calibration + TTS stub.

Tone modes: DIRECT | REFLECTIVE | WARM — user-selectable, applied at
generation time (not post-hoc string rewriting).

Each mode carries a system-prompt modifier that is PREPENDED to the
constrained system prompt before the LLM call, so tone is baked in
at generation time, never layered on after.

TTS stub: returns synthesised-speech placeholder bytes.
IMPORTANT: this is EXPLICITLY a synthesised voice stub, NOT a recording
of the user. Per the spec (Day 35): "explicitly synthesized, not a
recording of the user."

Bible ref: Part 5.21 (Insight Generation / The Mirror, Module 4.10)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claims_engine.grounded_generation import LLMClient


# ---------------------------------------------------------------------------
# Tone modes
# ---------------------------------------------------------------------------

class ToneMode(Enum):
    """
    Three user-selectable tone modes for The Mirror.

    Applied at generation time via system-prompt modifier — not post-hoc.
    Default is REFLECTIVE (balanced; non-directive).
    """
    DIRECT     = "direct"      # clear, specific, no hedging — still evidence-grounded
    REFLECTIVE = "reflective"  # tentative, open-ended, invites self-inquiry (default)
    WARM       = "warm"        # supportive register, empathetic framing, same evidence bar


# System-prompt modifiers — prepended to the constrained base prompt.
# Each modifier ADDS a framing instruction without removing any hard rules
# (clinical filter, citation requirement, 3-sentence cap, etc).
_TONE_MODIFIERS: dict[ToneMode, str] = {
    ToneMode.DIRECT: (
        "Tone: Direct and specific. State observations plainly and precisely. "
        "Do not soften or over-qualify — let the data speak. "
        "Still tentative (this is a pattern, not a verdict), but not hedged unnecessarily."
    ),
    ToneMode.REFLECTIVE: (
        "Tone: Reflective and open-ended. Frame each observation as something worth "
        "sitting with, not a conclusion. Use language that invites self-inquiry "
        "(e.g. 'It seems like…', 'You might notice…'). Never directive."
    ),
    ToneMode.WARM: (
        "Tone: Warm and supportive. Acknowledge the effort visible in the data before "
        "naming the pattern. Frame observations with care — not softening the truth, "
        "but delivering it with evident regard for the person. "
        "Avoid clinical distance; write as a thoughtful, evidence-bound friend."
    ),
}


def tone_system_prompt(base_prompt: str, tone: ToneMode) -> str:
    """
    Build the full system prompt by prepending the tone modifier to the base
    constrained prompt from Sprint 9.

    Args:
        base_prompt: The CONSTRAINED_SYSTEM_PROMPT from grounded_generation.py.
                     All its hard rules (no diagnosis, citation required, etc.)
                     remain in force regardless of tone.
        tone:        User-selected tone mode.

    Returns:
        Full system prompt string to pass to LLMClient.generate().
    """
    modifier = _TONE_MODIFIERS[tone]
    return f"{modifier}\n\n{base_prompt}"


# ---------------------------------------------------------------------------
# Tone-aware generation entry point
# ---------------------------------------------------------------------------

def generate_with_tone(
    system_prompt_base: str,
    user_content: str,
    tone: ToneMode,
    llm_client: "LLMClient",
) -> str:
    """
    Generate an insight with the given tone applied at generation time.

    The tone modifier is PREPENDED to the base system prompt so it affects
    the generation pass itself, not a post-hoc rewriting step.

    Args:
        system_prompt_base: CONSTRAINED_SYSTEM_PROMPT from Sprint 9.
        user_content:       Assembled excerpt + divergence context block.
        tone:               User's selected ToneMode.
        llm_client:         Self-hosted LLMClient (Protocol from Sprint 9).
                            Never a third-party API call.

    Returns:
        Raw generated text string (caller must still run through linter).
    """
    full_prompt = tone_system_prompt(system_prompt_base, tone)
    return llm_client.generate(full_prompt, user_content)


# ---------------------------------------------------------------------------
# Voice TTS stub
# ---------------------------------------------------------------------------

# Mandatory notice embedded in the stub output so any downstream consumer
# is unambiguous about what this is.
_TTS_STUB_NOTICE = (
    "[SYNTHESISED VOICE STUB — Sprint 12 Day 35. "
    "This is computer-generated speech, not a recording of the user. "
    "Wire a real prosody-calibrated TTS engine before production use.]"
)


@dataclass(frozen=True)
class TTSStubResult:
    """
    Placeholder result from the TTS stub.

    audio_bytes: UTF-8 encoded notice string (not real audio).
    is_stub:     Always True — callers must check this before playback.
    tone:        The tone mode the audio was requested for.
    text_length: Number of characters in the source text.
    notice:      Human-readable explanation that this is a stub.
    """
    audio_bytes: bytes
    is_stub: bool
    tone: ToneMode
    text_length: int
    notice: str


def synthesize_voice_stub(text: str, tone: ToneMode) -> TTSStubResult:
    """
    Prosody-calibrated TTS stub.

    Returns a TTSStubResult whose audio_bytes is a UTF-8 encoded notice
    string — NOT real audio. Callers MUST check is_stub == True before
    attempting playback.

    IMPORTANT: Per Sprint 12 Day 35 spec — this voice is explicitly
    SYNTHESISED, never a recording of the user. The production
    implementation must uphold the same constraint.

    Args:
        text: The insight text to synthesise.
        tone: ToneMode used for prosody selection (logged in result).

    Returns:
        TTSStubResult with is_stub=True.
    """
    stub_payload = (
        f"{_TTS_STUB_NOTICE}\n"
        f"tone={tone.value}  chars={len(text)}\n"
        f"text_preview={text[:80]}{'...' if len(text) > 80 else ''}"
    )
    return TTSStubResult(
        audio_bytes=stub_payload.encode("utf-8"),
        is_stub=True,
        tone=tone,
        text_length=len(text),
        notice=_TTS_STUB_NOTICE,
    )
