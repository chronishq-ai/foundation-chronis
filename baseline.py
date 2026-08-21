"""
Naive baseline for Pod B v0.2 calibration validation.

Deliberately dumb: fixed confidence, triggered only by keyword presence,
no real language understanding. This exists so evaluate.py has something
concrete for the LLM to beat - per protocol.py's PASS_CONDITION, if the
LLM's MAE against human ground truth doesn't beat this baseline's MAE,
its confidence scores aren't meaningfully calibrated.

Output shape matches EventSignals.model_dump() exactly - same
{"signals": {var: {"value": float, "confidence": float}}} structure the
LLM produces, so evaluate.py can score both identically without a branch.
"""
from __future__ import annotations

from schemas import VALID_VARIABLES

# Deliberately naive: a handful of obvious trigger words per variable.
# No synonym expansion, no negation handling, no sarcasm detection -
# that's the point. A real model should beat this without trying hard.
_KEYWORD_MAP: dict[str, list[str]] = {
    "mood": ["happy", "sad", "great", "terrible", "awful", "wonderful", "depressed", "excited"],
    "focus": ["focus", "concentrate", "distracted", "deadline", "study", "studying"],
    "stress": ["stress", "stressed", "anxious", "overwhelmed", "pressure", "argument", "fired", "fight"],
    "confidence": ["confident", "proud", "promoted", "achievement", "failed", "rejected"],
    "trust": ["trust", "lied", "betrayed", "honest", "cheated", "faithful"],
    "motivation": ["motivated", "inspired", "goal", "quit", "give up", "driven"],
    "social_engagement": ["friend", "party", "social", "alone", "isolated", "reconnected", "roommate"],
}

# Fixed values - the baseline does not attempt to judge direction or
# magnitude, only whether a variable was "triggered" at all.
_FIXED_VALUE = 5.0
_FIXED_CONFIDENCE = 0.5


def baseline_predict(event: str) -> dict:
    """
    Naive keyword-triggered prediction, same shape as analyze_event()'s
    return value: {"signals": {"<variable>": {"value": ..., "confidence": ...}}}

    A variable appears in the output iff one of its trigger words appears
    in the event text (case-insensitive substring match). Both value and
    confidence are fixed constants when triggered - this baseline makes no
    attempt to reason about magnitude or actual confidence.
    """
    text = event.lower()
    signals: dict[str, dict[str, float]] = {}

    for variable in VALID_VARIABLES:
        keywords = _KEYWORD_MAP.get(variable, [])
        if any(keyword in text for keyword in keywords):
            signals[variable] = {
                "value": _FIXED_VALUE,
                "confidence": _FIXED_CONFIDENCE,
            }

    return {"signals": signals}