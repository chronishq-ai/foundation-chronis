"""
Pod B v0.2 - Evaluation Protocol.

This file is the pass/fail contract, agreed with Pod A (Vagu) and the
Program Lead BEFORE any scoring happens. Per spec section 1: do not choose
metrics after seeing results. If this file changes after evaluate.py has
been run once, that's a violation of the protocol - re-run from scratch.

Locked: Day 1, v0.2 sprint (2026-07-28).
"""
from __future__ import annotations

METRIC = "mae"
# Mean Absolute Error between the model's stated confidence (0-1) and a
# human's blind retrospective rating of "how much should this event have
# moved the variable" (also normalized to 0-1). Chosen over Brier score
# because ground truth here isn't a binary correct/incorrect outcome -
# it's a continuous human judgment of magnitude, which MAE compares
# directly without forcing a binary reduction.

PASS_CONDITION = "llm_mae < baseline_mae"
# The LLM's calibration is only considered validated if it beats a naive
# fixed-confidence baseline on the same sentences, scored the same way.
# Matching the baseline (or losing to it) means confidence is not
# meaningfully calibrated, regardless of how good it looks on individual
# examples - the whole point of v0.2 is not trusting a "looks reasonable"
# read.

JUSTIFICATION = """
Why MAE, not a classification metric:
Pod B's confidence is not a prediction of "will this happen" (which would
suit Brier score / accuracy). It's a magnitude judgment ("how much should
this move the variable"), and ground truth is collected as a continuous
human rating for the same reason. MAE penalizes distance from the human
judgment directly and is interpretable in the same 0-1 units as the
confidence score itself, so a result of "LLM MAE = 0.12" is directly
readable without extra transformation.

Why beat-the-baseline, not an absolute threshold (e.g. "MAE < 0.2"):
An absolute threshold assumes we know in advance how hard the task is.
A naive baseline gives a moving, honest reference point: if the LLM can't
beat "always guess 0.5 confidence" (or a keyword heuristic), whatever its
raw MAE number is, it is not doing better than not thinking about it at
all. This keeps the bar meaningful even if human raters turn out to be
more or less lenient than expected.

Locked before evaluate.py is run against real ground truth. Any change to
METRIC, PASS_CONDITION, or the baseline definition after seeing results
invalidates the evaluation and requires a fresh run.
"""


def describe() -> str:
    """Human-readable one-shot summary, used in report.py's output header."""
    return (
        f"Metric: {METRIC.upper()}\n"
        f"Pass condition: {PASS_CONDITION}\n"
        f"{JUSTIFICATION.strip()}"
    )