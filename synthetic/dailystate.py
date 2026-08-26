"""A small deterministic helper letting independently-registered stream
generators loosely correlate on a per-participant-per-day basis, without
directly calling into each other (which would break the registry's
"each modality's logic is independent and testable in isolation"
principle).

Currently used by `EmaSurveyGenerator` to make self-reported stress
loosely track — but not perfectly derive from — physiological stress
signals, per spec Section 5.8: "correlates loosely... but isn't
perfectly derived from... real self-report never perfectly matches
physiology."

NOTE: `FitbitHRGenerator` does not yet consume this helper — it was
built and tested in Step 2 before this correlation requirement was
implemented. Wiring it in (so HR stress-window placement and EMA
stress scores share the same underlying latent intensity) is a
documented follow-up, not done here, to avoid changing already-shipped,
tested Step 2 behavior in this step.
"""

from __future__ import annotations

import hashlib
from datetime import date


def daily_stress_intensity(
    participant_id: str, day: date, *, salt: str = "chronis-stress-v1"
) -> float:
    """A deterministic pseudo-random value in [0.0, 1.0] for one
    participant on one day.

    Deterministic (not RNG-based) so that it can be computed
    independently by any generator without needing a shared `Random`
    instance to be threaded between them — same participant_id + day
    always yields the same intensity, from any call site.
    """

    digest = hashlib.sha256(f"{salt}:{participant_id}:{day.isoformat()}".encode()).digest()
    # Use the first 4 bytes as an unsigned integer, normalize to [0, 1).
    integer_value = int.from_bytes(digest[:4], byteorder="big")
    return integer_value / 0xFFFFFFFF
