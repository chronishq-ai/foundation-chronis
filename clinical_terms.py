"""
clinical_terms.py

TEMPORARY LOCAL COPY of Mansi's claims_engine/grounded_generation.py
CLINICAL_TERMS list and contains_clinical_terminology() function.

WHY THIS FILE EXISTS: Mansi's claims_engine package is not yet installed
on this repo's PYTHONPATH (it lives in a separate package/zip). Rather
than silently re-typing a *different* word list by hand (which could
quietly drift out of sync with the real one), this file copies her exact
list verbatim, with a test (see tests/test_anomaly_detection.py) that
will need manual re-checking against the real source whenever it's
available.

TODO: once claims_engine is a real importable package, DELETE this file
and replace every `from clinical_terms import ...` with
`from claims_engine.grounded_generation import contains_clinical_terminology`.
Do not let this local copy silently diverge from Mansi's real one.

Source: claims_engine/grounded_generation.py, Sprint 9 Day 27 (Mansi).
"""

from __future__ import annotations
from typing import Optional
import re

# Copied verbatim from Mansi's claims_engine/grounded_generation.py.
CLINICAL_TERMS = [
    "depression", "depressed", "anxiety", "anxious", "trauma", "traumatic",
    "disorder", "diagnosis", "diagnose", "pathology", "pathological",
]


def contains_clinical_terminology(text: str) -> Optional[str]:
    """Returns the matched clinical term if found, otherwise None.
    Identical logic to Mansi's real function -- word-boundary matching,
    case-insensitive, so it doesn't false-positive on substrings like
    'anxiousness' containing 'anxious' incorrectly, or miss capitalized text."""
    lowered = text.lower()
    for term in CLINICAL_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            return term
    return None