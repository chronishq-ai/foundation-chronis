"""
echo_detection.py

Sprint 11, Day 31 -- Echo Detection.

What this does, in plain words:
Looks through a user's history of behavioral snapshots (m_t) and finds
pairs of moments that look like "this happened before." Two moments count
as an echo only if BOTH of these are true:
  1. Their m_t vectors are similar (cosine similarity > 0.8)
  2. They were in the same behavioral regime (p_t.regime_label matches)

Why both conditions, not just one: math similarity alone is cheap and can
produce false positives (two unrelated days that happen to look similar in
10 numbers). Requiring the same regime too is the "matching social/behavioral
context" check the Bible directive calls for -- it's a second, independent
signal that has to agree before we call it a real echo.

NOTE on social context: the *real* spec also wants social context (who was
present, what kind of setting) as part of "matching context." That data has
not arrived from FOUNDRY yet (see upstream_interfaces.py TODO). Regime match
is used as a stand-in / partial context signal for now. This is called out
explicitly in the README as an open item, not silently treated as done.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence

from upstream_interfaces import BehavioralStateRecord

# Logged threshold, not a silent magic number: 0.8 is the Bible/directive's
# own stated cosine-similarity cutoff for Echo Detection.
ECHO_SIMILARITY_THRESHOLD = 0.8


@dataclass(frozen=True)
class EchoMatch:
    record_index_a: int
    record_index_b: int
    similarity_score: float
    regime_label: int


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Plain-loop cosine similarity. Returns a value from -1.0 to 1.0,
    where 1.0 means the two vectors point in exactly the same direction.

    If either vector is all zeros, similarity is undefined -- we return
    0.0 rather than dividing by zero or crashing.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("cosine_similarity: vectors must be the same length")

    dot_product = 0.0
    for i in range(len(vector_a)):
        dot_product += vector_a[i] * vector_b[i]

    magnitude_a = 0.0
    for value in vector_a:
        magnitude_a += value * value
    magnitude_a = magnitude_a ** 0.5

    magnitude_b = 0.0
    for value in vector_b:
        magnitude_b += value * value
    magnitude_b = magnitude_b ** 0.5

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def find_echoes(records: List[BehavioralStateRecord]) -> List[EchoMatch]:
    """Compare every pair of records for one user and return the ones that
    qualify as echoes. Plain nested loop -- Sprint 11's own data volumes
    (dozens to hundreds of sessions per user) make this fast enough; no
    need for a fancier nearest-neighbor structure at this scale.
    """
    echoes: List[EchoMatch] = []

    if len(records) < 2:
        return echoes

    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            record_a = records[i]
            record_b = records[j]

            # Condition 2 first (cheap check): same regime required.
            if record_a.p_t.regime_label != record_b.p_t.regime_label:
                continue

            # Condition 1: m_t similarity.
            similarity = cosine_similarity(record_a.m_t, record_b.m_t)

            if similarity > ECHO_SIMILARITY_THRESHOLD:
                echoes.append(EchoMatch(
                    record_index_a=i,
                    record_index_b=j,
                    similarity_score=similarity,
                    regime_label=record_a.p_t.regime_label,
                ))

    return echoes