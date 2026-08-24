"""Sprint 11 Day 31 -- Echo Detection."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import List, Sequence

from upstream_interfaces import BehavioralStateRecord, SocialContext

ECHO_SIMILARITY_THRESHOLD = 0.8


@dataclass(frozen=True)
class EchoMatch:
    record_index_a: int
    record_index_b: int
    similarity_score: float
    regime_label: int
    echo_type: str
    social_context_key: str


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError("cosine_similarity: vectors must be the same length")
    if not vector_a:
        raise ValueError("cosine_similarity: vectors must not be empty")
    if any(not isinstance(v, (int, float)) or not isfinite(v) for v in vector_a + vector_b):
        raise ValueError("cosine_similarity: vectors must contain finite numeric values")
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    na = sqrt(sum(a * a for a in vector_a))
    nb = sqrt(sum(b * b for b in vector_b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _validate_context(context: SocialContext | None) -> None:
    if context is None:
        return
    if context.context_type not in {"conversation", "behavioral-loop", "situational"}:
        raise ValueError("unsupported social context type")
    if not context.context_key:
        raise ValueError("social context key must not be empty")
    if any(ch in context.context_key for ch in "\n\r"):
        raise ValueError("social context key must be opaque and single-line")


def _classify_context(a: SocialContext, b: SocialContext) -> str | None:
    if a.context_key != b.context_key:
        return None
    # Type agreement is required: same opaque context and same declared kind.
    if a.context_type != b.context_type:
        return None
    return a.context_type


def find_echoes(records: List[BehavioralStateRecord]) -> List[EchoMatch]:
    if len(records) < 2:
        return []
    user_ids = {r.user_id for r in records}
    if len(user_ids) != 1:
        raise ValueError("Echo Detection requires one user at a time")
    for record in records:
        if not record.user_id:
            raise ValueError("user_id must not be empty")
        _validate_context(record.social_context)
        if not record.m_t:
            raise ValueError("m_t must not be empty")

    echoes: List[EchoMatch] = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            if a.p_t.regime_label != b.p_t.regime_label:
                continue
            if a.social_context is None or b.social_context is None:
                continue
            echo_type = _classify_context(a.social_context, b.social_context)
            if echo_type is None:
                continue
            similarity = cosine_similarity(a.m_t, b.m_t)
            if similarity > ECHO_SIMILARITY_THRESHOLD:
                echoes.append(EchoMatch(i, j, similarity, a.p_t.regime_label, echo_type, a.social_context.context_key))
    return echoes
