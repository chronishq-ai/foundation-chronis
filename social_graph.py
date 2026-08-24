"""
Sprint 11 - Social Graph.

Cross-session vocal-fingerprint clustering.

The graph is explicitly per-user and opaque:
fingerprints from one user can never enter another user's graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Sequence


class SocialGraphError(ValueError):
    """Invalid Social Graph input."""


@dataclass(frozen=True)
class VocalFingerprint:
    user_id: str
    session_id: str
    values: Sequence[float]


@dataclass(frozen=True)
class SocialNode:
    node_id: str
    user_id: str
    session_ids: tuple[str, ...]


@dataclass(frozen=True)
class SocialGraphResult:
    user_id: str
    nodes: tuple[SocialNode, ...]


class SocialGraph:
    """
    Deterministic per-user fingerprint clustering.

    A node represents a recurring vocal-fingerprint cluster.
    No identity/name is inferred.
    """

    def __init__(self, similarity_threshold: float = 0.8):
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError(
                "similarity_threshold must be in (0, 1]"
            )

        self.similarity_threshold = similarity_threshold

    def build(
        self,
        user_id: str,
        fingerprints: Sequence[VocalFingerprint],
    ) -> SocialGraphResult:

        if not user_id:
            raise SocialGraphError(
                "user_id must not be empty"
            )

        user_records = [
            record
            for record in fingerprints
            if record.user_id == user_id
        ]

        for record in fingerprints:
            self._validate(record)

        clusters: list[list[VocalFingerprint]] = []

        for record in user_records:
            placed = False

            for cluster in clusters:
                representative = cluster[0]

                similarity = self.cosine_similarity(
                    representative.values,
                    record.values,
                )

                if similarity >= self.similarity_threshold:
                    cluster.append(record)
                    placed = True
                    break

            if not placed:
                clusters.append([record])

        nodes = tuple(
            SocialNode(
                node_id=f"{user_id}:voice:{index}",
                user_id=user_id,
                session_ids=tuple(
                    record.session_id
                    for record in cluster
                ),
            )
            for index, cluster in enumerate(clusters)
        )

        return SocialGraphResult(
            user_id=user_id,
            nodes=nodes,
        )

    @staticmethod
    def cosine_similarity(
        a: Sequence[float],
        b: Sequence[float],
    ) -> float:

        if len(a) != len(b):
            raise SocialGraphError(
                "fingerprint dimensions must match"
            )

        dot = sum(x * y for x, y in zip(a, b))

        norm_a = sqrt(sum(x * x for x in a))
        norm_b = sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            raise SocialGraphError(
                "zero-vector fingerprint is invalid"
            )

        return dot / (norm_a * norm_b)

    @staticmethod
    def _validate(record: VocalFingerprint) -> None:
        if not record.user_id:
            raise SocialGraphError(
                "fingerprint user_id must not be empty"
            )

        if not record.session_id:
            raise SocialGraphError(
                "session_id must not be empty"
            )

        if not record.values:
            raise SocialGraphError(
                "fingerprint must not be empty"
            )

        for value in record.values:
            if not isinstance(value, (int, float)):
                raise SocialGraphError(
                    "fingerprint values must be numeric"
                )

            if not isfinite(value):
                raise SocialGraphError(
                    "fingerprint values must be finite"
                )