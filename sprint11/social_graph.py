"""Sprint 11 -- opaque, user-internal cross-session vocal-fingerprint graph."""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Sequence

class SocialGraphError(ValueError):
    pass

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
    def __init__(self, similarity_threshold: float = 0.8):
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        self.similarity_threshold = similarity_threshold

    def build(self, user_id: str, fingerprints: Sequence[VocalFingerprint]) -> SocialGraphResult:
        if not user_id:
            raise SocialGraphError("user_id must not be empty")
        for record in fingerprints:
            self._validate(record)
        user_records = [r for r in fingerprints if r.user_id == user_id]
        session_ids = [r.session_id for r in user_records]
        if len(session_ids) != len(set(session_ids)):
            raise SocialGraphError("duplicate session_id is not allowed")
        if not user_records:
            return SocialGraphResult(user_id, tuple())
        parent = list(range(len(user_records)))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        for i in range(len(user_records)):
            for j in range(i+1, len(user_records)):
                if self.cosine_similarity(user_records[i].values, user_records[j].values) >= self.similarity_threshold:
                    union(i,j)
        groups: dict[int,list[VocalFingerprint]] = {}
        for i, record in enumerate(user_records):
            groups.setdefault(find(i), []).append(record)
        ordered_groups = sorted((sorted(group, key=lambda r: r.session_id) for group in groups.values()), key=lambda g: g[0].session_id)
        nodes = tuple(SocialNode(f"{user_id}:voice:{i}", user_id, tuple(r.session_id for r in group)) for i, group in enumerate(ordered_groups))
        return SocialGraphResult(user_id, nodes)

    @staticmethod
    def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b or len(a) != len(b):
            raise SocialGraphError("fingerprint dimensions must match and be non-empty")
        if any(not isinstance(x,(int,float)) or not isfinite(x) for x in list(a)+list(b)):
            raise SocialGraphError("fingerprint values must be finite numeric values")
        dot = sum(x*y for x,y in zip(a,b)); na=sqrt(sum(x*x for x in a)); nb=sqrt(sum(y*y for y in b))
        if na == 0 or nb == 0: raise SocialGraphError("zero-vector fingerprint is invalid")
        return dot/(na*nb)

    @staticmethod
    def _validate(record: VocalFingerprint) -> None:
        if not record.user_id: raise SocialGraphError("fingerprint user_id must not be empty")
        if not record.session_id: raise SocialGraphError("session_id must not be empty")
        if not record.values: raise SocialGraphError("fingerprint must not be empty")
        for value in record.values:
            if not isinstance(value,(int,float)) or not isfinite(value):
                raise SocialGraphError("fingerprint values must be finite numeric values")
