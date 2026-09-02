from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from uuid import uuid4

from upstream_interfaces import AttractorRecord


class ClaimLevel(IntEnum):
    LEVEL_0 = 0
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3


@dataclass(frozen=True)
class GateEvaluation:
    level: ClaimLevel
    admissible: bool
    reason: str = ""


@dataclass(frozen=True)
class Claim:
    claim_id: str
    user_id: str
    domain_id: str
    level: ClaimLevel

    @classmethod
    def new(cls, user_id: str, domain_id: str, level: ClaimLevel, gate_eval: GateEvaluation) -> Claim:
        return cls(
            claim_id=f"claim-{user_id}-{domain_id}-{uuid4().hex[:8]}",
            user_id=user_id,
            domain_id=domain_id,
            level=level if gate_eval.admissible else ClaimLevel.LEVEL_0,
        )


def evaluate_level0(_attractor: AttractorRecord | None = None) -> GateEvaluation:
    return GateEvaluation(ClaimLevel.LEVEL_0, True, "level 0 always admissible")


def evaluate_level1(attractor: AttractorRecord) -> GateEvaluation:
    ok = bool(attractor.declared)
    return GateEvaluation(
        ClaimLevel.LEVEL_1,
        ok,
        "attractor declared" if ok else "attractor not declared",
    )
