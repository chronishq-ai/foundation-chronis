# Queryable index of claims that actually reached the user.
# Canonical-record-linked: each row names claim_id, domain, divergence type,
# level, and surfacing timestamp. Append-only. This is not the policy-engine
# audit log — that records every CLAIM_ACCESS attempt. This index records
# only SURFACE outcomes (what the user saw).
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterator, Optional

INFLUENCE_WINDOW_DAYS = 30
INFLUENCE_FLAG = "potentially_claim_influenced"
MIN_SURFACED_LEVEL = 1


@dataclass(frozen=True)
class SurfacedClaim:
    claim_id: str
    user_id: str
    domain: str
    level: int
    div_type: str
    when: date


class SurfacingIndex:
    """Append-only, queryable surfacing index (Day 43)."""

    def __init__(self) -> None:
        self._rows: list[SurfacedClaim] = []

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[SurfacedClaim]:
        return iter(self._rows)

    def append(self, rec: SurfacedClaim) -> Optional[SurfacedClaim]:
        if rec.level < MIN_SURFACED_LEVEL:
            return None
        self._rows.append(rec)
        return rec

    def for_user(self, user_id: str) -> list[SurfacedClaim]:
        return [r for r in self._rows if r.user_id == user_id]

    def for_domain(self, user_id: str, domain: str) -> list[SurfacedClaim]:
        return [r for r in self._rows if r.user_id == user_id and r.domain == domain]

    def in_window(self, user_id: str, domain: str, when: date) -> list[SurfacedClaim]:
        lo = when - timedelta(days=INFLUENCE_WINDOW_DAYS)
        return [
            r
            for r in self.for_domain(user_id, domain)
            if lo <= r.when <= when
        ]

    def would_flag(self, user_id: str, domain: str, when: date) -> bool:
        """True if a Level 1–3 claim for this domain was shown in the prior 30 days,
        inclusive of the surfacing day and day + 30."""
        return any(
            timedelta(0) <= (when - r.when) <= timedelta(days=INFLUENCE_WINDOW_DAYS)
            for r in self.for_domain(user_id, domain)
        )
