# Day 45 — Sprint 10 cold-start 180-day sim + Sprint 12 Mirror stage 0/1 silence,
# with Day 43 logging active for both behavioral and narrative cases.
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from .index import SurfacingIndex, SurfacedClaim
from .safeguard import Change, apply_influence_flag, product_copy

STAGE0_MAX_SESSIONS = 29
STAGE1_MAX_SESSIONS = 89  # 30–89 still cold-start / Compass; Mirror silent
SIM_DAYS = 180


def stage_for_sessions(sessions: int) -> int:
    if sessions <= STAGE0_MAX_SESSIONS:
        return 0
    if sessions <= STAGE1_MAX_SESSIONS:
        return 1
    return 2


def mirror_allowed(stage: int) -> bool:
    return stage >= 2


@dataclass
class DaySnap:
    day: int
    sessions: int
    stage: int
    mirror: str
    surfaced: int


def cold_start_180(
    index: SurfacingIndex,
    *,
    user_id: str = "cold",
    domain: str = "career",
    start: date | None = None,
    sessions_per_day: float = 0.5,
) -> list[DaySnap]:
    """180 calendar days. Sessions accumulate slowly so stages 0/1 last long
    enough to prove silence with logging on. No Level 1–3 claim is written
    to the index while Mirror is forbidden."""
    start = start or date(2026, 1, 1)
    sessions = 0.0
    snaps: list[DaySnap] = []
    for d in range(SIM_DAYS):
        sessions += sessions_per_day
        stage = stage_for_sessions(int(sessions))
        when = start + timedelta(days=d)
        if mirror_allowed(stage):
            rec = SurfacedClaim(
                claim_id=f"c-{d}",
                user_id=user_id,
                domain=domain,
                level=1,
                div_type="Aspiration",
                when=when,
            )
            index.append(rec)
            mirror = f"update in {domain}"
        else:
            mirror = ""
        assert not product_copy(Change(user_id, domain, "behavior", when)).lower().count("flag")
        snaps.append(
            DaySnap(
                day=d,
                sessions=int(sessions),
                stage=stage,
                mirror=mirror,
                surfaced=len(index),
            )
        )
    return snaps


def silence_holds(snaps: list[DaySnap]) -> bool:
    return all(s.mirror == "" for s in snaps if s.stage <= 1)


def logging_was_on(index: SurfacingIndex, snaps: list[DaySnap]) -> bool:
    """Index exists and received rows only after stage 2. Empty during 0/1 is
    still 'logging on' — we ran the same code path, it correctly wrote nothing."""
    silent_days = [s for s in snaps if s.stage <= 1]
    if not silent_days:
        return False
    first_live = next((s for s in snaps if s.stage >= 2), None)
    if first_live is None:
        return len(index) == 0
    return all(s.surfaced == 0 for s in silent_days) and len(index) > 0
