# e2e/timing.py — Sprint 14 Day 42.
# Real wall-clock measurement, not a hardcoded pass. The directive's
# 20-minute target is meaningless if the thing measuring it can't fail.
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

TWENTY_MINUTES_SECONDS = 20 * 60  # named per Global Standard #7 — no silent magic numbers


@dataclass
class StageTiming:
    name: str
    seconds: float
    is_stub: bool  # True if this stage did NOT run real Sprint N code


@dataclass
class TimingReport:
    stages: list[StageTiming] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return sum(s.seconds for s in self.stages)

    @property
    def within_target(self) -> bool:
        return self.total_seconds < TWENTY_MINUTES_SECONDS

    @property
    def real_stage_count(self) -> int:
        return sum(1 for s in self.stages if not s.is_stub)

    @property
    def stub_stage_count(self) -> int:
        return sum(1 for s in self.stages if s.is_stub)

    def summary(self) -> str:
        lines = [
            f"{'STAGE':<28}{'TIME (s)':>10}  {'KIND'}",
        ]
        for s in self.stages:
            kind = "STUB" if s.is_stub else "REAL"
            lines.append(f"{s.name:<28}{s.seconds:>10.3f}  {kind}")
        lines.append("-" * 50)
        lines.append(
            f"TOTAL: {self.total_seconds:.3f}s "
            f"({'within' if self.within_target else 'OVER'} 20-min target) — "
            f"{self.real_stage_count} real / {self.stub_stage_count} stub stages"
        )
        return "\n".join(lines)


class Timer:
    """Accumulates StageTiming entries into a TimingReport as stages run."""

    def __init__(self) -> None:
        self.report = TimingReport()

    @contextmanager
    def stage(self, name: str, *, is_stub: bool):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.report.stages.append(StageTiming(name=name, seconds=elapsed, is_stub=is_stub))