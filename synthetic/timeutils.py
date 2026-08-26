"""Shared time-generation utilities for synthetic stream generators."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from random import Random


def day_start(day: date) -> datetime:
    """UTC midnight at the start of the given calendar day."""
    return datetime.combine(day, time.min, tzinfo=UTC)


def irregular_timestamps_between(
    start: datetime,
    end: datetime,
    *,
    min_gap_seconds: float,
    max_gap_seconds: float,
    rng: Random,
) -> list[datetime]:
    """Generate an irregular, monotonically increasing sequence of
    timestamps between two arbitrary datetimes, with gaps drawn
    uniformly from [min_gap_seconds, max_gap_seconds].

    This is the general form used by `irregular_timestamps` (whole-day)
    and by any generator that only samples within a restricted window
    (e.g. audio_features during "at work" hours).
    """

    if min_gap_seconds <= 0:
        raise ValueError("min_gap_seconds must be positive")
    if max_gap_seconds < min_gap_seconds:
        raise ValueError("max_gap_seconds must be >= min_gap_seconds")
    if end < start:
        raise ValueError("end must be >= start")

    timestamps: list[datetime] = []
    current = start

    while current < end:
        timestamps.append(current)
        gap = rng.uniform(min_gap_seconds, max_gap_seconds)
        current = current + timedelta(seconds=gap)

    return timestamps


def irregular_timestamps(
    day: date,
    *,
    min_gap_seconds: float,
    max_gap_seconds: float,
    rng: Random,
) -> list[datetime]:
    """Generate an irregular, monotonically increasing sequence of
    timestamps spanning one full calendar day (UTC).

    "Irregular" is a deliberate spec requirement (Section 5.1: "never
    perfectly uniform") — real sensors do not sample on a perfect grid.
    """

    start = day_start(day)
    end = start + timedelta(days=1)

    return irregular_timestamps_between(
        start, end, min_gap_seconds=min_gap_seconds, max_gap_seconds=max_gap_seconds, rng=rng
    )


def random_windows(
    day: date,
    *,
    count_range: tuple[int, int],
    duration_minutes_range: tuple[int, int],
    rng: Random,
) -> tuple[tuple[datetime, datetime], ...]:
    """Pick a random number of non-overlapping-by-construction time
    windows within one calendar day, each with a random duration.

    Used to place "active" and "stress event" sub-windows (spec Section
    5.1) or activity "bursts" (Section 5.7) at random points in the day.
    Windows are generated independently and may occasionally overlap —
    that's acceptable and realistic (a stress event during an active
    period is plausible), not a bug.
    """

    count = rng.randint(*count_range)
    start = day_start(day)

    windows = []
    for _ in range(count):
        offset_seconds = rng.uniform(0, 23 * 3600)  # keep window within the day
        duration_minutes = rng.randint(*duration_minutes_range)

        window_start = start + timedelta(seconds=offset_seconds)
        window_end = window_start + timedelta(minutes=duration_minutes)

        windows.append((window_start, window_end))

    return tuple(windows)


def is_within_any_window(
    timestamp: datetime,
    windows: tuple[tuple[datetime, datetime], ...],
) -> bool:
    return any(window_start <= timestamp < window_end for window_start, window_end in windows)
