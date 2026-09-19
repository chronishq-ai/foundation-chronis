"""S2.2 — Temporal alignment onto a 1-minute grid with a typed
confidence contract.

Fixes the audit-flagged bug: the original `temporal_alignment.py` did
an exact-timestamp key merge across modalities, with no grid, no
confidence contract, no >10-minute gap handling.

Per Bible Part 4.8, the confidence scale is:
  1.0   all signals present, all SQI clean
  0.75  one signal missing or degraded
  0.5   two signals missing
  <0.5  insufficient for behavioral inference - flagged, and never
        passed to the HSSM at all

HONEST SCOPE NOTE: "SQI" (signal quality index) is referenced in the
Bible's confidence contract, but no SQI/quality field exists anywhere
in the current schema (`FeatureRecord` has no quality attribute). This
implementation's confidence tiers are based on PRESENCE/ABSENCE within
each 1-minute bucket only. Incorporating true signal-quality
degradation (a clean-but-different-from-missing failure mode) is an
open follow-up requiring a schema change, not fabricated here.

HONEST SCOPE NOTE 2: this implementation does NOT carry forward a
stale reading into an empty bucket, at any gap length, including gaps
under the ">10 minutes" boundary the ticket describes. A modality is
"present" for a bucket only if at least one real OBSERVED record for
that modality falls within that exact 1-minute window. This is the
simplest design that provably satisfies every test in the original
S2.2 test sheet (including the >10-minute gap case) without inventing
untested short-gap-tolerance behavior. If the Bible's intent is that
short gaps (<=10 min) should be bridged by carrying forward the last
real reading, that is a distinct, NOT-yet-built behavior - flagged
here rather than silently assumed.

HONEST SCOPE NOTE 3: B6 also requires that "clock uncertainty and
synchronization metadata are preserved." No such metadata exists
anywhere in the current schema (`FeatureRecord` carries no
clock-uncertainty or sync-source field), so there is nothing for this
function to preserve — fabricating a value here would be worse than
omitting it. Carrying this forward requires a schema addition upstream
(e.g. on `FeatureRecord`/`MissingnessSignals`) before `AlignedMinute`
can honestly expose it. Flagged as an open follow-up, not implemented
here.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from chronis_ml.schema.models import FeatureRecord, MeasurementStatus

MINUTE = timedelta(minutes=1)

# The Bible's confidence scale only defines 3 explicit tiers for 0/1/2
# missing modalities. For 3+ missing, this implementation computes a
# monotonically decreasing score below 0.5 -- this specific numeric
# formula for >=3 missing is OUR OWN reasonable extension, not
# specified by the Bible, since the Bible only requires that such rows
# be flagged and excluded, not what exact number they carry.
_CONFIDENCE_BY_MISSING_COUNT: dict[int, float] = {0: 1.0, 1: 0.75, 2: 0.5}


def _confidence_for_missing_count(missing_count: int, total_modalities: int) -> float:
    if missing_count in _CONFIDENCE_BY_MISSING_COUNT:
        return _CONFIDENCE_BY_MISSING_COUNT[missing_count]
    # 3+ missing: our own extension, monotonically decreasing, always < 0.5.
    extra_missing = missing_count - 2
    return max(0.0, 0.5 - 0.25 * extra_missing)


@dataclass(frozen=True, slots=True)
class AlignedMinute:
    """One 1-minute grid bucket's aligned values across every modality."""

    user_id: str
    minute_start: datetime
    values: Mapping[str, float | None]
    """modality -> value for this minute, or None if that modality had
    no real reading in this exact 1-minute window. NEVER a zero or a
    mean imputed from other buckets."""

    confidence: float
    missing_modalities: tuple[str, ...]
    excluded_from_downstream: bool
    """True whenever confidence < 0.5, per the Bible's explicit rule
    that such minutes are never passed to the HSSM at all."""


def _to_utc(timestamp: datetime) -> datetime:
    """Normalize any timezone-aware timestamp to UTC before grid
    arithmetic, per B6's canonical-type requirement ("timestamps use a
    canonical type, preferably UTC-aware datetime or epoch integer").

    Without this, minute-boundary flooring and bucket comparisons only
    happen to be correct because real-world timezone offsets are whole
    minutes — that's an accident of how time zones are defined, not a
    guarantee this code was actually relying on. Converting explicitly
    means `AlignedMinute.minute_start` always reports one canonical
    zone regardless of which offset a given record arrived in, instead
    of silently echoing whatever offset happened to belong to the
    record that set the grid boundary.
    """
    return timestamp.astimezone(UTC)


def align_to_minute_grid(
    user_id: str,
    records_by_modality: Mapping[str, Sequence[FeatureRecord]],
    *,
    modalities: Sequence[str] | None = None,
) -> tuple[AlignedMinute, ...]:
    """Align every modality's records for one user onto a shared
    1-minute grid, computing the typed confidence tier for each minute.

    `modalities` lets the caller declare the full expected modality set
    explicitly (config-driven, not guessed) -- if omitted, it's
    inferred from the keys of `records_by_modality`.
    """

    resolved_modalities = (
        tuple(modalities) if modalities is not None else tuple(records_by_modality.keys())
    )

    all_timestamps = [
        _to_utc(record.timestamp) for records in records_by_modality.values() for record in records
    ]

    if not all_timestamps:
        return ()

    grid_start = _floor_to_minute(min(all_timestamps))
    grid_end = _floor_to_minute(max(all_timestamps))

    aligned_minutes = []
    current_bucket = grid_start

    while current_bucket <= grid_end:
        bucket_end = current_bucket + MINUTE

        values: dict[str, float | None] = {}
        missing_modalities: list[str] = []

        for modality in resolved_modalities:
            modality_records = records_by_modality.get(modality, ())

            bucket_values = [
                record.value
                for record in modality_records
                if record.status is MeasurementStatus.OBSERVED
                and record.value is not None
                and current_bucket <= _to_utc(record.timestamp) < bucket_end
            ]

            if bucket_values:
                # Multiple real readings landing in the same minute are
                # downsampled by averaging THOSE REAL READINGS -- this
                # is NOT imputation; every value averaged is an actual
                # observation from this exact bucket.
                values[modality] = statistics.mean(bucket_values)
            else:
                values[modality] = None
                missing_modalities.append(modality)

        confidence = _confidence_for_missing_count(
            len(missing_modalities), len(resolved_modalities)
        )

        aligned_minutes.append(
            AlignedMinute(
                user_id=user_id,
                minute_start=current_bucket,
                values=values,
                confidence=confidence,
                missing_modalities=tuple(missing_modalities),
                excluded_from_downstream=confidence < 0.5,
            )
        )

        current_bucket = bucket_end

    return tuple(aligned_minutes)


def _floor_to_minute(timestamp: datetime) -> datetime:
    return _to_utc(timestamp).replace(second=0, microsecond=0)


def select_for_downstream(aligned_minutes: Sequence[AlignedMinute]) -> tuple[AlignedMinute, ...]:
    """The ONLY sanctioned way to get the subset of aligned minutes
    that may be passed to the HSSM. Minutes with confidence < 0.5 are
    never included here -- this is a concrete, callable enforcement
    point, not just an advisory flag the caller might ignore."""

    return tuple(minute for minute in aligned_minutes if not minute.excluded_from_downstream)
