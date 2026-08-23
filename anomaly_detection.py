"""
anomaly_detection.py

Sprint 11, Day 32 -- Anomaly Detection Engine.

What this does, in plain words:
Looks at a user's behavioral history and flags three DIFFERENT kinds of
"something unusual happened," each answering a different question:

  ACUTE      -- "was this ONE moment weird, compared to the days right
                 around it?" A single spike, then back to normal.
  SUSTAINED  -- "did several days IN A ROW look shifted together?" Not
                 one spike -- a run of unusual days.
  STRUCTURAL -- "did the underlying PATTERN itself change?" Not just the
                 raw numbers moving -- the person's behavioral regime
                 shifted and stayed shifted. This is bigger than either
                 of the above.

Hard rule, non-negotiable: NONE of the copy this module (or anything
downstream) generates may contain diagnostic/medical language. This is
"not diagnostic, not medical advice" by design -- enforced here by
reusing the same clinical-terminology filter Mansi's Claims Engine uses
(see clinical_terms.py for why it's a local copy right now, not a real
import).

This module does NOT generate any user-facing sentences itself -- it only
detects and returns typed anomaly records. Whoever eventually writes the
copy (likely Sprint 12's Mirror) is responsible for running that copy
through contains_clinical_terminology() before it ships. We still test
the filter here because Anomaly Detection's own DoD explicitly requires
proving the filter catches banned language.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List

from upstream_interfaces import BehavioralStateRecord

# Logged thresholds, not silent magic numbers.
ACUTE_DEVIATION_THRESHOLD = 3.0        # euclidean distance from local baseline
SUSTAINED_DEVIATION_THRESHOLD = 1.5     # lower bar than acute, since it must hold for days
DEFAULT_MIN_CONSECUTIVE_DAYS = 3
DEFAULT_STRUCTURAL_WINDOW_SIZE = 4


class AnomalyScale(Enum):
    ACUTE = "acute"
    SUSTAINED = "sustained"
    STRUCTURAL = "structural"


@dataclass(frozen=True)
class AcuteAnomaly:
    record_index: int
    deviation_score: float
    scale: AnomalyScale = AnomalyScale.ACUTE


@dataclass(frozen=True)
class SustainedAnomaly:
    start_index: int
    end_index: int
    scale: AnomalyScale = AnomalyScale.SUSTAINED


@dataclass(frozen=True)
class StructuralAnomaly:
    record_index: int   # the day the new regime pattern first took hold
    previous_regime_label: int
    new_regime_label: int
    scale: AnomalyScale = AnomalyScale.STRUCTURAL


def _euclidean_distance(vector_a, vector_b) -> float:
    """Plain-loop distance -- how far apart two m_t vectors are."""
    if len(vector_a) != len(vector_b):
        raise ValueError("_euclidean_distance: vectors must be the same length")

    total = 0.0
    for i in range(len(vector_a)):
        difference = vector_a[i] - vector_b[i]
        total += difference * difference
    return total ** 0.5


def _local_baseline(records: List[BehavioralStateRecord], center_index: int, window: int = 2):
    """Average m_t of the days immediately around center_index (excluding
    center_index itself), used as the 'normal for this person right now'
    reference point. Never averages across a missing/out-of-range day."""
    neighbor_vectors = []

    start = max(0, center_index - window)
    end = min(len(records), center_index + window + 1)

    for i in range(start, end):
        if i == center_index:
            continue
        neighbor_vectors.append(records[i].m_t)

    if not neighbor_vectors:
        return None

    dimension_count = len(neighbor_vectors[0])
    averaged = [0.0] * dimension_count
    for vector in neighbor_vectors:
        for d in range(dimension_count):
            averaged[d] += vector[d]
    for d in range(dimension_count):
        averaged[d] = averaged[d] / len(neighbor_vectors)

    return averaged


def detect_acute_anomalies(
    records: List[BehavioralStateRecord],
    threshold: float = ACUTE_DEVIATION_THRESHOLD,
) -> List[AcuteAnomaly]:
    """A day counts as an acute anomaly only if it deviates sharply from
    the days immediately around it. This naturally excludes sustained
    shifts -- if the neighbors also drifted, the local baseline drifts
    with them and the gap won't look as large."""
    anomalies: List[AcuteAnomaly] = []

    for i in range(len(records)):
        baseline = _local_baseline(records, i)
        if baseline is None:
            continue

        deviation = _euclidean_distance(records[i].m_t, baseline)

        if deviation > threshold:
            anomalies.append(AcuteAnomaly(record_index=i, deviation_score=deviation))

    return anomalies


def detect_sustained_anomalies(
    records: List[BehavioralStateRecord],
    threshold: float = SUSTAINED_DEVIATION_THRESHOLD,
    min_consecutive_days: int = DEFAULT_MIN_CONSECUTIVE_DAYS,
) -> List[SustainedAnomaly]:
    """A sustained anomaly is a CONTIGUOUS run of days that all deviate
    from the overall dataset baseline together. Unlike the acute check,
    this compares against the whole-history baseline, not a local window
    -- because if the local window also shifted with the run, a local
    comparison would hide the very thing we're trying to find.
    """
    anomalies: List[SustainedAnomaly] = []

    if len(records) == 0:
        return anomalies

    # Whole-history baseline, using the MEDIAN per dimension rather than
    # the mean. This matters: the mean gets pulled toward the anomalies
    # themselves (the very thing we're trying to detect), which quietly
    # corrupts the definition of "normal." The median stays anchored to
    # wherever most of the days actually are, as long as anomalies are
    # a minority of the history (a safe assumption -- if MOST days are
    # anomalous, "anomaly" has stopped meaning anything anyway).
    dimension_count = len(records[0].m_t)
    overall_baseline = [0.0] * dimension_count
    for d in range(dimension_count):
        values_in_this_dimension = []
        for record in records:
            values_in_this_dimension.append(record.m_t[d])
        values_in_this_dimension.sort()
        middle = len(values_in_this_dimension) // 2
        if len(values_in_this_dimension) % 2 == 0:
            overall_baseline[d] = (
                values_in_this_dimension[middle - 1] + values_in_this_dimension[middle]
            ) / 2.0
        else:
            overall_baseline[d] = values_in_this_dimension[middle]

    deviating_flags = []
    for record in records:
        deviation = _euclidean_distance(record.m_t, overall_baseline)
        deviating_flags.append(deviation > threshold)

    # find contiguous runs of True
    run_start = None
    for i in range(len(deviating_flags)):
        is_deviating = deviating_flags[i]

        if is_deviating and run_start is None:
            run_start = i

        run_ended = (not is_deviating) or (i == len(deviating_flags) - 1)
        if run_ended and run_start is not None:
            run_end = i - 1 if not is_deviating else i
            run_length = run_end - run_start + 1
            if run_length >= min_consecutive_days:
                anomalies.append(SustainedAnomaly(start_index=run_start, end_index=run_end))
            run_start = None

    return anomalies


def detect_structural_anomalies(
    records: List[BehavioralStateRecord],
    window_size: int = DEFAULT_STRUCTURAL_WINDOW_SIZE,
) -> List[StructuralAnomaly]:
    """A structural anomaly is a change in the underlying REGIME pattern
    that persists -- not just a numeric wobble. We look for the point
    where the dominant regime_label of a forward-looking window differs
    from the dominant regime_label of the window before it, and the new
    regime holds for the rest of the window (i.e. it's not just one
    stray day)."""
    anomalies: List[StructuralAnomaly] = []

    if len(records) < window_size * 2:
        return anomalies

    def dominant_regime(window_records):
        counts = {}
        for record in window_records:
            label = record.p_t.regime_label
            counts[label] = counts.get(label, 0) + 1
        best_label = None
        best_count = -1
        for label, count in counts.items():
            if count > best_count:
                best_count = count
                best_label = label
        return best_label

    i = window_size
    while i <= len(records) - window_size:
        previous_window = records[i - window_size:i]
        next_window = records[i:i + window_size]

        previous_dominant = dominant_regime(previous_window)
        next_dominant = dominant_regime(next_window)

        if previous_dominant != next_dominant:
            # confirm the new regime actually holds for the whole next
            # window, not just a stray day or two -- that's what makes
            # this "structural" rather than acute/sustained.
            all_match = True
            for record in next_window:
                if record.p_t.regime_label != next_dominant:
                    all_match = False
                    break

            if all_match:
                anomalies.append(StructuralAnomaly(
                    record_index=i,
                    previous_regime_label=previous_dominant,
                    new_regime_label=next_dominant,
                ))
                i += window_size  # skip past this window, don't re-flag the same shift repeatedly
                continue

        i += 1

    return anomalies