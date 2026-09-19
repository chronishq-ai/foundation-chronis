"""S2.1 — Personal z-score normalization.

Fixes the audit-flagged bug: the original `normalizer.py` computed
`current_value - baseline[feature]` (a raw centered difference), not a
z-score. Per Bible Part 5.1: all features must be expressed as
z-scores relative to the PERSONAL mean/SD, computed per person, never
across the population.

Required semantics, each with a corresponding test:
  - True (x - mu_personal) / sigma_personal over a rolling backward-only
    window, with the window length governed by a per-feature-family
    baseline policy (see `BASELINE_POLICY_REGISTRY` below) rather than
    one fixed window for every feature.
  - Insufficient baseline history (< the family's required distinct
    days of data) must be typed as low-confidence, never silently
    returned as a raw difference pretending to be a valid z-score.
  - Near-zero variance must never divide-by-zero; must be an explicit,
    documented guard state, not a crash or a silently wrong number.
  - The baseline window is STRICTLY backward-looking: a value computed
    for timestamp T must never be influenced by any record at or after
    T, even if such records exist in the full dataset (no future-data
    leakage).
  - Every output record carries its exact baseline window and a
    baseline_version, so any normalized value can be traced back to
    exactly what data and rule version produced it.

BASELINE REGISTRY (B2 requirement): the Bible requires family-specific
baseline policy — at minimum, a longer rolling window for prosody, a
rest/sleep-only + SQI-gated baseline for PPG, and a context-aware
baseline for motion (never comparing running against a sedentary
baseline). This module implements the part of that requirement the
current schema can actually support:

  - A `FeatureFamily` classification and a `BaselinePolicy` registry
    keyed by family, so prosody, PPG, and motion features each get
    their own window/min-required-days instead of one constant applied
    to everything.

HONEST SCOPE NOTE: the other two required policies — PPG "rest/sleep-
only + SQI gating" and motion "context-aware baseline" (excluding
running samples from a sedentary baseline) — are NOT implemented here,
because they require data this schema does not carry. `FeatureRecord`
has no rest/sleep-state flag, no SQI/signal-quality field, and no
activity-context field to filter on (the same gap already flagged in
`temporal_alignment.py`'s SQI note). Filtering `history` by rest state
or activity context is impossible without that data existing somewhere
upstream. This is flagged as an open follow-up requiring a schema
change, not fabricated here by guessing at a context signal that
doesn't exist.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from chronis_ml.schema.models import FeatureRecord, MeasurementStatus

DEFAULT_WINDOW = timedelta(days=7)
DEFAULT_MIN_REQUIRED_DAYS = 7
DEFAULT_NEAR_ZERO_VARIANCE_THRESHOLD = 1e-6
DEFAULT_BASELINE_VERSION = "1.0"


class NormalizationConfidence(StrEnum):
    """Typed confidence state for a normalization attempt. Never
    silently collapse an edge case into a value that looks like a
    normal, trustworthy z-score."""

    NORMAL = "normal"
    INSUFFICIENT_BASELINE = "insufficient_baseline"
    """Fewer than the required number of distinct calendar days of
    backward-looking history exist. z_score is None; the caller must
    not treat this record as usable for downstream inference."""

    NEAR_ZERO_VARIANCE = "near_zero_variance"
    """Baseline mean/SD were computable, but SD is too close to zero to
    safely divide by. z_score is None; baseline_mean/baseline_sd are
    still populated for diagnostic visibility."""


class NormalizationError(ValueError):
    """Raised for a genuinely invalid call (e.g. mismatched feature
    name/user between target and history) — never for a normal edge
    case like insufficient baseline, which is a typed result, not an
    error."""


class FeatureFamily(StrEnum):
    """Coarse feature-family classification used to select a baseline
    policy. Deliberately explicit and overridable per call — inferring
    a family purely by string-matching `feature_name` is a best-effort
    default, not a claim that this classification is authoritative."""

    PROSODY = "prosody"
    PPG_RESTING = "ppg_resting"
    MOTION = "motion"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class BaselinePolicy:
    """Baseline window policy for one feature family."""

    family: FeatureFamily
    window: timedelta
    min_required_days: int


BASELINE_POLICY_REGISTRY: dict[FeatureFamily, BaselinePolicy] = {
    FeatureFamily.PROSODY: BaselinePolicy(
        family=FeatureFamily.PROSODY,
        # Prosody baselines need a longer rolling window than other
        # families per B2 ("prosody -> longer rolling personal
        # baseline") — vocal patterns vary more day-to-day than a
        # 7-day window can stably characterize.
        window=timedelta(days=21),
        min_required_days=14,
    ),
    FeatureFamily.PPG_RESTING: BaselinePolicy(
        family=FeatureFamily.PPG_RESTING,
        window=DEFAULT_WINDOW,
        min_required_days=DEFAULT_MIN_REQUIRED_DAYS,
    ),
    FeatureFamily.MOTION: BaselinePolicy(
        family=FeatureFamily.MOTION,
        window=DEFAULT_WINDOW,
        min_required_days=DEFAULT_MIN_REQUIRED_DAYS,
    ),
    FeatureFamily.OTHER: BaselinePolicy(
        family=FeatureFamily.OTHER,
        window=DEFAULT_WINDOW,
        min_required_days=DEFAULT_MIN_REQUIRED_DAYS,
    ),
}

# Best-effort feature_name -> family classification, used only when the
# caller does not pass `family` explicitly to `normalize_value`. Extend
# as more feature names are confirmed. An unrecognized feature_name
# falls back to FeatureFamily.OTHER (the same window/min-days as the
# pre-registry defaults), never to a guessed family.
_FEATURE_NAME_TO_FAMILY: dict[str, FeatureFamily] = {
    "heart_rate": FeatureFamily.PPG_RESTING,
    "respiration_rate": FeatureFamily.PPG_RESTING,
    "spo2": FeatureFamily.PPG_RESTING,
    "steps": FeatureFamily.MOTION,
}


def classify_feature_family(feature_name: str) -> FeatureFamily:
    """Best-effort family classification from a feature name alone.
    Callers with better information (e.g. a config-driven feature
    catalog) should pass `family` to `normalize_value` explicitly
    rather than relying on this."""

    return _FEATURE_NAME_TO_FAMILY.get(feature_name, FeatureFamily.OTHER)


@dataclass(frozen=True, slots=True)
class NormalizedFeatureRecord:
    """A z-score-normalized value, with full provenance. Deliberately a
    distinct type from `FeatureRecord` — a normalized value is a
    derived artifact with extra required metadata (baseline window,
    version, confidence), not a raw observation, matching the same
    "don't collapse distinct kinds into one shape" principle used
    elsewhere in this codebase."""

    user_id: str
    timestamp: datetime
    feature_name: str
    raw_value: float
    z_score: float | None
    confidence: NormalizationConfidence
    baseline_window_start: datetime
    baseline_window_end: datetime
    baseline_version: str
    baseline_mean: float | None
    baseline_sd: float | None
    baseline_sample_count: int
    family: FeatureFamily


def normalize_value(
    history: Sequence[FeatureRecord],
    target: FeatureRecord,
    *,
    family: FeatureFamily | None = None,
    window: timedelta | None = None,
    min_required_days: int | None = None,
    near_zero_variance_threshold: float = DEFAULT_NEAR_ZERO_VARIANCE_THRESHOLD,
    baseline_version: str = DEFAULT_BASELINE_VERSION,
) -> NormalizedFeatureRecord:
    """Compute a personal z-score for `target`, using only `history`
    records strictly before `target.timestamp`.

    `history` may contain records for other users/features/timestamps
    at or after target — this function filters internally rather than
    requiring the caller to pre-filter, so it is safe to pass a whole
    dataset's records for one feature and let this function select the
    correct backward-looking window itself.

    `family` selects the baseline policy (window, min_required_days)
    from `BASELINE_POLICY_REGISTRY`. If omitted, it is inferred from
    `target.feature_name` via `classify_feature_family`. Explicit
    `window`/`min_required_days` arguments, if given, override the
    policy's values for this call only.
    """

    if target.status is not MeasurementStatus.OBSERVED or target.value is None:
        raise NormalizationError(
            "target record must be an OBSERVED record with a value; "
            "normalization is undefined for a MISSING record"
        )

    resolved_family = family if family is not None else classify_feature_family(target.feature_name)
    policy = BASELINE_POLICY_REGISTRY[resolved_family]
    resolved_window = window if window is not None else policy.window
    resolved_min_required_days = (
        min_required_days if min_required_days is not None else policy.min_required_days
    )

    baseline_window_end = target.timestamp
    baseline_window_start = target.timestamp - resolved_window

    # Strictly backward-looking: timestamp must be < target.timestamp,
    # never <=, so the target's own value (and anything at the exact
    # same instant) can never leak into its own baseline.
    baseline_records = [
        record
        for record in history
        if record.user_id == target.user_id
        and record.feature_name == target.feature_name
        and record.status is MeasurementStatus.OBSERVED
        and record.value is not None
        and baseline_window_start <= record.timestamp < baseline_window_end
    ]

    distinct_days = {record.timestamp.date() for record in baseline_records}

    if len(distinct_days) < resolved_min_required_days:
        return NormalizedFeatureRecord(
            user_id=target.user_id,
            timestamp=target.timestamp,
            feature_name=target.feature_name,
            raw_value=target.value,
            z_score=None,
            confidence=NormalizationConfidence.INSUFFICIENT_BASELINE,
            baseline_window_start=baseline_window_start,
            baseline_window_end=baseline_window_end,
            baseline_version=baseline_version,
            baseline_mean=None,
            baseline_sd=None,
            baseline_sample_count=len(baseline_records),
            family=resolved_family,
        )

    values = [record.value for record in baseline_records if record.value is not None]
    mean = statistics.mean(values)
    sd = statistics.pstdev(values)

    if sd < near_zero_variance_threshold:
        return NormalizedFeatureRecord(
            user_id=target.user_id,
            timestamp=target.timestamp,
            feature_name=target.feature_name,
            raw_value=target.value,
            z_score=None,
            confidence=NormalizationConfidence.NEAR_ZERO_VARIANCE,
            baseline_window_start=baseline_window_start,
            baseline_window_end=baseline_window_end,
            baseline_version=baseline_version,
            baseline_mean=mean,
            baseline_sd=sd,
            baseline_sample_count=len(values),
            family=resolved_family,
        )

    z_score = (target.value - mean) / sd

    return NormalizedFeatureRecord(
        user_id=target.user_id,
        timestamp=target.timestamp,
        feature_name=target.feature_name,
        raw_value=target.value,
        z_score=z_score,
        confidence=NormalizationConfidence.NORMAL,
        baseline_window_start=baseline_window_start,
        baseline_window_end=baseline_window_end,
        baseline_version=baseline_version,
        baseline_mean=mean,
        baseline_sd=sd,
        baseline_sample_count=len(values),
        family=resolved_family,
    )
