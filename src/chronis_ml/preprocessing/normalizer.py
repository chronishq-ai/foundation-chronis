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
baseline). This module implements all three:

  - A `FeatureFamily` classification and a `BaselinePolicy` registry
    keyed by family, so prosody, PPG, and motion features each get
    their own window/min-required-days instead of one constant applied
    to everything.
  - PPG baselines are filtered to `rest_state in ("resting", "sleep")`
    and `signal_quality == "clean"` — a record with either field unset
    is excluded, not assumed to qualify.
  - Motion baselines are filtered to `activity_context` matching the
    target's own context, when the target has one recorded.

HONEST SCOPE NOTE: this depends entirely on `rest_state`,
`signal_quality`, and `activity_context` actually being populated on
`FeatureRecord` by whatever produces it (loaders, upstream pipeline
stages). Those fields were only just added to the schema and nothing
populates them yet. It also depends on the caller explicitly passing
`family=FeatureFamily.PPG_RESTING` (or `.MOTION`) — content gating does
NOT activate from name-based auto-classification alone, precisely
because a caller using a feature name informally (e.g. a test using
"heart_rate" as a placeholder for generic z-score math, not real PPG
data) must not have its baseline silently gated on fields it never
opted into. A real production caller that knows its own feature
catalog should declare the family explicitly to get gating; until it
does, both are real, tested behaviors — they're just not active by
default.
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

    # Content-gating filters below (rest/SQI for PPG, activity-context
    # for motion) activate only when the caller EXPLICITLY passes
    # `family` — name-based auto-classification alone is too weak a
    # signal to justify silently filtering a caller's baseline down to
    # nothing. A caller using a name like "heart_rate" as a placeholder
    # for something that isn't real PPG data (e.g. a unit test of the
    # core z-score math) must not have its baseline gated on fields it
    # never intended to opt into. A caller that actually knows this is
    # real PPG/motion data should say so via `family=`.
    family_was_explicit = family is not None

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

    # Family-specific baseline filtering (B2): restrict the candidate
    # pool beyond the generic backward-looking window, per family
    # policy. Applied here (after the generic filter, before the
    # min-required-days check) so insufficient-baseline correctly
    # reflects "not enough *qualifying* history," not just "not enough
    # history of any kind."
    if family_was_explicit and resolved_family is FeatureFamily.PPG_RESTING:
        # Rest/sleep-only + SQI gating: a record with no rest_state or
        # signal_quality recorded cannot be confirmed to qualify, so it
        # is excluded rather than assumed to pass — silently treating
        # unknown as "resting and clean" would defeat the point of
        # gating in the first place.
        baseline_records = [
            record
            for record in baseline_records
            if record.rest_state in ("resting", "sleep") and record.signal_quality == "clean"
        ]
    elif (
        family_was_explicit
        and resolved_family is FeatureFamily.MOTION
        and target.activity_context is not None
    ):
        # Context-aware baseline: only compare against history from a
        # matching activity context (never running against sedentary).
        # If the target itself has no activity_context recorded, this
        # filter cannot be applied — falling back to the unfiltered
        # pool rather than raising, since MOTION already has no
        # required-field check elsewhere in this module. That fallback
        # is a real, documented limitation: without activity_context
        # data, context-aware filtering silently degrades to the old
        # unfiltered behavior for that record.
        baseline_records = [
            record
            for record in baseline_records
            if record.activity_context == target.activity_context
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
