from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chronis_ml.schema.classification import DataClassification


class MeasurementStatus(StrEnum):
    OBSERVED = "observed"
    MISSING = "missing"


class MissingReason(StrEnum):
    SENSOR_FAILURE = "sensor_failure"
    NOT_WORN = "not_worn"
    AUDIO_PAUSED = "audio_paused"


@dataclass(frozen=True, slots=True)
class MissingnessSignals:
    imu_stillness: bool = False
    ppg_dropout: bool = False
    mic_off_event: bool = False


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    name: str
    modality: str
    unit: str | None = None
    description: str | None = None
    source_feature: str | None = None


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    user_id: str
    timestamp: datetime
    feature_name: str
    value: float | None
    modality: str
    status: MeasurementStatus
    missing_reason: MissingReason | None = None
    unit: str | None = None
    source: str | None = None
    schema_version: str = "1.0"
    classification: DataClassification | None = None
    """Optional T4A source/object-type/representation classification.
    None by default (fully backward-compatible with every existing
    record). When present, `DataClassification.__post_init__` has
    already enforced structural validity — this field can never hold
    an invalid classification."""

    observation_id: str | None = None
    """A stable identifier for the specific raw observation this
    record represents, assigned by the loader that produced it (e.g. a
    source-system record ID or a UUID minted at ingestion). None by
    default — fully backward-compatible with every existing record and
    every existing loader that doesn't set it.

    This exists so a downstream store (e.g. FeatureStore) can record a
    real observation -> feature link instead of a free-text,
    unvalidated description of where a value "came from" (B7's
    traceability chain). When a loader doesn't provide one, callers can
    fall back to `composite_key()` below as a deterministic identifier
    — weaker than a true source-system ID (it identifies "this
    (user, time, feature, modality) slot," not "this specific raw
    reading," so it can't distinguish a corrected re-ingestion from the
    original), but still real and reproducible, never fabricated.
    """

    clock_uncertainty_ms: float | None = None
    """Estimated uncertainty, in milliseconds, in this record's
    `timestamp` — e.g. due to device clock drift, buffered/batched
    upload, or NTP sync error at capture time. None when the source
    device/loader provides no such estimate; this is never inferred or
    guessed downstream, only ever set by whatever is closest to the
    original capture.
    """

    sync_source: str | None = None
    """What `timestamp` was synchronized against at capture time (e.g.
    "device_ntp", "phone_clock", "server_receipt_time"). None when
    unknown. Exists alongside `clock_uncertainty_ms` so a consumer can
    tell not just how uncertain a timestamp might be, but what kind of
    clock produced it — the two are B6's "clock uncertainty and
    synchronization metadata" requirement, previously absent from this
    schema entirely (see `temporal_alignment.py`'s HONEST SCOPE NOTE 3).
    """

    rest_state: str | None = None
    """Whether the wearer was at rest/asleep when this reading was
    captured — e.g. "resting", "sleep", "active". None when unknown.
    Exists so a PPG baseline can be restricted to rest/sleep-only
    readings per B2 ("PPG -> rest/sleep-only... baseline"); without
    this field, `normalizer.py` had no way to distinguish a resting
    heart-rate reading from one taken mid-exercise when building a
    personal baseline. Set by whatever pipeline stage can determine
    wearer state (e.g. from IMU features) — never guessed here.
    """

    signal_quality: str | None = None
    """Signal-quality flag (SQI) for this specific reading — e.g.
    "clean", "degraded". None when the source has no quality
    assessment. Exists so a PPG baseline can be gated on quality per B2
    ("PPG -> ...+ SQI gating") — this is a per-FeatureRecord quality
    signal, distinct from `feature_store`'s `quality` table, which
    tracks quality of a *derived* feature value rather than a raw
    observation.
    """

    activity_context: str | None = None
    """A coarse activity-context tag for this reading — e.g.
    "sedentary", "walking", "running". None when unknown. Exists so a
    motion baseline can be restricted to matching-context history per
    B2 ("motion -> context-aware baseline... never comparing running
    motion with sedentary baselines"). Deliberately a free-form string
    rather than an enum for now, since the real set of contexts this
    schema needs to support isn't yet finalized — narrowing it to an
    enum prematurely would be guessing at that decision.
    """

    def composite_key(self) -> tuple[str, datetime, str, str]:
        """Deterministic fallback identifier for this record, usable as
        an observation link when `observation_id` is not set. Matches
        the same (user_id, timestamp, feature_name, modality) key
        `validation.py` already uses to detect duplicate observations —
        deliberately reusing that key rather than inventing a second,
        possibly-inconsistent notion of "identity" for the same record.
        """
        return (self.user_id, self.timestamp, self.feature_name, self.modality)


@dataclass(frozen=True, slots=True)
class ChronisDataset:
    records: tuple[FeatureRecord, ...]
    features: tuple[FeatureMetadata, ...] = field(default_factory=tuple)

    @classmethod
    def from_records(
        cls, records: Iterable[FeatureRecord], features: Iterable[FeatureMetadata] = ()
    ) -> ChronisDataset:
        return cls(tuple(records), tuple(features))

    @property
    def users(self) -> tuple[str, ...]:
        return tuple(sorted({r.user_id for r in self.records}))

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = {r.feature_name for r in self.records}
        names.update(f.name for f in self.features)
        return tuple(sorted(names))

    def by_user(self, user_id: str) -> tuple[FeatureRecord, ...]:
        return tuple(r for r in self.records if r.user_id == user_id)

    def by_feature(self, feature_name: str) -> tuple[FeatureRecord, ...]:
        return tuple(r for r in self.records if r.feature_name == feature_name)
