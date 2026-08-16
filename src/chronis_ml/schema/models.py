"""Canonical Chronis data models.

Dataset-specific loaders must convert their source data into these models.
No dataset-specific parsing belongs in this module.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class MeasurementStatus(StrEnum):
    """Whether a measurement was observed or is missing."""

    OBSERVED = "observed"
    MISSING = "missing"


class MissingReason(StrEnum):
    """Typed reasons for a missing measurement."""

    SENSOR_FAILURE = "sensor_failure"
    NOT_WORN = "not_worn"
    AUDIO_PAUSED = "audio_paused"


@dataclass(frozen=True, slots=True)
class FeatureMetadata:
    """Metadata describing one canonical feature."""

    name: str
    modality: str
    unit: str | None = None
    description: str | None = None
    source_feature: str | None = None


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    """One user/time/feature observation."""

    user_id: str
    timestamp: datetime
    feature_name: str
    value: float | None
    modality: str
    status: MeasurementStatus
    missing_reason: MissingReason | None = None
    unit: str | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class ChronisDataset:
    """Canonical dataset returned by a dataset loader."""

    records: tuple[FeatureRecord, ...]
    features: tuple[FeatureMetadata, ...] = field(default_factory=tuple)

    @classmethod
    def from_records(
        cls,
        records: Iterable[FeatureRecord],
        features: Iterable[FeatureMetadata] = (),
    ) -> ChronisDataset:
        return cls(tuple(records), tuple(features))

    @property
    def users(self) -> tuple[str, ...]:
        return tuple(sorted({record.user_id for record in self.records}))

    @property
    def timestamps(self) -> tuple[datetime, ...]:
        return tuple(sorted({record.timestamp for record in self.records}))

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = {record.feature_name for record in self.records}
        names.update(feature.name for feature in self.features)
        return tuple(sorted(names))

    def by_user(self, user_id: str) -> tuple[FeatureRecord, ...]:
        return tuple(r for r in self.records if r.user_id == user_id)

    def by_feature(self, feature_name: str) -> tuple[FeatureRecord, ...]:
        return tuple(r for r in self.records if r.feature_name == feature_name)
