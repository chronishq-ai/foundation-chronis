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
