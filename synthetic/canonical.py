"""Canonical adapters — spec Section 7.3.

Converts each stream's raw CSV rows into typed canonical records,
enforcing every rule from the spec's canonicalization section:

  - Preserve source values and original timestamp representation before
    any conversion.
  - Never collapse point, interval, event, and high-frequency snippet
    data into one generic point schema — hence 4 separate record kinds
    below, not a single shape forced onto every stream.
  - Attach explicit units to every canonical field.
  - Emit typed missing/null state only when the data actually justifies
    it — never fill gaps with interpolation, forward-fill, or zeros.
  - Every canonical field must trace back to a specific source file and
    row (`SourceReference`).
  - Ambiguous schema, unexpected timezone, or an unrecognized modality
    must fail loudly with the specific file, modality, and decision
    that caused the failure — never fail silently.

Reuses `chronis_ml.schema`'s already-tested `MeasurementStatus` /
`MissingReason` typed-missingness contract and `parse_timestamp` utility
rather than rebuilding that logic — those are proven, audited (S1.2/S1.3),
and this module deliberately does not modify them.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from chronis_ml.loaders.utils import parse_timestamp
from chronis_ml.schema.models import MeasurementStatus, MissingReason


class CanonicalizationError(ValueError):
    """Base class for every canonicalization failure. Always raised
    with the specific file, modality, and decision that caused it —
    never a silent skip."""


class SchemaDriftError(CanonicalizationError):
    """A required column is missing from the ENTIRE file (every row),
    not just some rows — indicates a whole-file schema change (e.g. the
    `schema_version_bump` corruption mode), distinct from a single
    malformed row."""


class MalformedRowError(CanonicalizationError):
    """A single row is missing a required field or has an unparseable
    value, while the rest of the file's schema is intact."""


@dataclass(frozen=True, slots=True)
class SourceReference:
    """Traces a canonical record back to the exact file and row it came
    from (spec requirement)."""

    relative_path: str
    row_index: int


@dataclass(frozen=True, slots=True)
class CanonicalPointRecord:
    """A single instantaneous reading at one timestamp."""

    participant_id: str
    timestamp: datetime
    feature_name: str
    value: float | None
    unit: str | None
    status: MeasurementStatus
    missing_reason: MissingReason | None
    source: SourceReference


@dataclass(frozen=True, slots=True)
class CanonicalIntervalRecord:
    """A reading that summarizes a time span (a bucket or a stage), not
    an instant — e.g. a 1-minute step-count bucket or a sleep stage."""

    participant_id: str
    start_time: datetime
    end_time: datetime
    feature_name: str
    value: float | None
    unit: str | None
    status: MeasurementStatus
    missing_reason: MissingReason | None
    source: SourceReference


@dataclass(frozen=True, slots=True)
class CanonicalEventRecord:
    """A discrete occurrence at a point in time, optionally with a
    duration — e.g. a phone interaction or a survey response. Has no
    "missing" semantics of its own: an event either exists in the
    source data or it doesn't; there is no concept of a "missing
    event reading" the way there is for a continuous sensor stream."""

    participant_id: str
    timestamp: datetime
    event_type: str
    duration_seconds: float | None
    source: SourceReference


@dataclass(frozen=True, slots=True)
class CanonicalSnippetRecord:
    """A high-frequency waveform snippet — never collapsed into a
    single point value, per the spec's explicit rule."""

    participant_id: str
    start_time: datetime
    sampling_rate_hz: float
    sample_count: int
    samples: tuple[float, ...]
    unit: str | None
    source: SourceReference


# --- Shared helpers ---------------------------------------------------------


def _require_columns(
    header: tuple[str, ...],
    required: tuple[str, ...],
    *,
    relative_path: str,
    modality: str,
) -> None:
    missing = [column for column in required if column not in header]
    if missing:
        raise SchemaDriftError(
            f"schema drift detected in {relative_path!r} (modality={modality!r}): "
            f"required column(s) {missing} are absent from the file header "
            f"entirely — every row is missing {missing}, not just some rows"
        )


def _require_row_value(
    row: dict[str, str],
    field: str,
    *,
    relative_path: str,
    row_index: int,
) -> str:
    value = row.get(field)
    if value is None or value == "":
        raise MalformedRowError(
            f"malformed row in {relative_path!r} at row {row_index}: "
            f"required field {field!r} is missing or empty"
        )
    return value


def _parse_float(
    value: str,
    *,
    field: str,
    relative_path: str,
    row_index: int,
) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise MalformedRowError(
            f"malformed row in {relative_path!r} at row {row_index}: "
            f"field {field!r} value {value!r} is not a valid number"
        ) from exc


def _parse_timestamp_field(
    value: str,
    *,
    field: str,
    relative_path: str,
    row_index: int,
) -> datetime:
    """Wraps `parse_timestamp` so ANY parsing failure — regardless of
    the underlying exception type raised by the parsing library — is
    normalized to a `MalformedRowError` with file/row context. A raw
    third-party exception must never propagate uncaught past this
    module; that would violate the spec's "fail loudly with the
    specific file, modality, and decision" requirement by leaking an
    unrecognized, uncontextualized error instead."""

    try:
        return parse_timestamp(value)
    except Exception as exc:
        raise MalformedRowError(
            f"malformed row in {relative_path!r} at row {row_index}: "
            f"field {field!r} value {value!r} could not be parsed as a timestamp"
        ) from exc


def _deduplicate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deterministically remove exact-duplicate rows.

    Per spec Section 6: "Canonicalization must de-duplicate
    deterministically, not just take 'first seen'." Taking the first
    occurrence in file order is NOT deterministic across corruption
    modes, since `out_of_order` can shuffle which copy appears first.
    Instead, dedupe by exact row content and sort the unique rows by
    their content, so the result is identical regardless of original
    file order.
    """

    unique_signatures: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        signature = tuple(f"{key}={value}" for key, value in sorted(row.items()))
        unique_signatures[signature] = row

    return [unique_signatures[signature] for signature in sorted(unique_signatures.keys())]


def _read_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    return header, rows


# --- Per-stream adapters -----------------------------------------------------


def adapt_fitbit_heart_rate(
    relative_path: str, header: tuple[str, ...], rows: list[dict[str, str]]
) -> list[CanonicalRecord]:
    modality = "fitbit.heart_rate"
    _require_columns(
        header,
        ("timestamp", "participant_id", "heart_rate_bpm"),
        relative_path=relative_path,
        modality=modality,
    )

    rows = _deduplicate_rows(rows)
    rows.sort(key=lambda r: r.get("timestamp", ""))  # never assume file order == time order

    records: list[CanonicalRecord] = []
    for row_index, row in enumerate(rows):
        participant_id = _require_row_value(
            row, "participant_id", relative_path=relative_path, row_index=row_index
        )
        raw_timestamp = _require_row_value(
            row, "timestamp", relative_path=relative_path, row_index=row_index
        )
        raw_value = _require_row_value(
            row, "heart_rate_bpm", relative_path=relative_path, row_index=row_index
        )

        timestamp = _parse_timestamp_field(
            raw_timestamp, field="timestamp", relative_path=relative_path, row_index=row_index
        )
        value = _parse_float(
            raw_value, field="heart_rate_bpm", relative_path=relative_path, row_index=row_index
        )

        records.append(
            CanonicalPointRecord(
                participant_id=participant_id,
                timestamp=timestamp,
                feature_name="heart_rate",
                value=value,
                unit="bpm",
                status=MeasurementStatus.OBSERVED,
                missing_reason=None,
                source=SourceReference(relative_path=relative_path, row_index=row_index),
            )
        )

    return records


def adapt_fitbit_steps(
    relative_path: str, header: tuple[str, ...], rows: list[dict[str, str]]
) -> list[CanonicalRecord]:
    modality = "fitbit.steps"
    _require_columns(
        header,
        ("window_start", "window_end", "participant_id", "step_count"),
        relative_path=relative_path,
        modality=modality,
    )

    rows = _deduplicate_rows(rows)
    rows.sort(key=lambda r: r.get("window_start", ""))

    records: list[CanonicalRecord] = []
    for row_index, row in enumerate(rows):
        participant_id = _require_row_value(
            row, "participant_id", relative_path=relative_path, row_index=row_index
        )
        raw_start = _require_row_value(
            row, "window_start", relative_path=relative_path, row_index=row_index
        )
        raw_end = _require_row_value(
            row, "window_end", relative_path=relative_path, row_index=row_index
        )
        raw_value = _require_row_value(
            row, "step_count", relative_path=relative_path, row_index=row_index
        )

        records.append(
            CanonicalIntervalRecord(
                participant_id=participant_id,
                start_time=_parse_timestamp_field(
                    raw_start,
                    field="window_start",
                    relative_path=relative_path,
                    row_index=row_index,
                ),
                end_time=_parse_timestamp_field(
                    raw_end, field="window_end", relative_path=relative_path, row_index=row_index
                ),
                feature_name="step_count",
                value=_parse_float(
                    raw_value, field="step_count", relative_path=relative_path, row_index=row_index
                ),
                unit="steps",
                status=MeasurementStatus.OBSERVED,
                missing_reason=None,
                source=SourceReference(relative_path=relative_path, row_index=row_index),
            )
        )

    return records


def adapt_fitbit_sleep(
    relative_path: str, header: tuple[str, ...], rows: list[dict[str, str]]
) -> list[CanonicalRecord]:
    modality = "fitbit.sleep"
    _require_columns(
        header,
        (
            "sleep_session_id",
            "participant_id",
            "start_time",
            "end_time",
            "stage",
            "stage_duration_seconds",
        ),
        relative_path=relative_path,
        modality=modality,
    )

    rows = _deduplicate_rows(rows)
    rows.sort(key=lambda r: r.get("start_time", ""))

    records: list[CanonicalRecord] = []
    for row_index, row in enumerate(rows):
        participant_id = _require_row_value(
            row, "participant_id", relative_path=relative_path, row_index=row_index
        )
        raw_start = _require_row_value(
            row, "start_time", relative_path=relative_path, row_index=row_index
        )
        raw_end = _require_row_value(
            row, "end_time", relative_path=relative_path, row_index=row_index
        )
        stage = _require_row_value(row, "stage", relative_path=relative_path, row_index=row_index)
        raw_duration = _require_row_value(
            row, "stage_duration_seconds", relative_path=relative_path, row_index=row_index
        )

        records.append(
            CanonicalIntervalRecord(
                participant_id=participant_id,
                start_time=_parse_timestamp_field(
                    raw_start, field="start_time", relative_path=relative_path, row_index=row_index
                ),
                end_time=_parse_timestamp_field(
                    raw_end, field="end_time", relative_path=relative_path, row_index=row_index
                ),
                feature_name=f"sleep_stage_{stage}",
                value=_parse_float(
                    raw_duration,
                    field="stage_duration_seconds",
                    relative_path=relative_path,
                    row_index=row_index,
                ),
                unit="seconds",
                status=MeasurementStatus.OBSERVED,
                missing_reason=None,
                source=SourceReference(relative_path=relative_path, row_index=row_index),
            )
        )

    return records


def adapt_chest_ecg_snippet(
    relative_path: str, header: tuple[str, ...], rows: list[dict[str, str]]
) -> list[CanonicalRecord]:
    import json

    modality = "chest_ecg.snippet"
    _require_columns(
        header,
        (
            "snippet_id",
            "participant_id",
            "start_time",
            "sampling_rate_hz",
            "sample_count",
            "samples",
        ),
        relative_path=relative_path,
        modality=modality,
    )

    rows = _deduplicate_rows(rows)
    rows.sort(key=lambda r: r.get("start_time", ""))

    records: list[CanonicalRecord] = []
    for row_index, row in enumerate(rows):
        participant_id = _require_row_value(
            row, "participant_id", relative_path=relative_path, row_index=row_index
        )
        raw_start = _require_row_value(
            row, "start_time", relative_path=relative_path, row_index=row_index
        )
        raw_rate = _require_row_value(
            row, "sampling_rate_hz", relative_path=relative_path, row_index=row_index
        )
        raw_count = _require_row_value(
            row, "sample_count", relative_path=relative_path, row_index=row_index
        )
        raw_samples = _require_row_value(
            row, "samples", relative_path=relative_path, row_index=row_index
        )

        try:
            samples = tuple(float(s) for s in json.loads(raw_samples))
        except (ValueError, TypeError) as exc:
            raise MalformedRowError(
                f"malformed row in {relative_path!r} at row {row_index}: "
                f"'samples' field is not valid JSON array of numbers"
            ) from exc

        records.append(
            CanonicalSnippetRecord(
                participant_id=participant_id,
                start_time=_parse_timestamp_field(
                    raw_start, field="start_time", relative_path=relative_path, row_index=row_index
                ),
                sampling_rate_hz=_parse_float(
                    raw_rate,
                    field="sampling_rate_hz",
                    relative_path=relative_path,
                    row_index=row_index,
                ),
                sample_count=int(
                    _parse_float(
                        raw_count,
                        field="sample_count",
                        relative_path=relative_path,
                        row_index=row_index,
                    )
                ),
                samples=samples,
                unit="millivolts",
                source=SourceReference(relative_path=relative_path, row_index=row_index),
            )
        )

    return records


def adapt_audio_features(
    relative_path: str, header: tuple[str, ...], rows: list[dict[str, str]]
) -> list[CanonicalRecord]:
    modality = "audio_features.summary"
    _require_columns(
        header,
        ("timestamp", "participant_id", "voice_activity_score", "pitch_variance"),
        relative_path=relative_path,
        modality=modality,
    )

    rows = _deduplicate_rows(rows)
    rows.sort(key=lambda r: r.get("timestamp", ""))

    records: list[CanonicalRecord] = []
    for row_index, row in enumerate(rows):
        participant_id = _require_row_value(
            row, "participant_id", relative_path=relative_path, row_index=row_index
        )
        raw_timestamp = _require_row_value(
            row, "timestamp", relative_path=relative_path, row_index=row_index
        )
        timestamp = _parse_timestamp_field(
            raw_timestamp, field="timestamp", relative_path=relative_path, row_index=row_index
        )

        for field, unit in (
            ("voice_activity_score", "score_0_to_1"),
            ("pitch_variance", "hz_variance"),
        ):
            raw_value = _require_row_value(
                row, field, relative_path=relative_path, row_index=row_index
            )
            records.append(
                CanonicalPointRecord(
                    participant_id=participant_id,
                    timestamp=timestamp,
                    feature_name=field,
                    value=_parse_float(
                        raw_value, field=field, relative_path=relative_path, row_index=row_index
                    ),
                    unit=unit,
                    status=MeasurementStatus.OBSERVED,
                    missing_reason=None,
                    source=SourceReference(relative_path=relative_path, row_index=row_index),
                )
            )

    return records


def adapt_proximity(
    relative_path: str, header: tuple[str, ...], rows: list[dict[str, str]]
) -> list[CanonicalRecord]:
    modality = "proximity.participant_beacon"
    _require_columns(
        header,
        ("timestamp", "source_id", "source_type", "target_id", "target_type", "rssi_dbm"),
        relative_path=relative_path,
        modality=modality,
    )

    rows = _deduplicate_rows(rows)
    rows.sort(key=lambda r: r.get("timestamp", ""))

    records: list[CanonicalRecord] = []
    for row_index, row in enumerate(rows):
        participant_id = _require_row_value(
            row, "source_id", relative_path=relative_path, row_index=row_index
        )
        raw_timestamp = _require_row_value(
            row, "timestamp", relative_path=relative_path, row_index=row_index
        )
        target_id = _require_row_value(
            row, "target_id", relative_path=relative_path, row_index=row_index
        )
        raw_rssi = _require_row_value(
            row, "rssi_dbm", relative_path=relative_path, row_index=row_index
        )

        records.append(
            CanonicalPointRecord(
                participant_id=participant_id,
                timestamp=_parse_timestamp_field(
                    raw_timestamp,
                    field="timestamp",
                    relative_path=relative_path,
                    row_index=row_index,
                ),
                feature_name=f"proximity_rssi_{target_id}",
                value=_parse_float(
                    raw_rssi, field="rssi_dbm", relative_path=relative_path, row_index=row_index
                ),
                unit="dbm",
                status=MeasurementStatus.OBSERVED,
                missing_reason=None,
                source=SourceReference(relative_path=relative_path, row_index=row_index),
            )
        )

    return records


def adapt_environment(
    relative_path: str, header: tuple[str, ...], rows: list[dict[str, str]]
) -> list[CanonicalRecord]:
    modality = "environment.device"
    _require_columns(
        header,
        (
            "timestamp",
            "device_id",
            "light_lux",
            "motion_x",
            "motion_y",
            "motion_z",
            "temperature_c",
            "humidity_pct",
        ),
        relative_path=relative_path,
        modality=modality,
    )

    rows = _deduplicate_rows(rows)
    rows.sort(key=lambda r: r.get("timestamp", ""))

    field_units = {
        "light_lux": "lux",
        "motion_x": "g",
        "motion_y": "g",
        "motion_z": "g",
        "temperature_c": "celsius",
        "humidity_pct": "percent",
    }

    records: list[CanonicalRecord] = []
    for row_index, row in enumerate(rows):
        device_id = _require_row_value(
            row, "device_id", relative_path=relative_path, row_index=row_index
        )
        raw_timestamp = _require_row_value(
            row, "timestamp", relative_path=relative_path, row_index=row_index
        )
        timestamp = _parse_timestamp_field(
            raw_timestamp, field="timestamp", relative_path=relative_path, row_index=row_index
        )

        for field, unit in field_units.items():
            raw_value = _require_row_value(
                row, field, relative_path=relative_path, row_index=row_index
            )
            records.append(
                CanonicalPointRecord(
                    participant_id=device_id,  # device-scoped: this is really the device
                    timestamp=timestamp,
                    feature_name=field,
                    value=_parse_float(
                        raw_value, field=field, relative_path=relative_path, row_index=row_index
                    ),
                    unit=unit,
                    status=MeasurementStatus.OBSERVED,
                    missing_reason=None,
                    source=SourceReference(relative_path=relative_path, row_index=row_index),
                )
            )

    return records


def adapt_phone_events(
    relative_path: str, header: tuple[str, ...], rows: list[dict[str, str]]
) -> list[CanonicalRecord]:
    modality = "phone_events.interaction"
    _require_columns(
        header,
        ("timestamp", "participant_id", "event_type", "duration_seconds"),
        relative_path=relative_path,
        modality=modality,
    )

    rows = _deduplicate_rows(rows)
    rows.sort(key=lambda r: r.get("timestamp", ""))

    records: list[CanonicalRecord] = []
    for row_index, row in enumerate(rows):
        participant_id = _require_row_value(
            row, "participant_id", relative_path=relative_path, row_index=row_index
        )
        raw_timestamp = _require_row_value(
            row, "timestamp", relative_path=relative_path, row_index=row_index
        )
        event_type = _require_row_value(
            row, "event_type", relative_path=relative_path, row_index=row_index
        )
        raw_duration = _require_row_value(
            row, "duration_seconds", relative_path=relative_path, row_index=row_index
        )

        records.append(
            CanonicalEventRecord(
                participant_id=participant_id,
                timestamp=_parse_timestamp_field(
                    raw_timestamp,
                    field="timestamp",
                    relative_path=relative_path,
                    row_index=row_index,
                ),
                event_type=event_type,
                duration_seconds=_parse_float(
                    raw_duration,
                    field="duration_seconds",
                    relative_path=relative_path,
                    row_index=row_index,
                ),
                source=SourceReference(relative_path=relative_path, row_index=row_index),
            )
        )

    return records


def adapt_surveys_ema(
    relative_path: str, header: tuple[str, ...], rows: list[dict[str, str]]
) -> list[CanonicalRecord]:
    modality = "surveys.ema"
    _require_columns(
        header,
        ("timestamp", "participant_id", "survey_type", "stress_1_to_5", "response_id"),
        relative_path=relative_path,
        modality=modality,
    )

    rows = _deduplicate_rows(rows)
    rows.sort(key=lambda r: r.get("timestamp", ""))

    records: list[CanonicalRecord] = []
    for row_index, row in enumerate(rows):
        participant_id = _require_row_value(
            row, "participant_id", relative_path=relative_path, row_index=row_index
        )
        raw_timestamp = _require_row_value(
            row, "timestamp", relative_path=relative_path, row_index=row_index
        )
        survey_type = _require_row_value(
            row, "survey_type", relative_path=relative_path, row_index=row_index
        )
        raw_score = _require_row_value(
            row, "stress_1_to_5", relative_path=relative_path, row_index=row_index
        )

        timestamp = _parse_timestamp_field(
            raw_timestamp, field="timestamp", relative_path=relative_path, row_index=row_index
        )

        records.append(
            CanonicalEventRecord(
                participant_id=participant_id,
                timestamp=timestamp,
                event_type=f"ema_survey_{survey_type}",
                duration_seconds=None,
                source=SourceReference(relative_path=relative_path, row_index=row_index),
            )
        )
        # stress score is carried as a second, related point-style fact;
        # modeled as a distinct canonical point so numeric downstream
        # analysis doesn't need to parse it back out of an event_type string.
        records.append(
            CanonicalPointRecord(
                participant_id=participant_id,
                timestamp=timestamp,
                feature_name="ema_stress_score",
                value=_parse_float(
                    raw_score,
                    field="stress_1_to_5",
                    relative_path=relative_path,
                    row_index=row_index,
                ),
                unit="score_1_to_5",
                status=MeasurementStatus.OBSERVED,
                missing_reason=None,
                source=SourceReference(relative_path=relative_path, row_index=row_index),
            )
        )

    return records


CanonicalRecord = (
    CanonicalPointRecord | CanonicalIntervalRecord | CanonicalEventRecord | CanonicalSnippetRecord
)
"""Union of every canonical record kind — used wherever a function must
accept/return "any canonical record" generically, without collapsing
the 4 distinct shapes into one class."""


ADAPTERS: dict[
    str, Callable[[str, tuple[str, ...], list[dict[str, str]]], list[CanonicalRecord]]
] = {
    "fitbit.heart_rate": adapt_fitbit_heart_rate,
    "fitbit.steps": adapt_fitbit_steps,
    "fitbit.sleep": adapt_fitbit_sleep,
    "chest_ecg.snippet": adapt_chest_ecg_snippet,
    "audio_features.summary": adapt_audio_features,
    "proximity.participant_beacon": adapt_proximity,
    "environment.device": adapt_environment,
    "phone_events.interaction": adapt_phone_events,
    "surveys.ema": adapt_surveys_ema,
}


def adapt_file(path: Path, stream_name: str, relative_path: str) -> list[CanonicalRecord]:
    """Read and canonicalize one file, dispatching to the correct
    per-stream adapter. Fails loudly (never silently) for an
    unrecognized modality, per spec Section 7.3."""

    if stream_name not in ADAPTERS:
        raise CanonicalizationError(
            f"unrecognized modality {stream_name!r} for file {relative_path!r}: "
            f"no adapter registered"
        )

    header, rows = _read_rows(path)
    return ADAPTERS[stream_name](relative_path, header, rows)
