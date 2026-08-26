"""Participant/file index and manifest builder — spec Section 7.

Same responsibilities as the original Sprint 1 design (this module name
is carried over deliberately, per the Sprint 1B spec's note that it is
"same responsibilities as before, now synthetic-only"):

  - discover(root)          - walk the tree, enumerate files cheaply.
  - probe_file(path)        - inspect header + a small sample, report
                               detected fields rather than assuming a
                               fixed schema.
  - infer_stream(path)      - map a file to its modality via a
                               registry, never by guessing from filename
                               patterns alone.
  - build_manifest()        - produce a reproducible manifest record.
  - validate_manifest()     - reject a manifest that no longer matches
                               what's actually on disk.
  - save_cache()/load_cache() - cache is only valid while source
                               fingerprint, parser version, and config
                               all match.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from synthetic.config import GenConfig
from synthetic.corruption import DEFAULT_TIME_FIELDS, TIME_FIELDS_BY_STREAM
from synthetic.writer import GRANULARITY_BY_STREAM, STREAM_PATH_PREFIXES

PARSER_VERSION = "1.0"
"""This indexer module's own version — separate from the generator's
version. A cache built by an older parser version must be invalidated
even if the underlying source files haven't changed at all, since the
parsing/manifest logic itself may have changed."""

DATASET_IDENTIFIER = "chronis-synthetic"


class ManifestValidationError(ValueError):
    """Raised when a manifest no longer matches what's on disk, or was
    built by an incompatible parser/config."""


# --- Discovery ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DiscoveredFile:
    """One file found under synthetic_root, enumerated without reading
    its full contents (spec: "walk the tree and enumerate files without
    reading full contents")."""

    absolute_path: Path
    relative_path: str
    size_bytes: int
    mtime: float


def discover(root: Path) -> tuple[DiscoveredFile, ...]:
    """Walk `root` and enumerate every regular file, cheaply (stat
    only, no content read). Returned in a deterministic (sorted)
    order so downstream manifest building is itself reproducible."""

    if not root.exists():
        raise FileNotFoundError(f"synthetic_root does not exist: {root}")

    files = []
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            files.append(
                DiscoveredFile(
                    absolute_path=path,
                    relative_path=str(path.relative_to(root)).replace("\\", "/"),
                    size_bytes=stat.st_size,
                    mtime=stat.st_mtime,
                )
            )

    return tuple(sorted(files, key=lambda f: f.relative_path))


# --- Probing ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """A cheap inspection of one file's header and a small sample of
    rows — never a full-file read for large files (callers that also
    need full time-coverage scanning do that separately)."""

    columns: tuple[str, ...]
    sample_rows: tuple[dict[str, str], ...]


def probe_file(path: Path, *, sample_size: int = 5) -> ProbeResult:
    """Inspect a CSV file's header row and up to `sample_size` data
    rows, reporting the detected fields rather than assuming a fixed
    schema (spec requirement)."""

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())

        sample_rows = []
        for row in reader:
            sample_rows.append(dict(row))
            if len(sample_rows) >= sample_size:
                break

    return ProbeResult(columns=columns, sample_rows=tuple(sample_rows))


# --- Stream inference -------------------------------------------------------


def infer_stream(relative_path: str) -> str:
    """Map a file's relative path to its registered stream name via
    `STREAM_PATH_PREFIXES` — a longest-prefix match against the SAME
    registry the writer uses, never a filename-pattern guess.

    Fails loudly (per spec Section 7.3) if no registered prefix
    matches, rather than silently skipping or mis-classifying the file.
    """

    normalized = relative_path.replace("\\", "/")

    matches = [
        (prefix, stream_name)
        for stream_name, prefix in STREAM_PATH_PREFIXES.items()
        if normalized.startswith(prefix + "/")
    ]

    if not matches:
        raise ManifestValidationError(
            f"unrecognized modality: no registered stream prefix matches {relative_path!r}"
        )

    # Longest-prefix match, in case of any future overlapping prefixes
    # (e.g. "fitbit" vs "fitbit/heart_rate").
    _, stream_name = max(matches, key=lambda item: len(item[0]))
    return stream_name


def _extract_group_id(relative_path: str, stream_name: str) -> str:
    """Extract the participant_id/device_id path segment immediately
    after the stream's prefix directory."""

    prefix = STREAM_PATH_PREFIXES[stream_name]
    remainder = relative_path[len(prefix) + 1 :]  # strip "prefix/"
    group_id = remainder.split("/")[0]
    return group_id


# --- Manifest -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    relative_path: str
    stream_name: str
    modality: str
    granularity: str
    participant_id: str | None
    """Per spec: "Participant ID where participant-scoped." None for
    device-scoped streams (e.g. environment.device)."""
    size_bytes: int
    checksum_sha256: str
    schema_fingerprint: str
    time_coverage_start: str | None
    time_coverage_end: str | None
    source_fingerprint: str
    """hash(file content checksum + mtime) — this specific file's
    fingerprint, used for per-file cache invalidation."""

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "stream_name": self.stream_name,
            "modality": self.modality,
            "granularity": self.granularity,
            "participant_id": self.participant_id,
            "size_bytes": self.size_bytes,
            "checksum_sha256": self.checksum_sha256,
            "schema_fingerprint": self.schema_fingerprint,
            "time_coverage_start": self.time_coverage_start,
            "time_coverage_end": self.time_coverage_end,
            "source_fingerprint": self.source_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ManifestEntry:
        return cls(**data)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Manifest:
    dataset_identifier: str
    generator_version: str
    parser_version: str
    config_fingerprint: str
    created_at: str
    entries: tuple[ManifestEntry, ...]
    dataset_fingerprint: str
    """Aggregate hash over every entry's source_fingerprint — a single
    value that changes if ANY file in the dataset changes, is added, or
    is removed."""

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_identifier": self.dataset_identifier,
            "generator_version": self.generator_version,
            "parser_version": self.parser_version,
            "config_fingerprint": self.config_fingerprint,
            "created_at": self.created_at,
            "entries": [entry.to_dict() for entry in self.entries],
            "dataset_fingerprint": self.dataset_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Manifest:
        raw_entries = cast(list[dict[str, object]], data["entries"])
        entries = tuple(ManifestEntry.from_dict(e) for e in raw_entries)
        return cls(
            dataset_identifier=data["dataset_identifier"],  # type: ignore[arg-type]
            generator_version=data["generator_version"],  # type: ignore[arg-type]
            parser_version=data["parser_version"],  # type: ignore[arg-type]
            config_fingerprint=data["config_fingerprint"],  # type: ignore[arg-type]
            created_at=data["created_at"],  # type: ignore[arg-type]
            entries=entries,
            dataset_fingerprint=data["dataset_fingerprint"],  # type: ignore[arg-type]
        )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def config_fingerprint(config: GenConfig) -> str:
    serialized = json.dumps(config.to_dict(), sort_keys=True)
    return _sha256_text(serialized)


def _build_entry(discovered: DiscoveredFile) -> ManifestEntry:
    stream_name = infer_stream(discovered.relative_path)
    granularity = GRANULARITY_BY_STREAM[stream_name]
    group_id = _extract_group_id(discovered.relative_path, stream_name)

    content = discovered.absolute_path.read_bytes()
    checksum = _sha256_bytes(content)

    probe = probe_file(discovered.absolute_path)
    schema_fingerprint = _sha256_text(",".join(probe.columns))

    time_start, time_end = _detect_time_coverage(discovered.absolute_path, stream_name)

    source_fingerprint = _sha256_text(f"{checksum}:{discovered.mtime}")

    return ManifestEntry(
        relative_path=discovered.relative_path,
        stream_name=stream_name,
        modality=stream_name.split(".")[0],
        granularity=granularity,
        participant_id=group_id if granularity == "participant" else None,
        size_bytes=discovered.size_bytes,
        checksum_sha256=checksum,
        schema_fingerprint=schema_fingerprint,
        time_coverage_start=time_start,
        time_coverage_end=time_end,
        source_fingerprint=source_fingerprint,
    )


def _detect_time_coverage(path: Path, stream_name: str) -> tuple[str | None, str | None]:
    """Full-file scan of the relevant time field(s) to find min/max
    coverage. Documented as a full scan (not a cheap sample) because
    accurate coverage detection needs it — callers with very large
    files may want to swap this for a sampled/indexed approach later,
    which is an open follow-up, not implemented here."""

    time_fields = TIME_FIELDS_BY_STREAM.get(stream_name, DEFAULT_TIME_FIELDS)
    primary_field = time_fields[0]

    values: list[str] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get(primary_field)
            if value:
                values.append(value)

    if not values:
        return None, None

    return min(values), max(values)


def build_manifest(root: Path, config: GenConfig) -> Manifest:
    """Discover every file under `root`, probe and classify each one,
    and produce a single reproducible `Manifest`."""

    discovered_files = discover(root)

    entries = tuple(_build_entry(f) for f in discovered_files)

    dataset_fingerprint = _sha256_text(
        ",".join(sorted(entry.source_fingerprint for entry in entries))
    )

    return Manifest(
        dataset_identifier=DATASET_IDENTIFIER,
        generator_version=config.generator_version,
        parser_version=PARSER_VERSION,
        config_fingerprint=config_fingerprint(config),
        created_at=datetime.now(UTC).isoformat(),
        entries=entries,
        dataset_fingerprint=dataset_fingerprint,
    )


def validate_manifest(manifest: Manifest, root: Path) -> None:
    """Reject a manifest that no longer matches what's actually on
    disk. Raises `ManifestValidationError` with the specific mismatch,
    never silently accepts stale data."""

    if manifest.parser_version != PARSER_VERSION:
        raise ManifestValidationError(
            f"manifest was built by parser_version={manifest.parser_version!r}, "
            f"current parser_version={PARSER_VERSION!r}"
        )

    current_files = {f.relative_path: f for f in discover(root)}

    for entry in manifest.entries:
        current = current_files.get(entry.relative_path)
        if current is None:
            raise ManifestValidationError(
                f"file referenced in manifest is missing: {entry.relative_path}"
            )

        current_checksum = _sha256_bytes(current.absolute_path.read_bytes())
        current_fingerprint = _sha256_text(f"{current_checksum}:{current.mtime}")

        if current_fingerprint != entry.source_fingerprint:
            raise ManifestValidationError(
                f"source file changed since manifest was built: {entry.relative_path}"
            )

    manifest_paths = {entry.relative_path for entry in manifest.entries}
    current_paths = set(current_files.keys())
    new_files = current_paths - manifest_paths
    if new_files:
        raise ManifestValidationError(
            f"{len(new_files)} file(s) on disk are not covered by the manifest: "
            f"{sorted(new_files)[:3]}..."
        )


# --- Cache ----------------------------------------------------------------


def save_cache(manifest: Manifest, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")


def load_cache(
    cache_path: Path,
    *,
    expected_config_fingerprint: str,
    expected_generator_version: str,
) -> Manifest | None:
    """Load a cached manifest IF AND ONLY IF its parser version,
    generator version, and config fingerprint all still match what the
    caller expects. Returns None (never raises) on any mismatch or
    missing cache file — the caller is expected to rebuild in that
    case, this function only decides freshness at the metadata level,
    not by re-scanning every file on disk (see `validate_manifest` for
    the more expensive full re-verification against disk).
    """

    if not cache_path.exists():
        return None

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        manifest = Manifest.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None

    if manifest.parser_version != PARSER_VERSION:
        return None
    if manifest.generator_version != expected_generator_version:
        return None
    if manifest.config_fingerprint != expected_config_fingerprint:
        return None

    return manifest
