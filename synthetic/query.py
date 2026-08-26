"""Participant-scoped query API.

This is deliberately a SEPARATE, narrower surface from `discover()` and
`build_manifest()` (both of which are global, dataset-wide operations).
Every function here takes a specific participant_id and is designed so
that it CANNOT, even under a manifest bug or a corrupted file, return
another participant's data.

Two layers of defense, both required:
  1. Manifest-level filtering: only consider entries whose
     `ManifestEntry.participant_id` matches the requested ID.
  2. Record-level filtering: after canonicalizing a file, keep only
     records whose OWN participant_id field (from the actual row
     content) matches — never trust the manifest/path alone. This is
     what makes the path-based-bypass isolation test meaningful.
"""

from __future__ import annotations

from pathlib import Path

from synthetic.canonical import CanonicalRecord, adapt_file
from synthetic.tiles_participant_index import Manifest, ManifestEntry


class InvalidParticipantIdError(ValueError):
    """Raised for a null/empty participant_id. An unknown-but-well-formed
    ID is NOT an error — see `entries_for_participant`'s docstring."""


def _validate_participant_id(participant_id: object) -> str:
    if not isinstance(participant_id, str) or not participant_id.strip():
        raise InvalidParticipantIdError(
            f"participant_id must be a non-empty string, got {participant_id!r}"
        )
    return participant_id


def entries_for_participant(manifest: Manifest, participant_id: str) -> tuple[ManifestEntry, ...]:
    """Return only the manifest entries belonging to `participant_id`.

    - A null/empty participant_id raises `InvalidParticipantIdError`
      immediately — never silently treated as "all participants."
    - An unknown-but-well-formed participant_id (no matching entries)
      returns an empty tuple, not an error and never a fallback to any
      other participant's data.
    - Device-scoped entries (`entry.participant_id is None`) are NEVER
      returned here, regardless of participant_id — use
      `entries_for_device` for those.
    - Matching is EXACT equality only, never substring/prefix matching
      — see the permanent `leaky_fixture` regression test for why this
      matters.
    """

    validated_id = _validate_participant_id(participant_id)

    return tuple(entry for entry in manifest.entries if entry.participant_id == validated_id)


def entries_for_participant_and_stream(
    manifest: Manifest, participant_id: str, stream_name: str
) -> tuple[ManifestEntry, ...]:
    """Further restrict `entries_for_participant` to one specific
    stream. Querying participant A with a stream_name that only
    participant B has data for must return empty — proven by the
    isolation suite's cross-filter test."""

    participant_entries = entries_for_participant(manifest, participant_id)
    return tuple(entry for entry in participant_entries if entry.stream_name == stream_name)


def entries_for_device(manifest: Manifest, device_id: str) -> tuple[ManifestEntry, ...]:
    """Device-scoped equivalent of `entries_for_participant`, for
    streams like environment.device where `participant_id is None`.
    Kept as a clearly separate function/API surface — device identity
    and participant identity are never conflated."""

    if not device_id or not device_id.strip():
        raise InvalidParticipantIdError(f"device_id must be a non-empty string, got {device_id!r}")

    return tuple(
        entry
        for entry in manifest.entries
        if entry.participant_id is None and f"/{device_id}/" in f"/{entry.relative_path}"
    )


def load_participant_records(
    root: Path, manifest: Manifest, participant_id: str
) -> list[CanonicalRecord]:
    """Load and canonicalize every record belonging to `participant_id`,
    across every stream that has data for them.

    Only opens files whose manifest entry is already scoped to this
    participant (layer 1), AND, after canonicalizing each file, keeps
    only the individual records whose own participant_id field actually
    matches (layer 2) — so even a mislabeled manifest entry or a
    corrupted file cannot leak another participant's rows through.
    """

    entries = entries_for_participant(manifest, participant_id)

    records: list[CanonicalRecord] = []
    for entry in entries:
        file_records = adapt_file(
            root / entry.relative_path, entry.stream_name, entry.relative_path
        )
        records.extend(
            record
            for record in file_records
            if getattr(record, "participant_id", None) == participant_id
        )

    return records


def manifest_participant_ids(manifest: Manifest) -> frozenset[str]:
    """The full set of distinct participant IDs present in a manifest.

    This is a deliberately GLOBAL, admin-style operation — separate from
    every participant-scoped function above. Per spec Section 8: "the
    participant index must never expose a global file list through a
    participant-scoped API." This function exists precisely so that
    global enumeration has its own explicit, clearly-named entry point,
    rather than being reachable by accident through a participant-scoped
    call (e.g. by passing a wildcard or empty filter into
    `entries_for_participant`, which is explicitly rejected above).
    """

    return frozenset(
        entry.participant_id for entry in manifest.entries if entry.participant_id is not None
    )
