"""Writes `_generator_manifest.json` at the root of a generation run.

This is DISTINCT from `tiles_participant_index.Manifest`:
  - `_generator_manifest.json` (this module) is written ONCE by the
    generator itself, at generation time, recording the seed/config/
    version used to produce this specific run's output (spec Section
    4.1: "Self-describing. Every generation run writes a
    _generator_manifest.json...").
  - `tiles_participant_index.Manifest` is built LATER, by scanning
    whatever files actually exist on disk, and is used for
    reproducibility/cache-invalidation of the discovery/indexing
    process itself (spec Section 7.2).

Both are needed; they answer different questions ("what config produced
this data?" vs. "what files currently exist and are they still valid?").
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from synthetic.config import GenConfig

GENERATOR_MANIFEST_FILENAME = "_generator_manifest.json"


def write_generation_manifest(root: Path, config: GenConfig) -> Path:
    """Write `_generator_manifest.json` at the root of a generation run."""

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / GENERATOR_MANIFEST_FILENAME

    payload = {
        "generator_version": config.generator_version,
        "config": config.to_dict(),
        "generated_at": datetime.now(UTC).isoformat(),
    }

    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return manifest_path


def read_generation_manifest(root: Path) -> dict[str, object]:
    """Read back a previously-written `_generator_manifest.json`."""

    manifest_path = root / GENERATOR_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"no {GENERATOR_MANIFEST_FILENAME} found under {root}")

    parsed: dict[str, object] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return parsed
