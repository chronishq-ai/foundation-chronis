"""Tests for synthetic.generation_manifest — the _generator_manifest.json writer."""

import json
from datetime import date
from pathlib import Path

import pytest
from synthetic.config import GenConfig
from synthetic.generation_manifest import (
    GENERATOR_MANIFEST_FILENAME,
    read_generation_manifest,
    write_generation_manifest,
)


def make_config() -> GenConfig:
    return GenConfig(
        seed=42,
        participant_count=5,
        study_start_date=date(2026, 1, 1),
        study_length_days=14,
    )


def test_write_creates_expected_file(tmp_path: Path) -> None:
    config = make_config()

    manifest_path = write_generation_manifest(tmp_path, config)

    assert manifest_path == tmp_path / GENERATOR_MANIFEST_FILENAME
    assert manifest_path.exists()


def test_written_manifest_contains_seed_and_config(tmp_path: Path) -> None:
    config = make_config()

    write_generation_manifest(tmp_path, config)
    data = json.loads((tmp_path / GENERATOR_MANIFEST_FILENAME).read_text())

    assert data["config"]["seed"] == 42
    assert data["config"]["participant_count"] == 5
    assert "generated_at" in data
    assert "generator_version" in data


def test_read_generation_manifest_roundtrip(tmp_path: Path) -> None:
    config = make_config()

    write_generation_manifest(tmp_path, config)
    loaded = read_generation_manifest(tmp_path)

    assert loaded["config"]["seed"] == 42


def test_read_generation_manifest_raises_if_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_generation_manifest(tmp_path)


def test_write_creates_root_if_missing(tmp_path: Path) -> None:
    nested_root = tmp_path / "does" / "not" / "exist" / "yet"
    config = make_config()

    write_generation_manifest(nested_root, config)

    assert nested_root.exists()
    assert (nested_root / GENERATOR_MANIFEST_FILENAME).exists()
