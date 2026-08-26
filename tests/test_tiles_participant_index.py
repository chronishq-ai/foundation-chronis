"""Tests for Step 5: synthetic.writer + synthetic.tiles_participant_index."""

import time
from datetime import date
from pathlib import Path
from random import Random

import pytest
from synthetic.config import GenConfig, Participant
from synthetic.registry import REGISTRY
from synthetic.tiles_participant_index import (
    ManifestValidationError,
    build_manifest,
    config_fingerprint,
    discover,
    infer_stream,
    load_cache,
    probe_file,
    save_cache,
    validate_manifest,
)
from synthetic.writer import write_records

TEST_DAY = date(2026, 3, 1)


def make_participant() -> Participant:
    return Participant(
        participant_id="synthetic_p0001",
        enrollment_date=TEST_DAY,
        active_days=(TEST_DAY,),
        resting_heart_rate=68.0,
    )


def write_sample_dataset(root: Path, *, seed: int = 1) -> None:
    """Write a small real dataset to disk using 3 different streams,
    covering participant-scoped and device-scoped granularity."""

    participant = make_participant()
    rng = Random(seed)

    hr_records = REGISTRY["fitbit.heart_rate"].generate(participant, TEST_DAY, rng)
    write_records(
        root, "fitbit.heart_rate", participant.participant_id, TEST_DAY.isoformat(), hr_records
    )

    steps_records = REGISTRY["fitbit.steps"].generate(participant, TEST_DAY, rng)
    write_records(
        root, "fitbit.steps", participant.participant_id, TEST_DAY.isoformat(), steps_records
    )

    env_records = REGISTRY["environment.device"].generate(participant, TEST_DAY, rng)
    device_id = f"{participant.participant_id}_env_device"
    write_records(root, "environment.device", device_id, TEST_DAY.isoformat(), env_records)


def make_config() -> GenConfig:
    return GenConfig(
        seed=1,
        participant_count=1,
        study_start_date=TEST_DAY,
        study_length_days=1,
    )


# --- writer -----------------------------------------------------------------


def test_write_records_creates_expected_path(tmp_path: Path) -> None:
    participant = make_participant()
    records = REGISTRY["fitbit.heart_rate"].generate(participant, TEST_DAY, Random(1))

    file_path = write_records(
        tmp_path, "fitbit.heart_rate", participant.participant_id, TEST_DAY.isoformat(), records
    )

    assert file_path.exists()
    assert file_path == tmp_path / "fitbit" / "heart_rate" / "synthetic_p0001" / "2026-03-01.csv"


def test_write_records_rejects_unknown_stream(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown stream_name"):
        write_records(tmp_path, "not_a_real_stream", "p1", "2026-01-01", [])


def test_write_records_handles_heterogeneous_columns(tmp_path: Path) -> None:
    """Records with different keys (e.g. after malformed_row corruption)
    must not crash the writer or silently lose columns."""

    records = [
        {"timestamp": "2026-01-01T00:00:00+00:00", "participant_id": "p1", "value": 1.0},
        {"timestamp": "2026-01-01T00:01:00+00:00", "participant_id": "p1", "extra_field": "x"},
    ]

    file_path = write_records(tmp_path, "fitbit.heart_rate", "p1", "2026-01-01", records)

    content = file_path.read_text()
    assert "value" in content
    assert "extra_field" in content


# --- discover -----------------------------------------------------------


def test_discover_finds_all_written_files(tmp_path: Path) -> None:
    write_sample_dataset(tmp_path)

    files = discover(tmp_path)

    assert len(files) == 3


def test_discover_raises_on_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover(tmp_path / "does_not_exist")


def test_discover_is_deterministically_ordered(tmp_path: Path) -> None:
    write_sample_dataset(tmp_path)

    files_a = discover(tmp_path)
    files_b = discover(tmp_path)

    assert [f.relative_path for f in files_a] == [f.relative_path for f in files_b]


# --- probe_file ---------------------------------------------------------


def test_probe_file_reports_columns_and_sample(tmp_path: Path) -> None:
    write_sample_dataset(tmp_path)
    hr_file = tmp_path / "fitbit" / "heart_rate" / "synthetic_p0001" / "2026-03-01.csv"

    result = probe_file(hr_file, sample_size=3)

    assert "timestamp" in result.columns
    assert "heart_rate_bpm" in result.columns
    assert len(result.sample_rows) <= 3


# --- infer_stream ---------------------------------------------------------


def test_infer_stream_maps_known_paths() -> None:
    assert infer_stream("fitbit/heart_rate/p1/2026-01-01.csv") == "fitbit.heart_rate"
    assert infer_stream("environment/device1/2026-01-01.csv") == "environment.device"
    assert (
        infer_stream("proximity/participant_beacon/p1/2026-01-01.csv")
        == "proximity.participant_beacon"
    )


def test_infer_stream_rejects_unrecognized_path() -> None:
    with pytest.raises(ManifestValidationError, match="unrecognized modality"):
        infer_stream("totally_unknown_folder/p1/2026-01-01.csv")


# --- build_manifest ---------------------------------------------------------


def test_build_manifest_produces_one_entry_per_file(tmp_path: Path) -> None:
    write_sample_dataset(tmp_path)
    config = make_config()

    manifest = build_manifest(tmp_path, config)

    assert len(manifest.entries) == 3


def test_build_manifest_entries_have_correct_granularity_and_participant_id(tmp_path: Path) -> None:
    write_sample_dataset(tmp_path)
    config = make_config()

    manifest = build_manifest(tmp_path, config)
    by_stream = {entry.stream_name: entry for entry in manifest.entries}

    assert by_stream["fitbit.heart_rate"].granularity == "participant"
    assert by_stream["fitbit.heart_rate"].participant_id == "synthetic_p0001"

    assert by_stream["environment.device"].granularity == "device"
    assert (
        by_stream["environment.device"].participant_id is None
    )  # device-scoped, not participant-scoped


def test_build_manifest_detects_time_coverage(tmp_path: Path) -> None:
    write_sample_dataset(tmp_path)
    config = make_config()

    manifest = build_manifest(tmp_path, config)
    hr_entry = next(e for e in manifest.entries if e.stream_name == "fitbit.heart_rate")

    assert hr_entry.time_coverage_start is not None
    assert hr_entry.time_coverage_end is not None
    assert hr_entry.time_coverage_start <= hr_entry.time_coverage_end


def test_build_manifest_dataset_fingerprint_is_stable_for_unchanged_data(tmp_path: Path) -> None:
    write_sample_dataset(tmp_path)
    config = make_config()

    manifest_a = build_manifest(tmp_path, config)
    manifest_b = build_manifest(tmp_path, config)

    assert manifest_a.dataset_fingerprint == manifest_b.dataset_fingerprint


def test_build_manifest_fingerprint_changes_when_a_file_changes(tmp_path: Path) -> None:
    write_sample_dataset(tmp_path)
    config = make_config()

    manifest_before = build_manifest(tmp_path, config)

    hr_file = tmp_path / "fitbit" / "heart_rate" / "synthetic_p0001" / "2026-03-01.csv"
    time.sleep(0.01)  # ensure mtime actually advances on fast filesystems
    hr_file.write_text(hr_file.read_text() + "\n")  # trivial content change

    manifest_after = build_manifest(tmp_path, config)

    assert manifest_before.dataset_fingerprint != manifest_after.dataset_fingerprint


# --- validate_manifest -------------------------------------------------------


def test_validate_manifest_passes_for_unchanged_dataset(tmp_path: Path) -> None:
    write_sample_dataset(tmp_path)
    manifest = build_manifest(tmp_path, make_config())

    validate_manifest(manifest, tmp_path)  # should not raise


def test_validate_manifest_rejects_modified_source_file(tmp_path: Path) -> None:
    """Spec acceptance item 6: "Cache invalidation is proven with a
    modified-source test." """

    write_sample_dataset(tmp_path)
    manifest = build_manifest(tmp_path, make_config())

    hr_file = tmp_path / "fitbit" / "heart_rate" / "synthetic_p0001" / "2026-03-01.csv"
    time.sleep(0.01)
    hr_file.write_text(hr_file.read_text() + "\n")

    with pytest.raises(ManifestValidationError, match="changed since manifest"):
        validate_manifest(manifest, tmp_path)


def test_validate_manifest_rejects_missing_file(tmp_path: Path) -> None:
    write_sample_dataset(tmp_path)
    manifest = build_manifest(tmp_path, make_config())

    hr_file = tmp_path / "fitbit" / "heart_rate" / "synthetic_p0001" / "2026-03-01.csv"
    hr_file.unlink()

    with pytest.raises(ManifestValidationError, match="missing"):
        validate_manifest(manifest, tmp_path)


def test_validate_manifest_rejects_untracked_new_file(tmp_path: Path) -> None:
    write_sample_dataset(tmp_path)
    manifest = build_manifest(tmp_path, make_config())

    participant = make_participant()
    extra_records = REGISTRY["fitbit.sleep"].generate(participant, TEST_DAY, Random(9))
    write_records(
        tmp_path, "fitbit.sleep", participant.participant_id, TEST_DAY.isoformat(), extra_records
    )

    with pytest.raises(ManifestValidationError, match="not covered by the manifest"):
        validate_manifest(manifest, tmp_path)


# --- cache ----------------------------------------------------------------


def test_cache_roundtrip(tmp_path: Path) -> None:
    write_sample_dataset(tmp_path)
    config = make_config()
    manifest = build_manifest(tmp_path, config)
    cache_path = tmp_path / "_cache" / "manifest.json"

    save_cache(manifest, cache_path)
    loaded = load_cache(
        cache_path,
        expected_config_fingerprint=config_fingerprint(config),
        expected_generator_version=config.generator_version,
    )

    assert loaded is not None
    assert loaded.dataset_fingerprint == manifest.dataset_fingerprint
    assert len(loaded.entries) == len(manifest.entries)


def test_load_cache_returns_none_for_missing_file(tmp_path: Path) -> None:
    result = load_cache(
        tmp_path / "nonexistent.json",
        expected_config_fingerprint="anything",
        expected_generator_version="anything",
    )

    assert result is None


def test_load_cache_returns_none_on_config_fingerprint_mismatch(tmp_path: Path) -> None:
    write_sample_dataset(tmp_path)
    config = make_config()
    manifest = build_manifest(tmp_path, config)
    cache_path = tmp_path / "_cache" / "manifest.json"
    save_cache(manifest, cache_path)

    result = load_cache(
        cache_path,
        expected_config_fingerprint="a_totally_different_fingerprint",
        expected_generator_version=config.generator_version,
    )

    assert result is None


def test_load_cache_returns_none_on_generator_version_mismatch(tmp_path: Path) -> None:
    write_sample_dataset(tmp_path)
    config = make_config()
    manifest = build_manifest(tmp_path, config)
    cache_path = tmp_path / "_cache" / "manifest.json"
    save_cache(manifest, cache_path)

    result = load_cache(
        cache_path,
        expected_config_fingerprint=config_fingerprint(config),
        expected_generator_version="99.0.0-does-not-exist",
    )

    assert result is None
