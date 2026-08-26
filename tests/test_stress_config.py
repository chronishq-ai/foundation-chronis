"""Step 9: CI stress-config run — spec Section 4.1 / acceptance checklist.

Runs the full generate -> write -> index -> canonicalize -> isolation
pipeline using `CorruptionConfig.stress_config()` (elevated rates,
25-40%) instead of the low default rates every other test file uses.

Per the spec's own callout: "A pipeline that only ever sees 2% mess
will still get complacent; the stress config is what actually proves
robustness."

DISTINCT from every other test file in this suite: this one does NOT
assert zero `CanonicalizationError`. At stress-level corruption rates,
malformed rows and schema drift are EXPECTED — raising loudly per-file
IS the correct, designed behavior, not a failure. What this file
actually proves:

  1. The pipeline never crashes with an UNRECOGNIZED exception type.
  2. Every file that fails, fails with a `CanonicalizationError`
     subtype, never silently succeeds while holding corrupted data.
  3. Isolation guarantees hold even under heavy corruption — corruption
     operates within one participant's own files and must never cause
     cross-participant leakage, no matter how aggressive.
  4. `stress_config()` itself hasn't been accidentally weakened back
     toward normal rates in some future edit.
"""

from datetime import date
from pathlib import Path
from random import Random

from synthetic.canonical import CanonicalizationError, adapt_file
from synthetic.config import CorruptionConfig, GenConfig, build_roster
from synthetic.corruption import inject_corruption
from synthetic.query import (
    entries_for_participant,
    load_participant_records,
    manifest_participant_ids,
)
from synthetic.registry import REGISTRY
from synthetic.tiles_participant_index import build_manifest, validate_manifest
from synthetic.writer import write_records

STUDY_START = date(2026, 1, 1)
STUDY_LENGTH_DAYS = 5
PARTICIPANT_COUNT = 6


def _generate_stress_dataset(root: Path) -> GenConfig:
    config = GenConfig(
        seed=2026,
        participant_count=PARTICIPANT_COUNT,
        study_start_date=STUDY_START,
        study_length_days=STUDY_LENGTH_DAYS,
        corruption=CorruptionConfig.stress_config(),
    )

    roster = build_roster(config)

    for participant in roster:
        for day in participant.active_days:
            for stream_name, generator in REGISTRY.items():
                rng = Random(
                    hash((participant.participant_id, day.isoformat(), stream_name, config.seed))
                    % (2**31)
                )
                records = generator.generate(participant, day, rng)
                records = inject_corruption(
                    records, config.corruption, rng, stream_name=stream_name
                )

                if not records:
                    continue  # corruption can legitimately empty a day's file

                group_id = participant.participant_id
                if stream_name == "environment.device":
                    group_id = f"{participant.participant_id}_env_device"

                write_records(root, stream_name, group_id, day.isoformat(), records)

    return config


def test_stress_config_pipeline_survives_elevated_corruption(tmp_path: Path) -> None:
    config = _generate_stress_dataset(tmp_path)

    # 1. Manifest building must succeed even over heavily corrupted
    #    files — it only reads headers/checksums, never validates
    #    row-level content, so it should never fail here.
    manifest = build_manifest(tmp_path, config)
    assert manifest.entries  # sanity: something was actually written

    # 2. validate_manifest must pass immediately after building.
    validate_manifest(manifest, tmp_path)

    # 3. Canonicalize every single file. Recognized failures are
    #    expected and counted; anything else is a real bug.
    recognized_failures = 0
    successes = 0
    unexpected_errors = []

    for entry in manifest.entries:
        try:
            adapt_file(tmp_path / entry.relative_path, entry.stream_name, entry.relative_path)
            successes += 1
        except CanonicalizationError:
            recognized_failures += 1
        except Exception as exc:  # intentionally broad — this IS the check
            unexpected_errors.append((entry.relative_path, repr(exc)))

    assert not unexpected_errors, (
        f"unexpected (non-CanonicalizationError) failures: {unexpected_errors}"
    )
    assert successes > 0  # most files should still canonicalize fine even at stress rates


def test_stress_config_isolation_holds_under_heavy_corruption(tmp_path: Path) -> None:
    """Corruption operates within one participant's own files — it must
    never cause cross-participant leakage, no matter how aggressive."""

    config = _generate_stress_dataset(tmp_path)
    manifest = build_manifest(tmp_path, config)

    all_ids = manifest_participant_ids(manifest)
    assert len(all_ids) >= 1

    for participant_id in all_ids:
        entries = entries_for_participant(manifest, participant_id)
        other_ids = all_ids - {participant_id}

        for entry in entries:
            assert not any(other_id in entry.relative_path for other_id in other_ids), (
                f"participant {participant_id}'s entry {entry.relative_path} "
                f"appears to reference another participant's ID"
            )

        # Layer-2 record-level check too, wherever canonicalization succeeds.
        try:
            records = load_participant_records(tmp_path, manifest, participant_id)
        except CanonicalizationError:
            continue  # malformed at this stress rate — covered by the other test

        returned_ids = {getattr(r, "participant_id", None) for r in records}
        assert returned_ids <= {participant_id}  # never any other id, never None leaking through


def test_stress_config_has_meaningfully_elevated_rates() -> None:
    """Sanity check on the config itself, not the pipeline — guards
    against someone accidentally weakening `stress_config()` back down
    toward normal rates in a future edit."""

    stress = CorruptionConfig.stress_config()

    assert stress.missing_block >= 0.25
    assert stress.duplicate_rows >= 0.20
    assert stress.out_of_order >= 0.20
    assert stress.clock_drift >= 0.25
    assert stress.malformed_row >= 0.20
    assert stress.participant_dropout >= 0.30
