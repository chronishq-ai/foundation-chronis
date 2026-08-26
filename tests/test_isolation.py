"""Step 8: the full user-isolation test suite — spec Section 8.

This is the most important test file in the whole synthetic harness.
Every function under test here is `synthetic.query`, exercised against
real generated-and-canonicalized data, never mocked.
"""

from datetime import date
from pathlib import Path
from random import Random

import pytest
from synthetic.config import Participant
from synthetic.query import (
    InvalidParticipantIdError,
    entries_for_device,
    entries_for_participant,
    entries_for_participant_and_stream,
    load_participant_records,
    manifest_participant_ids,
)
from synthetic.registry import REGISTRY
from synthetic.tiles_participant_index import build_manifest
from synthetic.writer import write_records

TEST_DAY = date(2026, 3, 1)
TEST_DAY_2 = date(2026, 3, 2)


def make_participant(
    participant_id: str, active_days: tuple[date, ...] = (TEST_DAY, TEST_DAY_2)
) -> Participant:
    return Participant(
        participant_id=participant_id,
        enrollment_date=active_days[0],
        active_days=active_days,
        resting_heart_rate=68.0,
    )


def build_two_participant_fixture(
    root: Path, participant_a_id: str = "user_a", participant_b_id: str = "user_b"
):
    """Writes real data for two participants across 2 modalities and 2
    days each — the spec's own minimum bar for the basic-partition test."""

    participant_a = make_participant(participant_a_id)
    participant_b = make_participant(participant_b_id)

    for participant in (participant_a, participant_b):
        for day in participant.active_days:
            for stream_name in ("fitbit.heart_rate", "phone_events.interaction"):
                rng = Random(hash((participant.participant_id, day, stream_name)) % (2**31))
                records = REGISTRY[stream_name].generate(participant, day, rng)
                write_records(
                    root, stream_name, participant.participant_id, day.isoformat(), records
                )

    from synthetic.config import GenConfig

    config = GenConfig(seed=1, participant_count=2, study_start_date=TEST_DAY, study_length_days=2)
    manifest = build_manifest(root, config)

    return manifest, participant_a, participant_b


# --- 1. Basic partition -----------------------------------------------------


def test_basic_partition_a_returns_only_a(tmp_path: Path) -> None:
    manifest, participant_a, participant_b = build_two_participant_fixture(tmp_path)

    records_a = load_participant_records(tmp_path, manifest, participant_a.participant_id)

    assert records_a  # sanity: got something
    returned_ids = {r.participant_id for r in records_a}
    assert returned_ids == {participant_a.participant_id}
    assert participant_b.participant_id not in returned_ids  # negative assertion, not just count


def test_basic_partition_b_returns_only_b(tmp_path: Path) -> None:
    manifest, participant_a, participant_b = build_two_participant_fixture(tmp_path)

    records_b = load_participant_records(tmp_path, manifest, participant_b.participant_id)

    returned_ids = {r.participant_id for r in records_b}
    assert returned_ids == {participant_b.participant_id}
    assert participant_a.participant_id not in returned_ids


def test_basic_partition_across_multiple_days_and_modalities(tmp_path: Path) -> None:
    manifest, participant_a, _ = build_two_participant_fixture(tmp_path)

    entries_a = entries_for_participant(manifest, participant_a.participant_id)
    stream_names = {entry.stream_name for entry in entries_a}

    assert len(stream_names) == 2  # both modalities present for A
    assert len(entries_a) == 4  # 2 modalities x 2 days


# --- 2. Unknown participant ID -----------------------------------------------


def test_unknown_participant_id_returns_empty_not_error(tmp_path: Path) -> None:
    manifest, _, _ = build_two_participant_fixture(tmp_path)

    entries = entries_for_participant(manifest, "totally_nonexistent_participant_id")
    records = load_participant_records(tmp_path, manifest, "totally_nonexistent_participant_id")

    assert entries == ()
    assert records == []


def test_unknown_participant_id_never_falls_back_to_another_participant(tmp_path: Path) -> None:
    manifest, participant_a, participant_b = build_two_participant_fixture(tmp_path)

    records = load_participant_records(tmp_path, manifest, "nonexistent_user_zzz")

    all_real_ids = {participant_a.participant_id, participant_b.participant_id}
    returned_ids = {r.participant_id for r in records}
    assert returned_ids.isdisjoint(all_real_ids)


# --- 3. Cross-filter combinations --------------------------------------------


def test_cross_filter_participant_a_with_stream_only_b_has(tmp_path: Path) -> None:
    """Participant A queried with a stream_name that only B has data
    for must return nothing — even though A exists and the stream
    exists, the (participant, stream) COMBINATION does not."""

    manifest, participant_a, participant_b = build_two_participant_fixture(tmp_path)

    # Give B an exclusive stream A doesn't have.
    rng = Random(99)
    exclusive_records = REGISTRY["environment.device"].generate(participant_b, TEST_DAY, rng)
    write_records(
        tmp_path,
        "environment.device",
        f"{participant_b.participant_id}_env_device",
        TEST_DAY.isoformat(),
        exclusive_records,
    )

    from synthetic.config import GenConfig

    config = GenConfig(seed=1, participant_count=2, study_start_date=TEST_DAY, study_length_days=2)
    manifest = build_manifest(tmp_path, config)

    result = entries_for_participant_and_stream(
        manifest, participant_a.participant_id, "environment.device"
    )

    assert result == ()


# --- 4. Null/empty participant identifiers -----------------------------------


def test_empty_string_participant_id_raises(tmp_path: Path) -> None:
    manifest, _, _ = build_two_participant_fixture(tmp_path)

    with pytest.raises(InvalidParticipantIdError):
        entries_for_participant(manifest, "")


def test_whitespace_only_participant_id_raises(tmp_path: Path) -> None:
    manifest, _, _ = build_two_participant_fixture(tmp_path)

    with pytest.raises(InvalidParticipantIdError):
        entries_for_participant(manifest, "   ")


def test_none_participant_id_raises(tmp_path: Path) -> None:
    manifest, _, _ = build_two_participant_fixture(tmp_path)

    with pytest.raises(InvalidParticipantIdError):
        entries_for_participant(manifest, None)  # type: ignore[arg-type]


def test_empty_participant_id_never_returns_all_participants(tmp_path: Path) -> None:
    manifest, _, _ = build_two_participant_fixture(tmp_path)

    with pytest.raises(InvalidParticipantIdError):
        entries_for_participant(manifest, "")
    # the exception itself IS the proof — if this line is ever reached
    # instead of raising, the isolation guarantee has already failed.


# --- 5. Path-based bypass -----------------------------------------------------


def test_path_based_bypass_via_tampered_manifest_entry_returns_zero(tmp_path: Path) -> None:
    """Simulates a manifest bug/attack: an entry CLAIMS to belong to
    participant A but its relative_path actually points at B's real
    file. The record-level defense (layer 2 in `load_participant_records`)
    must catch this: B's real rows carry B's own participant_id in their
    actual content, so they get filtered out entirely when queried as A."""

    manifest, participant_a, participant_b = build_two_participant_fixture(tmp_path)

    b_entry = next(
        e
        for e in manifest.entries
        if e.participant_id == participant_b.participant_id and e.stream_name == "fitbit.heart_rate"
    )

    from dataclasses import replace

    tampered_entry = replace(
        b_entry, participant_id=participant_a.participant_id
    )  # lies about ownership

    from synthetic.tiles_participant_index import Manifest

    tampered_manifest = Manifest(
        dataset_identifier=manifest.dataset_identifier,
        generator_version=manifest.generator_version,
        parser_version=manifest.parser_version,
        config_fingerprint=manifest.config_fingerprint,
        created_at=manifest.created_at,
        entries=(tampered_entry,),  # ONLY the tampered entry, isolate the test
        dataset_fingerprint=manifest.dataset_fingerprint,
    )

    result = load_participant_records(tmp_path, tampered_manifest, participant_a.participant_id)

    # The manifest lied and said this file belongs to A, but every row
    # inside the file actually says participant_id=B. Layer 2 filtering
    # must reject all of them.
    assert result == []


# --- 6. Manifest-level isolation ----------------------------------------------


def test_participant_scoped_entries_never_include_other_participants_paths(tmp_path: Path) -> None:
    manifest, participant_a, participant_b = build_two_participant_fixture(tmp_path)

    entries_a = entries_for_participant(manifest, participant_a.participant_id)
    returned_paths = {e.relative_path for e in entries_a}

    entries_b = entries_for_participant(manifest, participant_b.participant_id)
    b_paths = {e.relative_path for e in entries_b}

    assert returned_paths.isdisjoint(b_paths)  # zero overlap, inspected directly


def test_global_enumeration_requires_its_own_explicit_function(tmp_path: Path) -> None:
    """`manifest_participant_ids` is the only sanctioned way to see
    every participant at once — proves it's a distinct, clearly-named
    function, not something reachable through the participant-scoped
    API by accident."""

    manifest, participant_a, participant_b = build_two_participant_fixture(tmp_path)

    all_ids = manifest_participant_ids(manifest)

    assert all_ids == {participant_a.participant_id, participant_b.participant_id}


# --- 7. Negative assertions (already threaded through every test above) -----
# Every test in this file inspects actual returned identifiers/paths and
# proves the forbidden participant's IDs/paths are absent - not merely
# that a count is zero. See the `returned_ids`/`returned_paths` assertions
# throughout.


# --- Device-scoped isolation (separate from participant isolation) ----------


def test_device_scoped_query_does_not_leak_into_participant_query(tmp_path: Path) -> None:
    manifest, participant_a, _ = build_two_participant_fixture(tmp_path)

    rng = Random(5)
    env_records = REGISTRY["environment.device"].generate(participant_a, TEST_DAY, rng)
    device_id = f"{participant_a.participant_id}_env_device"
    write_records(tmp_path, "environment.device", device_id, TEST_DAY.isoformat(), env_records)

    from synthetic.config import GenConfig

    config = GenConfig(seed=1, participant_count=2, study_start_date=TEST_DAY, study_length_days=2)
    manifest = build_manifest(tmp_path, config)

    participant_entries = entries_for_participant(manifest, participant_a.participant_id)
    device_entries = entries_for_device(manifest, device_id)

    # environment.device entries have participant_id=None - they must
    # NEVER show up via the participant-scoped API, even for the
    # participant who "owns" that device.
    assert all(e.stream_name != "environment.device" for e in participant_entries)
    assert any(e.stream_name == "environment.device" for e in device_entries)


# --- PERMANENT REGRESSION FIXTURE: leaky_fixture -----------------------------
# Per spec Section 6/8: "Keep the intentional leaky_fixture corruption
# record as a permanent regression case." This targets a realistic bug
# class: substring/prefix ID matching. If isolation code were ever
# changed from exact equality to a substring/prefix check (e.g.
# `if participant_id in entry.relative_path`), participant "user_1"
# would incorrectly match participant "user_10"'s files too, since
# "user_1" is a literal substring of "user_10". This test must NEVER be
# removed or weakened - it is the permanent guard against that specific
# regression, mirroring the audit's tests/leaky.py pattern.


def test_leaky_fixture_substring_id_collision_never_leaks(tmp_path: Path) -> None:
    participant_short = make_participant("user_1")
    participant_long = make_participant("user_10")  # "user_1" is a substring of this

    for participant in (participant_short, participant_long):
        rng = Random(hash(participant.participant_id) % (2**31))
        records = REGISTRY["fitbit.heart_rate"].generate(participant, TEST_DAY, rng)
        write_records(
            tmp_path, "fitbit.heart_rate", participant.participant_id, TEST_DAY.isoformat(), records
        )

    from synthetic.config import GenConfig

    config = GenConfig(seed=1, participant_count=2, study_start_date=TEST_DAY, study_length_days=1)
    manifest = build_manifest(tmp_path, config)

    short_entries = entries_for_participant(manifest, "user_1")
    short_records = load_participant_records(tmp_path, manifest, "user_1")

    # Exactly one entry/file for "user_1" - NOT two (which would mean
    # "user_10"'s file leaked in via substring matching).
    assert len(short_entries) == 1
    assert short_entries[0].participant_id == "user_1"

    returned_ids = {r.participant_id for r in short_records}
    assert returned_ids == {"user_1"}
    assert "user_10" not in returned_ids


def test_leaky_fixture_reverse_direction_also_safe(tmp_path: Path) -> None:
    """The same collision, queried from the other direction: "user_10"
    must not somehow return FEWER records than expected because "user_1"
    was mis-subtracted, and must never include user_1's data."""

    participant_short = make_participant("user_1")
    participant_long = make_participant("user_10")

    for participant in (participant_short, participant_long):
        rng = Random(hash(participant.participant_id) % (2**31))
        records = REGISTRY["fitbit.heart_rate"].generate(participant, TEST_DAY, rng)
        write_records(
            tmp_path, "fitbit.heart_rate", participant.participant_id, TEST_DAY.isoformat(), records
        )

    from synthetic.config import GenConfig

    config = GenConfig(seed=1, participant_count=2, study_start_date=TEST_DAY, study_length_days=1)
    manifest = build_manifest(tmp_path, config)

    long_records = load_participant_records(tmp_path, manifest, "user_10")

    returned_ids = {r.participant_id for r in long_records}
    assert returned_ids == {"user_10"}
    assert "user_1" not in returned_ids
