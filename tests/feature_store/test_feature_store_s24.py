"""S2.4 test sheet - original 4 test cases, plus a full cross-user
isolation sweep across every exposed query path (reusing Sprint 13's
tests/leaky.py pattern, per the ticket's explicit instruction).

Test Sheet - S2.4:
  T1: Fresh DB, migration up then down -> Clean round-trip, no data
      loss
  T2: Two users' records seeded, cross-user read via every exposed
      query path -> Each returns empty or raises
  T3: Any hard delete -> Deletion audit log has timestamp, actor, and
      record count
  T4: A feature whose definition changes -> old and new versions
      inserted -> old records keep the old feature_version tag
"""

from __future__ import annotations

import sqlite3

import pytest

from chronis_ml.feature_store.migrations import (
    MIGRATIONS,
    current_version,
    get_table_names,
    migrate_down,
    migrate_up,
)
from chronis_ml.feature_store.store import FeatureStore, FeatureStoreError


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    yield connection
    connection.close()


# --- T1: fresh DB, migration up -> down -> up, clean round trip -------------


def test_s24_t1_migration_up_creates_all_tables(conn) -> None:
    migrate_up(conn)

    tables = get_table_names(conn)
    expected = {
        "feature_values",
        "feature_versions",
        "sessions",
        "provenance",
        "quality",
        "deletion_audit_log",
        "schema_migrations",
    }
    assert expected <= tables
    assert current_version(conn) == MIGRATIONS[-1].version


def test_s24_t1_migration_down_removes_all_tables_cleanly() -> None:
    conn = sqlite3.connect(":memory:")
    migrate_up(conn)
    migrate_down(conn)

    tables = get_table_names(conn)
    # Only the tracking table itself may remain (it's not one of the
    # migrations, so migrate_down never touches it).
    assert tables <= {"schema_migrations"}
    assert current_version(conn) == 0


def test_s24_t1_up_down_up_round_trip_is_clean_and_repeatable() -> None:
    """The actual round-trip test: up -> down -> up must produce an
    IDENTICAL, fully-working schema each time, with no errors and no
    leftover artifacts from the first pass."""

    conn = sqlite3.connect(":memory:")

    migrate_up(conn)
    tables_after_first_up = get_table_names(conn)

    migrate_down(conn)

    migrate_up(conn)  # must not raise (e.g. "table already exists")
    tables_after_second_up = get_table_names(conn)

    assert tables_after_first_up == tables_after_second_up
    assert current_version(conn) == MIGRATIONS[-1].version

    # Prove the schema is genuinely usable after the round trip, not
    # just structurally present.
    store = FeatureStore(conn)
    record_id = store.insert_feature_value(
        user_id="user_001",
        timestamp="2026-08-16T12:00:00+00:00",
        feature_name="heart_rate",
        value=70.0,
        feature_version="1.0",
    )
    assert record_id is not None


def test_s24_t1_partial_migrate_up_to_specific_version(conn) -> None:
    migrate_up(conn, target_version=1)

    assert get_table_names(conn) >= {"feature_values", "schema_migrations"}
    assert "feature_versions" not in get_table_names(conn)
    assert current_version(conn) == 1


# --- T2: cross-user isolation across EVERY exposed query path ---------------


@pytest.fixture
def seeded_store(conn):
    migrate_up(conn)
    store = FeatureStore(conn)

    id_a = store.insert_feature_value(
        user_id="user_a",
        timestamp="2026-08-16T12:00:00+00:00",
        feature_name="heart_rate",
        value=70.0,
        feature_version="1.0",
    )
    id_b = store.insert_feature_value(
        user_id="user_b",
        timestamp="2026-08-16T12:00:00+00:00",
        feature_name="heart_rate",
        value=72.0,
        feature_version="1.0",
    )

    return store, id_a, id_b


def test_s24_t2_query_by_user_never_returns_other_users_records(seeded_store) -> None:
    store, id_a, id_b = seeded_store

    result_a = store.query_by_user("user_a")
    result_b = store.query_by_user("user_b")

    assert all(r.user_id == "user_a" for r in result_a)
    assert all(r.user_id == "user_b" for r in result_b)
    assert not any(r.id == id_b for r in result_a)
    assert not any(r.id == id_a for r in result_b)


def test_s24_t2_query_by_user_and_feature_never_leaks(seeded_store) -> None:
    store, id_a, id_b = seeded_store

    result = store.query_by_user_and_feature("user_a", "heart_rate")

    assert all(r.user_id == "user_a" for r in result)
    assert not any(r.id == id_b for r in result)


def test_s24_t2_get_by_id_for_user_rejects_wrong_users_record_id(seeded_store) -> None:
    """The leaky.py-style regression: even a DIRECT by-ID lookup, using
    a real, valid record ID that genuinely exists, must return nothing
    when the requesting user_id doesn't match that record's owner."""

    store, id_a, id_b = seeded_store

    # user_b attempts to fetch user_a's real record by its real ID.
    result = store.get_by_id_for_user("user_b", id_a)

    assert result is None  # never returns another user's record


def test_s24_t2_empty_user_id_is_rejected_not_treated_as_wildcard(seeded_store) -> None:
    store, _, _ = seeded_store

    with pytest.raises(FeatureStoreError):
        store.query_by_user("")


def test_s24_t2_unknown_user_id_returns_empty_not_error(seeded_store) -> None:
    store, _, _ = seeded_store

    result = store.query_by_user("totally_nonexistent_user")

    assert result == ()


def test_s24_t2_global_enumeration_never_exposes_per_user_values(seeded_store) -> None:
    """The one sanctioned global function returns only feature NAMES,
    never any user's actual values."""

    store, _, _ = seeded_store

    names = store.list_known_feature_names()

    assert names == ("heart_rate",)
    # Confirm the return type genuinely carries no per-user value data.
    assert all(isinstance(name, str) for name in names)


# --- T3: hard delete -> audit log has timestamp, actor, record count -------


def test_s24_t3_hard_delete_logs_timestamp_actor_and_count(seeded_store) -> None:
    store, id_a, id_b = seeded_store

    deleted_count = store.delete_feature_values_for_user("user_a", actor="admin_jane")

    assert deleted_count == 1

    audit_entries = store.get_deletion_audit_log(user_id="user_a")
    assert len(audit_entries) == 1

    entry = audit_entries[0]
    assert entry.actor == "admin_jane"
    assert entry.record_count == 1
    assert entry.user_id == "user_a"
    assert entry.deleted_at  # a real timestamp string is present


def test_s24_t3_hard_delete_actually_removes_the_records(seeded_store) -> None:
    store, id_a, id_b = seeded_store

    store.delete_feature_values_for_user("user_a", actor="admin_jane")

    assert store.query_by_user("user_a") == ()
    assert len(store.query_by_user("user_b")) == 1  # other user untouched


def test_s24_t3_delete_snapshot_preserves_what_was_deleted(seeded_store) -> None:
    store, id_a, id_b = seeded_store

    store.delete_feature_values_for_user("user_a", actor="admin_jane")

    entry = store.get_deletion_audit_log(user_id="user_a")[0]
    assert entry.record_snapshot[0]["feature_name"] == "heart_rate"
    assert entry.record_snapshot[0]["value"] == 70.0


def test_s24_t3_empty_actor_is_rejected(seeded_store) -> None:
    store, _, _ = seeded_store

    with pytest.raises(FeatureStoreError):
        store.delete_feature_values_for_user("user_a", actor="")


def test_s24_t3_deleting_a_user_with_no_records_is_a_safe_noop(seeded_store) -> None:
    store, _, _ = seeded_store

    deleted_count = store.delete_feature_values_for_user("nonexistent_user", actor="admin_jane")

    assert deleted_count == 0
    assert store.get_deletion_audit_log(user_id="nonexistent_user") == ()


# --- T4: feature version change -> old records keep old tag -----------------


def test_s24_t4_old_records_keep_old_feature_version_tag(conn) -> None:
    migrate_up(conn)
    store = FeatureStore(conn)

    store.register_feature_version(feature_name="heart_rate", version="1.0", description="raw bpm")
    old_id = store.insert_feature_value(
        user_id="user_a",
        timestamp="2026-08-16T12:00:00+00:00",
        feature_name="heart_rate",
        value=70.0,
        feature_version="1.0",
    )

    # The feature definition changes -> a new version is registered.
    store.register_feature_version(
        feature_name="heart_rate", version="2.0", description="bpm, recalibrated sensor"
    )
    new_id = store.insert_feature_value(
        user_id="user_a",
        timestamp="2026-08-17T12:00:00+00:00",
        feature_name="heart_rate",
        value=71.0,
        feature_version="2.0",
    )

    all_records = store.query_by_user("user_a")
    old_record = next(r for r in all_records if r.id == old_id)
    new_record = next(r for r in all_records if r.id == new_id)

    assert old_record.feature_version == "1.0"  # never silently rewritten
    assert new_record.feature_version == "2.0"


def test_s24_t4_feature_versions_table_tracks_both_versions(conn) -> None:
    migrate_up(conn)
    store = FeatureStore(conn)

    store.register_feature_version(feature_name="heart_rate", version="1.0")
    store.register_feature_version(feature_name="heart_rate", version="2.0")

    rows = conn.execute(
        "SELECT version FROM feature_versions WHERE feature_name = ? ORDER BY version",
        ("heart_rate",),
    ).fetchall()

    assert [row[0] for row in rows] == ["1.0", "2.0"]
