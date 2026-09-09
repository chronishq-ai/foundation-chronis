"""S2.4 — Feature-store schema migrations.

Fixes the audit-flagged gap: the original feature store was a
prototype table with no migrations, no versioning, no provenance/
session/quality tables, and no deletion audit log.

Migrations are plain (up_sql, down_sql) pairs applied in order,
tracked in a `schema_migrations` table so `apply()`/`rollback()` always
know exactly which version is currently applied -- this is what makes
the up -> down -> up round trip in T1 provably clean rather than
assumed.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    up_sql: str
    down_sql: str


MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        version=1,
        name="create_feature_values",
        up_sql="""
            CREATE TABLE feature_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                value REAL NOT NULL,
                feature_version TEXT NOT NULL,
                source TEXT
            );
            CREATE INDEX idx_feature_values_user_id ON feature_values(user_id);
        """,
        down_sql="DROP TABLE feature_values;",
    ),
    Migration(
        version=2,
        name="create_feature_versions",
        up_sql="""
            CREATE TABLE feature_versions (
                feature_name TEXT NOT NULL,
                version TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (feature_name, version)
            );
        """,
        down_sql="DROP TABLE feature_versions;",
    ),
    Migration(
        version=3,
        name="create_sessions",
        up_sql="""
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT
            );
            CREATE INDEX idx_sessions_user_id ON sessions(user_id);
        """,
        down_sql="DROP TABLE sessions;",
    ),
    Migration(
        version=4,
        name="create_provenance",
        up_sql="""
            CREATE TABLE provenance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_value_id INTEGER NOT NULL,
                loader_name TEXT NOT NULL,
                source_file TEXT,
                ingested_at TEXT NOT NULL,
                FOREIGN KEY (feature_value_id) REFERENCES feature_values(id)
            );
        """,
        down_sql="DROP TABLE provenance;",
    ),
    Migration(
        version=5,
        name="create_quality",
        up_sql="""
            CREATE TABLE quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_value_id INTEGER NOT NULL,
                quality_flag TEXT NOT NULL,
                quality_reason TEXT,
                FOREIGN KEY (feature_value_id) REFERENCES feature_values(id)
            );
        """,
        down_sql="DROP TABLE quality;",
    ),
    Migration(
        version=6,
        name="create_deletion_audit_log",
        up_sql="""
            CREATE TABLE deletion_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deleted_at TEXT NOT NULL,
                actor TEXT NOT NULL,
                user_id TEXT NOT NULL,
                record_count INTEGER NOT NULL,
                record_snapshot TEXT NOT NULL
            );
        """,
        down_sql="DROP TABLE deletion_audit_log;",
    ),
)


def _ensure_migration_tracking_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )


def current_version(conn: sqlite3.Connection) -> int:
    _ensure_migration_tracking_table(conn)
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    return row[0] if row[0] is not None else 0


def migrate_up(conn: sqlite3.Connection, *, target_version: int | None = None) -> None:
    """Apply every migration with version > current, up to and
    including `target_version` (or all of them, if None)."""

    _ensure_migration_tracking_table(conn)
    applied = current_version(conn)
    ceiling = target_version if target_version is not None else MIGRATIONS[-1].version

    for migration in MIGRATIONS:
        if applied < migration.version <= ceiling:
            conn.executescript(migration.up_sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (?, ?)",
                (migration.version, migration.name),
            )

    conn.commit()


def migrate_down(conn: sqlite3.Connection, *, target_version: int = 0) -> None:
    """Roll back every migration with version > target_version, in
    reverse order."""

    _ensure_migration_tracking_table(conn)
    applied = current_version(conn)

    for migration in reversed(MIGRATIONS):
        if target_version < migration.version <= applied:
            conn.executescript(migration.down_sql)
            conn.execute("DELETE FROM schema_migrations WHERE version = ?", (migration.version,))

    conn.commit()


def get_table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {row[0] for row in rows}
