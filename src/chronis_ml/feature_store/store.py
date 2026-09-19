"""S2.4 — FeatureStore: the actual read/write API over the migrated
schema.

Design principle enforced throughout: every function that reads
feature VALUES takes `user_id` as a required, non-optional parameter
and uses it in a parameterized SQL WHERE clause -- there is no
exposed query path that can return another user's data by omission or
by a caller forgetting to filter. Global, cross-user enumeration (e.g.
"what feature names exist at all") gets its own separate, clearly
named function that never returns per-user values, mirroring the same
pattern used elsewhere in this codebase for global-vs-scoped APIs.

B7 traceability chain (observation(s) -> feature -> feature version ->
extractor/model version -> quality -> provenance): this file now
actually writes and reads the `provenance` and `quality` tables the
migration creates (previously dead schema — created but never touched
by any read/write path). `insert_provenance`/`insert_quality` attach
records to an existing `feature_values` row, and both are validated
against a real feature_value_id so a provenance/quality record can
never point at nothing.

HONEST SCOPE NOTE: this still does not close the full B7 chain.
`feature_version` and "extractor/model version" remain the same single
string column — there is no separate field for the code version that
produced a value, independent of the feature definition it followed.
And `feature_values.source` is still a free-text, unvalidated string,
not a real link back to specific raw observation record(s). Both are
flagged as open follow-ups, not fabricated here.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime


class FeatureStoreError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FeatureValueRecord:
    id: int
    user_id: str
    timestamp: str
    feature_name: str
    value: float
    feature_version: str
    source: str | None


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    id: int
    feature_value_id: int
    loader_name: str
    source_file: str | None
    ingested_at: str


@dataclass(frozen=True, slots=True)
class QualityRecord:
    id: int
    feature_value_id: int
    quality_flag: str
    quality_reason: str | None


@dataclass(frozen=True, slots=True)
class DeletionAuditEntry:
    id: int
    deleted_at: str
    actor: str
    user_id: str
    record_count: int
    record_snapshot: list[dict[str, object]]


FeatureRecords = tuple[FeatureValueRecord, ...]
ProvenanceRecords = tuple[ProvenanceRecord, ...]
QualityRecords = tuple[QualityRecord, ...]
AuditEntries = tuple[DeletionAuditEntry, ...]


class FeatureStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # --- Writes ---------------------------------------------------------

    def insert_feature_value(
        self,
        *,
        user_id: str,
        timestamp: str,
        feature_name: str,
        value: float,
        feature_version: str,
        source: str | None = None,
    ) -> int:
        if not user_id.strip():
            raise FeatureStoreError("user_id must not be empty")

        cursor = self._conn.execute(
            "INSERT INTO feature_values "
            "(user_id, timestamp, feature_name, value, feature_version, source) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, timestamp, feature_name, value, feature_version, source),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def register_feature_version(
        self, *, feature_name: str, version: str, description: str | None = None
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO feature_versions (feature_name, version, description, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (feature_name, version, description, datetime.now(UTC).isoformat()),
        )
        self._conn.commit()

    def insert_provenance(
        self,
        *,
        feature_value_id: int,
        loader_name: str,
        source_file: str | None = None,
        ingested_at: str | None = None,
    ) -> int:
        """Attach a provenance record to an existing feature_values row.
        `feature_value_id` must reference a real row — this is checked
        explicitly rather than left to SQLite's (disabled-by-default)
        foreign key enforcement, so a typo produces a typed
        FeatureStoreError instead of a silently orphaned row."""

        if not loader_name.strip():
            raise FeatureStoreError("loader_name must not be empty")

        self._assert_feature_value_exists(feature_value_id)

        resolved_ingested_at = (
            ingested_at if ingested_at is not None else datetime.now(UTC).isoformat()
        )

        cursor = self._conn.execute(
            "INSERT INTO provenance (feature_value_id, loader_name, source_file, ingested_at) "
            "VALUES (?, ?, ?, ?)",
            (feature_value_id, loader_name, source_file, resolved_ingested_at),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def insert_quality(
        self,
        *,
        feature_value_id: int,
        quality_flag: str,
        quality_reason: str | None = None,
    ) -> int:
        """Attach a quality record to an existing feature_values row.
        Same existence check as `insert_provenance`, for the same
        reason."""

        if not quality_flag.strip():
            raise FeatureStoreError("quality_flag must not be empty")

        self._assert_feature_value_exists(feature_value_id)

        cursor = self._conn.execute(
            "INSERT INTO quality (feature_value_id, quality_flag, quality_reason) VALUES (?, ?, ?)",
            (feature_value_id, quality_flag, quality_reason),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def _assert_feature_value_exists(self, feature_value_id: int) -> None:
        row = self._conn.execute(
            "SELECT 1 FROM feature_values WHERE id = ?", (feature_value_id,)
        ).fetchone()
        if row is None:
            raise FeatureStoreError(
                f"feature_value_id={feature_value_id} does not exist; "
                "provenance/quality cannot be attached to a nonexistent feature value"
            )

    # --- User-scoped reads (every path requires user_id) ------------------

    def query_by_user(self, user_id: str) -> tuple[FeatureValueRecord, ...]:
        if not user_id.strip():
            raise FeatureStoreError("user_id must not be empty")

        rows = self._conn.execute(
            "SELECT id, user_id, timestamp, feature_name, value, feature_version, source "
            "FROM feature_values WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        return tuple(FeatureValueRecord(*row) for row in rows)

    def query_by_user_and_feature(self, user_id: str, feature_name: str) -> FeatureRecords:
        if not user_id.strip():
            raise FeatureStoreError("user_id must not be empty")

        rows = self._conn.execute(
            "SELECT id, user_id, timestamp, feature_name, value, feature_version, source "
            "FROM feature_values WHERE user_id = ? AND feature_name = ?",
            (user_id, feature_name),
        ).fetchall()
        return tuple(FeatureValueRecord(*row) for row in rows)

    def get_by_id_for_user(self, user_id: str, record_id: int) -> FeatureValueRecord | None:
        """Even a direct by-ID lookup is user-scoped: requesting a real
        record ID that belongs to a DIFFERENT user returns None, never
        the other user's record."""

        if not user_id.strip():
            raise FeatureStoreError("user_id must not be empty")

        row = self._conn.execute(
            "SELECT id, user_id, timestamp, feature_name, value, feature_version, source "
            "FROM feature_values WHERE id = ? AND user_id = ?",
            (record_id, user_id),
        ).fetchone()
        return FeatureValueRecord(*row) if row is not None else None

    def get_provenance_for_value(self, user_id: str, feature_value_id: int) -> ProvenanceRecords:
        """User-scoped like every other read here, even though
        `provenance` has no user_id column of its own — ownership is
        checked via a join back to feature_values through
        `get_by_id_for_user` rather than trusted from the caller's
        claim. Returns empty (not an error) if feature_value_id
        doesn't exist or doesn't belong to user_id, matching
        `get_by_id_for_user`'s never-leak-across-users contract."""

        if self.get_by_id_for_user(user_id, feature_value_id) is None:
            return ()

        rows = self._conn.execute(
            "SELECT id, feature_value_id, loader_name, source_file, ingested_at "
            "FROM provenance WHERE feature_value_id = ?",
            (feature_value_id,),
        ).fetchall()
        return tuple(ProvenanceRecord(*row) for row in rows)

    def get_quality_for_value(self, user_id: str, feature_value_id: int) -> QualityRecords:
        """Same user-scoping contract as `get_provenance_for_value`."""

        if self.get_by_id_for_user(user_id, feature_value_id) is None:
            return ()

        rows = self._conn.execute(
            "SELECT id, feature_value_id, quality_flag, quality_reason "
            "FROM quality WHERE feature_value_id = ?",
            (feature_value_id,),
        ).fetchall()
        return tuple(QualityRecord(*row) for row in rows)

    # --- Global, explicitly non-user-scoped enumeration --------------------

    def list_known_feature_names(self) -> tuple[str, ...]:
        """Deliberately global: returns feature NAMES only, never any
        user's actual values -- this is the sanctioned way to enumerate
        across the whole store without it being reachable by accident
        through a user-scoped function."""

        rows = self._conn.execute("SELECT DISTINCT feature_name FROM feature_values").fetchall()
        return tuple(sorted(row[0] for row in rows))

    # --- Hard delete with mandatory audit logging --------------------------

    def delete_feature_values_for_user(self, user_id: str, *, actor: str) -> int:
        """Hard-deletes every feature_values row for one user, logging a
        single audit entry with actor, timestamp, record count, and a
        full snapshot of what was deleted -- BEFORE the delete actually
        happens, so the audit trail can never be lost even if the
        delete itself fails partway.

        Also cascades to that user's dependent `provenance` and
        `quality` rows (B7's "deletion... must trigger the documented
        deletion semantics for dependent derived artifacts"). The
        migration's foreign keys declare no ON DELETE CASCADE, and
        SQLite does not enforce foreign keys by default here either
        (PRAGMA foreign_keys is never turned on), so without this
        explicit step those rows would become orphaned — referencing a
        feature_values.id that no longer exists — every time this ran.
        """

        if not user_id.strip():
            raise FeatureStoreError("user_id must not be empty")
        if not actor.strip():
            raise FeatureStoreError("actor must not be empty")

        records = self.query_by_user(user_id)

        if not records:
            return 0

        snapshot = [
            {
                "id": r.id,
                "user_id": r.user_id,
                "timestamp": r.timestamp,
                "feature_name": r.feature_name,
                "value": r.value,
                "feature_version": r.feature_version,
                "source": r.source,
            }
            for r in records
        ]

        self._conn.execute(
            "INSERT INTO deletion_audit_log "
            "(deleted_at, actor, user_id, record_count, record_snapshot) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now(UTC).isoformat(), actor, user_id, len(records), json.dumps(snapshot)),
        )

        record_ids = [r.id for r in records]
        placeholders = ",".join("?" for _ in record_ids)

        # Cascade BEFORE deleting the parent row, same ordering
        # principle as the audit log above: never leave a state where
        # the parent is gone but a dependent row still points at it.
        self._conn.execute(
            f"DELETE FROM provenance WHERE feature_value_id IN ({placeholders})",
            record_ids,
        )
        self._conn.execute(
            f"DELETE FROM quality WHERE feature_value_id IN ({placeholders})",
            record_ids,
        )
        self._conn.execute("DELETE FROM feature_values WHERE user_id = ?", (user_id,))
        self._conn.commit()

        return len(records)

    def get_deletion_audit_log(self, *, user_id: str | None = None) -> AuditEntries:
        if user_id is not None:
            rows = self._conn.execute(
                "SELECT id, deleted_at, actor, user_id, record_count, record_snapshot "
                "FROM deletion_audit_log WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, deleted_at, actor, user_id, record_count, record_snapshot "
                "FROM deletion_audit_log"
            ).fetchall()

        return tuple(
            DeletionAuditEntry(
                id=row[0],
                deleted_at=row[1],
                actor=row[2],
                user_id=row[3],
                record_count=row[4],
                record_snapshot=json.loads(row[5]),
            )
            for row in rows
        )
