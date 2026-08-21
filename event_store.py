"""Append-only SQLite event history store."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).with_name("events.db")


@dataclass(frozen=True)
class Event:
    id: int
    description: str
    happened_at: str
    change_data: Any
    confidence: float
    created_at: str
    pilot_id: str | None = None
    input_id: str | None = None
    data_kind: str = "unattributed"


def _connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def _normalise_datetime(value: datetime | str) -> str:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError("Date/time must be a datetime or ISO-8601 string.")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc).isoformat(timespec="seconds")


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    schema = """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        description TEXT NOT NULL,
        happened_at TEXT NOT NULL,
        change_data TEXT NOT NULL DEFAULT '{}',
        confidence REAL NOT NULL
            CHECK (confidence >= 0.0 AND confidence <= 1.0),
        created_at TEXT NOT NULL DEFAULT (
            strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
        ),
        pilot_id TEXT,
        input_id TEXT,
        data_kind TEXT NOT NULL DEFAULT 'unattributed'
            CHECK (data_kind IN ('real', 'synthetic', 'unattributed'))
    );

    CREATE INDEX IF NOT EXISTS idx_events_happened_at
        ON events (happened_at);

    CREATE TRIGGER IF NOT EXISTS events_prevent_update
    BEFORE UPDATE ON events
    BEGIN
        SELECT RAISE(
            ABORT,
            'events table is append-only: updates are forbidden'
        );
    END;

    CREATE TRIGGER IF NOT EXISTS events_prevent_delete
    BEFORE DELETE ON events
    BEGIN
        SELECT RAISE(
            ABORT,
            'events table is append-only: deletes are forbidden'
        );
    END;
    """

    connection = _connect(db_path)
    try:
        connection.executescript(schema)
        existing = {row[1] for row in connection.execute("PRAGMA table_info(events)")}
        # Metadata-only migration: canonical event content is never rewritten.
        if "pilot_id" not in existing:
            connection.execute("ALTER TABLE events ADD COLUMN pilot_id TEXT")
        if "input_id" not in existing:
            connection.execute("ALTER TABLE events ADD COLUMN input_id TEXT")
        if "data_kind" not in existing:
            connection.execute("ALTER TABLE events ADD COLUMN data_kind TEXT NOT NULL DEFAULT 'unattributed'")
        connection.commit()
    finally:
        connection.close()


def add_event(
    description: str,
    happened_at: datetime | str,
    change_data: Any,
    confidence: float,
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    pilot_id: str | None = None,
    input_id: str | None = None,
    data_kind: str = "unattributed",
) -> int:
    if not description or not description.strip():
        raise ValueError("description must not be empty")

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    if data_kind not in {"real", "synthetic", "unattributed"}:
        raise ValueError("data_kind must be real, synthetic, or unattributed")
    if data_kind == "real" and (not pilot_id or not input_id):
        raise ValueError("real pilot events require pilot_id and input_id")

    initialize_database(db_path)

    timestamp = _normalise_datetime(happened_at)
    serialised_change = json.dumps(
        change_data,
        ensure_ascii=False,
        sort_keys=True,
    )

    connection = _connect(db_path)
    try:
        if data_kind == "real":
            duplicate = connection.execute(
                "SELECT id FROM events WHERE pilot_id = ? AND input_id = ? AND data_kind = 'real' LIMIT 1",
                (pilot_id, input_id),
            ).fetchone()
            if duplicate:
                raise ValueError(f"duplicate real pilot submission for pilot_id={pilot_id!r}, input_id={input_id!r}")
        cursor = connection.execute(
            """
            INSERT INTO events (
                description,
                happened_at,
                change_data,
                confidence,
                pilot_id,
                input_id,
                data_kind
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                description.strip(),
                timestamp,
                serialised_change,
                confidence,
                pilot_id,
                input_id,
                data_kind,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def fetch_events_between(
    start: datetime | str,
    end: datetime | str,
    db_path: str | Path = DEFAULT_DB_PATH,
    *,
    pilot_id: str | None = None,
    data_kind: str | None = None,
) -> list[Event]:
    initialize_database(db_path)

    start_value = _normalise_datetime(start)
    end_value = _normalise_datetime(end)

    if start_value > end_value:
        raise ValueError("start must be earlier than or equal to end")

    if data_kind is not None and data_kind not in {"real", "synthetic", "unattributed"}:
        raise ValueError("data_kind must be real, synthetic, or unattributed")

    clauses = ["happened_at BETWEEN ? AND ?"]
    parameters: list[str] = [start_value, end_value]
    if pilot_id is not None:
        clauses.append("pilot_id = ?")
        parameters.append(pilot_id)
    if data_kind is not None:
        clauses.append("data_kind = ?")
        parameters.append(data_kind)
    where_clause = " AND ".join(clauses)
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            f"""
            SELECT
                id,
                description,
                happened_at,
                change_data,
                confidence,
                created_at,
                pilot_id,
                input_id,
                data_kind
            FROM events
            WHERE {where_clause}
            ORDER BY happened_at ASC, id ASC
            """,
            parameters,
        ).fetchall()
    finally:
        connection.close()

    return [
        Event(
            id=row["id"],
            description=row["description"],
            happened_at=row["happened_at"],
            change_data=json.loads(row["change_data"]),
            confidence=row["confidence"],
            created_at=row["created_at"],
            pilot_id=row["pilot_id"],
            input_id=row["input_id"],
            data_kind=row["data_kind"],
        )
        for row in rows
    ]
