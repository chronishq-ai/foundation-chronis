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
        )
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
        connection.commit()
    finally:
        connection.close()


def add_event(
    description: str,
    happened_at: datetime | str,
    change_data: Any,
    confidence: float,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    if not description or not description.strip():
        raise ValueError("description must not be empty")

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")

    initialize_database(db_path)

    timestamp = _normalise_datetime(happened_at)
    serialised_change = json.dumps(
        change_data,
        ensure_ascii=False,
        sort_keys=True,
    )

    connection = _connect(db_path)
    try:
        cursor = connection.execute(
            """
            INSERT INTO events (
                description,
                happened_at,
                change_data,
                confidence
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                description.strip(),
                timestamp,
                serialised_change,
                confidence,
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
) -> list[Event]:
    initialize_database(db_path)

    start_value = _normalise_datetime(start)
    end_value = _normalise_datetime(end)

    if start_value > end_value:
        raise ValueError("start must be earlier than or equal to end")

    connection = _connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT
                id,
                description,
                happened_at,
                change_data,
                confidence,
                created_at
            FROM events
            WHERE happened_at BETWEEN ? AND ?
            ORDER BY happened_at ASC, id ASC
            """,
            (start_value, end_value),
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
        )
        for row in rows
    ]
