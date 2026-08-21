"""Append-only SQLite event history store, with periodic state checkpoints.

v0.2 adds a `checkpoints` table so that long event histories don't require
replaying every event from the beginning to know "what the state was" at a
given point in time. A checkpoint is a folded (merged) snapshot of every
event's `change_data` up to and including some event, stored as JSON.

Checkpoints are created automatically every `DEFAULT_CHECKPOINT_INTERVAL`
events (see `add_event`), and can also be created on demand via
`create_checkpoint()`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

DEFAULT_DB_PATH = Path(__file__).with_name("events.db")

# Number of new events that triggers an automatic checkpoint.
DEFAULT_CHECKPOINT_INTERVAL = 5

# Seconds sqlite3 will wait on a locked database before raising
# `sqlite3.OperationalError: database is locked`. Needed once more than one
# writer (e.g. Pod E's concurrent pilot users) can hit this module at once.
DEFAULT_BUSY_TIMEOUT = 30.0

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class Checkpoint:
    id: int
    happened_at: str
    last_event_id: int
    event_count: int
    state_data: Any
    size_bytes: int
    created_at: str


# --------------------------------------------------------------------------
# State merging is pluggable. The naive `default_merge_state` below is a
# last-writer-wins dict merge -- it does NOT reproduce Pod A's confidence /
# spread-weighted blending. This module has no access to that algorithm, so
# instead of guessing at it (and risking silently wrong state), callers that
# own that logic should register it once at startup via
# `set_state_merge_function(...)`. Every fold (checkpoint creation and
# `get_state_at`) then goes through whatever function is registered.
# --------------------------------------------------------------------------

StateMerger = Callable[[dict[str, Any], "Event"], dict[str, Any]]


def default_merge_state(state: dict[str, Any], event: "Event") -> dict[str, Any]:
    """Last-writer-wins shallow merge. Fallback only -- see module note above."""
    merged = dict(state)
    merged.update(event.change_data)
    return merged


_STATE_MERGE_FN: StateMerger = default_merge_state


def set_state_merge_function(fn: StateMerger | None) -> None:
    """Register the function used to fold each event's `change_data` into
    the running state. Pass None to restore the naive default merge.

    Example, wiring in Pod A's real engine:
        from pod_a.engine import update_state
        event_store.set_state_merge_function(update_state)
    """
    global _STATE_MERGE_FN
    _STATE_MERGE_FN = fn if fn is not None else default_merge_state


def _connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    # isolation_level=None -> autocommit mode, so we control transactions
    # explicitly (needed for the BEGIN IMMEDIATE lock in create_checkpoint).
    # timeout -> how long sqlite3 waits on a locked db instead of failing
    # immediately, which matters once multiple pilot users write concurrently.
    connection = sqlite3.connect(
        str(db_path),
        timeout=DEFAULT_BUSY_TIMEOUT,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout = {int(DEFAULT_BUSY_TIMEOUT * 1000)}")
    connection.execute("PRAGMA journal_mode = WAL")
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

    CREATE TABLE IF NOT EXISTS checkpoints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        happened_at TEXT NOT NULL,
        last_event_id INTEGER NOT NULL,
        event_count INTEGER NOT NULL,
        state_data TEXT NOT NULL DEFAULT '{}',
        size_bytes INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT (
            strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
        )
    );

    CREATE INDEX IF NOT EXISTS idx_checkpoints_happened_at
        ON checkpoints (happened_at);

    CREATE INDEX IF NOT EXISTS idx_checkpoints_last_event_id
        ON checkpoints (last_event_id);

    CREATE TRIGGER IF NOT EXISTS checkpoints_prevent_update
    BEFORE UPDATE ON checkpoints
    BEGIN
        SELECT RAISE(
            ABORT,
            'checkpoints table is append-only: updates are forbidden'
        );
    END;

    CREATE TRIGGER IF NOT EXISTS checkpoints_prevent_delete
    BEFORE DELETE ON checkpoints
    BEGIN
        SELECT RAISE(
            ABORT,
            'checkpoints table is append-only: deletes are forbidden'
        );
    END;
    """

    connection = _connect(db_path)
    try:
        connection.executescript(schema)
        existing = {row[1] for row in connection.execute("PRAGMA table_info(events)")}
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
    auto_checkpoint: bool = True,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
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

    if not isinstance(change_data, dict):
        # Folding (checkpoints, get_state_at) treats change_data as a state
        # patch and merges its keys. A non-dict here used to be silently
        # dropped during folding -- the event would persist in `events` but
        # vanish from `checkpoints`, so replaying full history and loading
        # a checkpoint could disagree. Reject it up front instead.
        raise TypeError(
            "change_data must be a dict (it is folded into state as a "
            f"patch); got {type(change_data).__name__}"
        )

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
        new_id = int(cursor.lastrowid)
    finally:
        connection.close()

    if auto_checkpoint and checkpoint_interval > 0:
        # Don't pre-check "pending >= interval" here: with concurrent
        # writers, two callers can both see the threshold crossed before
        # either checkpoints, and both then create one. The real check now
        # happens inside create_checkpoint() itself, under a write lock, so
        # it's safe (and cheap) to just always ask.
        create_checkpoint(db_path, min_new_events=checkpoint_interval)

    return new_id


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

    return [_row_to_event(row) for row in rows]


def _row_to_event(row: sqlite3.Row) -> Event:
    return Event(
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


def _row_to_checkpoint(row: sqlite3.Row) -> Checkpoint:
    return Checkpoint(
        id=row["id"],
        happened_at=row["happened_at"],
        last_event_id=row["last_event_id"],
        event_count=row["event_count"],
        state_data=json.loads(row["state_data"]),
        size_bytes=row["size_bytes"],
        created_at=row["created_at"],
    )


def _fold_state(
    events: list[Event],
    base_state: dict | None = None,
    merge_fn: StateMerger | None = None,
) -> dict:
    """Merge a sequence of events' change_data into a single state dict.

    Events are folded in order via `merge_fn` (defaults to whatever is
    registered through `set_state_merge_function`, or the naive
    last-writer-wins merge if nothing has been registered).

    `change_data` is validated as a dict at write time in `add_event`, so
    every event reaching this function is safe to fold -- nothing is
    silently skipped here.
    """
    merge = merge_fn if merge_fn is not None else _STATE_MERGE_FN
    state: dict[str, Any] = dict(base_state) if base_state else {}
    for event in events:
        state = merge(state, event)
    return state


def _count_events_since_last_checkpoint(db_path: str | Path = DEFAULT_DB_PATH) -> int:
    """Informational only -- not used to gate checkpoint creation anymore.

    Reading this and then deciding whether to call create_checkpoint()
    outside of its own transaction reintroduces the check-then-act race.
    Use create_checkpoint(..., min_new_events=N) instead, which does this
    check atomically under its write lock.
    """
    initialize_database(db_path)
    connection = _connect(db_path)
    try:
        last_event_id = _latest_checkpoint_last_event_id(connection)
        row = connection.execute(
            "SELECT COUNT(*) AS pending FROM events WHERE id > ?",
            (last_event_id,),
        ).fetchone()
        return int(row["pending"])
    finally:
        connection.close()


def _latest_checkpoint_last_event_id(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT last_event_id FROM checkpoints ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return int(row["last_event_id"]) if row else 0


def create_checkpoint(
    db_path: str | Path = DEFAULT_DB_PATH,
    min_new_events: int = 1,
    merge_fn: StateMerger | None = None,
) -> int | None:
    """Fold every event since the last checkpoint into a new checkpoint.

    `min_new_events` lets callers (like `add_event`'s auto-checkpoint) ask
    "only checkpoint if there are at least N new events" -- and have that
    check happen atomically, under the write lock below, rather than in a
    separate read beforehand. That's what makes concurrent writers safe:
    with `BEGIN IMMEDIATE`, only one connection at a time can be inside this
    function's transaction, so two callers racing to cross the threshold
    will serialize here instead of both creating a checkpoint. The second
    one to run sees the already-advanced last_event_id and (correctly)
    finds fewer pending events than `min_new_events`, and skips.

    Returns the new checkpoint's id, or None if there were fewer than
    `min_new_events` new events since the last checkpoint.
    """
    initialize_database(db_path)

    connection = _connect(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")

        last_checkpoint_row = connection.execute(
            "SELECT last_event_id, state_data FROM checkpoints "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()

        last_event_id = (
            int(last_checkpoint_row["last_event_id"]) if last_checkpoint_row else 0
        )
        base_state = (
            json.loads(last_checkpoint_row["state_data"])
            if last_checkpoint_row
            else {}
        )

        rows = connection.execute(
            """
            SELECT id, description, happened_at, change_data, confidence, created_at, pilot_id, input_id, data_kind
            FROM events
            WHERE id > ?
            ORDER BY id ASC
            """,
            (last_event_id,),
        ).fetchall()

        if len(rows) < max(min_new_events, 1):
            connection.execute("ROLLBACK")
            return None

        new_events = [_row_to_event(row) for row in rows]
        new_state = _fold_state(new_events, base_state, merge_fn=merge_fn)
        serialised_state = json.dumps(new_state, ensure_ascii=False, sort_keys=True)
        size_bytes = len(serialised_state.encode("utf-8"))

        cursor = connection.execute(
            """
            INSERT INTO checkpoints (
                happened_at,
                last_event_id,
                event_count,
                state_data,
                size_bytes
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                new_events[-1].happened_at,
                new_events[-1].id,
                len(new_events),
                serialised_state,
                size_bytes,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def fetch_latest_checkpoint_before(
    timestamp: datetime | str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> Checkpoint | None:
    """Return the most recent checkpoint at or before `timestamp`, if any."""
    initialize_database(db_path)

    timestamp_value = _normalise_datetime(timestamp)

    connection = _connect(db_path)
    try:
        row = connection.execute(
            """
            SELECT id, happened_at, last_event_id, event_count, state_data,
                   size_bytes, created_at
            FROM checkpoints
            WHERE happened_at <= ?
            ORDER BY happened_at DESC, id DESC
            LIMIT 1
            """,
            (timestamp_value,),
        ).fetchone()
    finally:
        connection.close()

    return _row_to_checkpoint(row) if row else None


def get_state_at(
    timestamp: datetime | str,
    db_path: str | Path = DEFAULT_DB_PATH,
    merge_fn: StateMerger | None = None,
) -> dict:
    """Reconstruct the folded state as of `timestamp`.

    Starts from the latest checkpoint at or before `timestamp` (if any) and
    replays only the remaining events, instead of replaying the full
    history from the beginning.
    """
    timestamp_value = _normalise_datetime(timestamp)
    checkpoint = fetch_latest_checkpoint_before(timestamp_value, db_path)

    base_state = checkpoint.state_data if checkpoint else {}
    since_event_id = checkpoint.last_event_id if checkpoint else 0

    initialize_database(db_path)
    connection = _connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT id, description, happened_at, change_data, confidence, created_at, pilot_id, input_id, data_kind
            FROM events
            WHERE id > ? AND happened_at <= ?
            ORDER BY id ASC
            """,
            (since_event_id, timestamp_value),
        ).fetchall()
    finally:
        connection.close()

    remaining_events = [_row_to_event(row) for row in rows]
    return _fold_state(remaining_events, base_state, merge_fn=merge_fn)


def get_storage_stats(db_path: str | Path = DEFAULT_DB_PATH) -> dict:
    """Return simple counts and average row sizes, useful for projecting
    long-term storage growth."""
    initialize_database(db_path)
    connection = _connect(db_path)
    try:
        event_rows = connection.execute(
            "SELECT description, change_data FROM events"
        ).fetchall()
        checkpoint_rows = connection.execute(
            "SELECT size_bytes FROM checkpoints"
        ).fetchall()
    finally:
        connection.close()

    event_count = len(event_rows)
    if event_count:
        total_event_bytes = sum(
            len(row["description"].encode("utf-8"))
            + len(row["change_data"].encode("utf-8"))
            for row in event_rows
        )
        avg_event_bytes = total_event_bytes / event_count
    else:
        avg_event_bytes = 0.0

    checkpoint_count = len(checkpoint_rows)
    if checkpoint_count:
        avg_checkpoint_bytes = (
            sum(row["size_bytes"] for row in checkpoint_rows) / checkpoint_count
        )
    else:
        avg_checkpoint_bytes = 0.0

    db_file_bytes = Path(db_path).stat().st_size if Path(db_path).exists() else 0

    return {
        "event_count": event_count,
        "checkpoint_count": checkpoint_count,
        "avg_event_bytes": avg_event_bytes,
        "avg_checkpoint_bytes": avg_checkpoint_bytes,
        "db_file_bytes": db_file_bytes,
    }


def estimate_storage_for_years(
    years: float,
    events_per_day: float,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict:
    """Project total storage (events + checkpoints) after `years` of use at
    a steady `events_per_day` rate, based on average sizes observed in the
    current database.
    """
    stats = get_storage_stats(db_path)

    projected_events = events_per_day * 365 * years
    projected_checkpoints = (
        projected_events / checkpoint_interval if checkpoint_interval > 0 else 0
    )

    projected_event_bytes = projected_events * stats["avg_event_bytes"]
    projected_checkpoint_bytes = (
        projected_checkpoints * stats["avg_checkpoint_bytes"]
    )
    total_bytes = projected_event_bytes + projected_checkpoint_bytes

    return {
        "years": years,
        "events_per_day": events_per_day,
        "projected_events": projected_events,
        "projected_checkpoints": projected_checkpoints,
        "projected_event_bytes": projected_event_bytes,
        "projected_checkpoint_bytes": projected_checkpoint_bytes,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
    }