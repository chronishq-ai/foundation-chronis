"""Tests for the append-only event store and its checkpoints."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from event_store import (
    add_event,
    create_checkpoint,
    fetch_events_between,
    fetch_latest_checkpoint_before,
    get_state_at,
    initialize_database,
)


class EventStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_events.db"
        initialize_database(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_fetches_only_events_inside_range(self) -> None:
        add_event("Before", "2026-06-01T08:00:00Z", {}, 0.9, self.db_path)
        add_event("Inside", "2026-06-10T08:00:00Z", {"x": 1}, 0.8, self.db_path)
        add_event("After", "2026-06-20T08:00:00Z", {}, 0.7, self.db_path)

        results = fetch_events_between(
            "2026-06-09T00:00:00Z",
            "2026-06-11T00:00:00Z",
            self.db_path,
        )

        self.assertEqual(
            [event.description for event in results],
            ["Inside"],
        )

    def test_database_rejects_updates(self) -> None:
        event_id = add_event(
            "Original event",
            "2026-06-10T08:00:00Z",
            {},
            0.9,
            self.db_path,
        )

        connection = sqlite3.connect(self.db_path)
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE events SET description = ? WHERE id = ?",
                    ("Changed event", event_id),
                )
        finally:
            connection.close()

    def test_database_rejects_deletes(self) -> None:
        event_id = add_event(
            "Permanent event",
            "2026-06-10T08:00:00Z",
            {},
            0.9,
            self.db_path,
        )

        connection = sqlite3.connect(self.db_path)
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "DELETE FROM events WHERE id = ?",
                    (event_id,),
                )
        finally:
            connection.close()

    def test_checkpoint_created_automatically_at_interval(self) -> None:
        for i in range(5):
            add_event(
                f"Event {i}",
                f"2026-06-{i + 1:02d}T08:00:00Z",
                {"counter": i},
                0.9,
                self.db_path,
                checkpoint_interval=5,
            )

        checkpoint = fetch_latest_checkpoint_before(
            "2026-06-30T00:00:00Z", self.db_path
        )

        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint.event_count, 5)
        self.assertEqual(checkpoint.last_event_id, 5)
        self.assertEqual(checkpoint.state_data["counter"], 4)

    def test_no_checkpoint_before_interval_reached(self) -> None:
        add_event(
            "Only one event",
            "2026-06-01T08:00:00Z",
            {"x": 1},
            0.9,
            self.db_path,
            checkpoint_interval=5,
        )

        checkpoint = fetch_latest_checkpoint_before(
            "2026-06-30T00:00:00Z", self.db_path
        )
        self.assertIsNone(checkpoint)

    def test_manual_checkpoint_flushes_remaining_events(self) -> None:
        add_event("Only one event", "2026-06-01T08:00:00Z", {"x": 1}, 0.9, self.db_path)

        checkpoint_id = create_checkpoint(self.db_path)
        self.assertIsNotNone(checkpoint_id)

        checkpoint = fetch_latest_checkpoint_before(
            "2026-06-30T00:00:00Z", self.db_path
        )
        self.assertEqual(checkpoint.event_count, 1)
        self.assertEqual(checkpoint.state_data, {"x": 1})

    def test_create_checkpoint_returns_none_without_new_events(self) -> None:
        add_event("Only one event", "2026-06-01T08:00:00Z", {"x": 1}, 0.9, self.db_path)
        create_checkpoint(self.db_path)

        # No new events since the checkpoint above.
        result = create_checkpoint(self.db_path)
        self.assertIsNone(result)

    def test_checkpoints_table_rejects_updates_and_deletes(self) -> None:
        add_event("Only one event", "2026-06-01T08:00:00Z", {"x": 1}, 0.9, self.db_path)
        create_checkpoint(self.db_path)

        connection = sqlite3.connect(self.db_path)
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("UPDATE checkpoints SET event_count = 99")
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("DELETE FROM checkpoints")
        finally:
            connection.close()

    def test_get_state_at_uses_checkpoint_and_remaining_events(self) -> None:
        add_event("a", "2026-06-01T08:00:00Z", {"a": 1}, 0.9, self.db_path, checkpoint_interval=2)
        add_event("b", "2026-06-02T08:00:00Z", {"b": 2}, 0.9, self.db_path, checkpoint_interval=2)
        # Checkpoint now covers events 1-2 (a=1, b=2).
        add_event("c", "2026-06-03T08:00:00Z", {"c": 3}, 0.9, self.db_path, checkpoint_interval=2)

        state = get_state_at("2026-06-03T12:00:00Z", self.db_path)
        self.assertEqual(state, {"a": 1, "b": 2, "c": 3})


if __name__ == "__main__":
    unittest.main()
