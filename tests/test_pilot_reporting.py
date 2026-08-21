import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from datetime import date
from pathlib import Path

from event_store import add_event, initialize_database
from pilot_reporting import build_report, parse_roster


def signal(mood):
    return {"mood": {"value": mood, "confidence": 1.0}}


class PilotReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "pilot.db"

    def tearDown(self):
        self.temp.cleanup()

    def add_real(self, pilot, input_id, timestamp, mood=5):
        return add_event("private event", timestamp, signal(mood), 1.0, self.db,
                         pilot_id=pilot, input_id=input_id, data_kind="real")

    def test_roster_parsing_and_empty_roster(self):
        roster = Path(self.temp.name) / "roster.md"
        roster.write_text("| Pilot ID | Status | Day 1 |\n|---|---|---|\n| pilot-01 | ACTIVE | [ ] |\n")
        self.assertEqual(parse_roster(roster), [{"pilot_id": "pilot-01", "status": "ACTIVE"}])
        roster.write_text("| Pilot ID | Status |\n|---|---|\n")
        self.assertEqual(parse_roster(roster), [])

    def test_complete_and_missing_four_day_coverage(self):
        for day in range(4):
            self.add_real("pilot-01", f"d{day + 1}", f"2026-08-{10 + day}T10:00:00Z")
        self.add_real("pilot-02", "d1", "2026-08-10T10:00:00Z")
        report = build_report(self.db, start=date(2026, 8, 10))
        self.assertEqual(report["coverage"]["pilot-01"]["status"], "COMPLETE")
        self.assertEqual(report["coverage"]["pilot-02"]["status"], "PARTIAL")
        self.assertEqual(len(report["coverage"]["pilot-02"]["missing_days"]), 3)

    def test_synthetic_and_legacy_events_are_excluded(self):
        add_event("synthetic", "2026-08-10T10:00:00Z", signal(9), 1.0, self.db,
                  pilot_id="fixture", input_id="fixture-1", data_kind="synthetic")
        add_event("legacy", "2026-08-10T10:00:00Z", signal(9), 1.0, self.db)
        report = build_report(self.db)
        self.assertEqual(report["classification"]["real_pilot_events"], 0)
        self.assertEqual(report["classification"]["synthetic_events"], 1)
        self.assertEqual(report["classification"]["unattributed_legacy_events"], 1)
        self.assertEqual(report["demo2_status"], "NO_QUALIFYING_PATTERN")

    def test_duplicate_and_malformed_detection(self):
        self.add_real("pilot-01", "d1", "2026-08-10T10:00:00Z")
        initialize_database(self.db)
        connection = sqlite3.connect(self.db)
        connection.execute("DROP TRIGGER events_prevent_update")
        connection.execute("INSERT INTO events (description, happened_at, change_data, confidence, pilot_id, input_id, data_kind) VALUES (?, ?, ?, ?, ?, ?, ?)",
                           ("duplicate", "2026-08-10T11:00:00Z", "{}", .5, "pilot-01", "d1", "real"))
        connection.execute("INSERT INTO events (description, happened_at, change_data, confidence, pilot_id, input_id, data_kind) VALUES (?, ?, ?, ?, ?, ?, ?)",
                           ("bad", "not-a-time", "{}", .5, "pilot-02", "bad-1", "real"))
        connection.commit()
        connection.close()
        report = build_report(self.db, start=date(2026, 8, 10))
        self.assertEqual(len(report["duplicates"]), 1)
        self.assertIn("invalid_timestamp", report["malformed_events"][0]["errors"])

    def test_duplicate_real_submission_is_rejected(self):
        self.add_real("pilot-01", "d1", "2026-08-10T10:00:00Z")
        with self.assertRaises(ValueError):
            self.add_real("pilot-01", "d1", "2026-08-10T11:00:00Z")

    def test_candidate_and_no_pattern_result(self):
        self.add_real("pilot-01", "d1", "2026-08-10T10:00:00Z", mood=9)
        self.add_real("pilot-01", "d2", "2026-08-11T10:00:00Z", mood=1)
        report = build_report(self.db, start=date(2026, 8, 10))
        self.assertEqual(report["demo2_status"], "QUALIFYING_PATTERN_FOUND")
        candidate = report["candidate_patterns"][0]
        self.assertEqual(candidate["pilot_id"], "pilot-01")
        self.assertIn("earlier_observation_ref", candidate)

    def test_inspection_cli_json(self):
        self.add_real("pilot-01", "d1", "2026-08-10T10:00:00Z")
        environment = dict(os.environ, CHRONIS_DB_PATH=str(self.db))
        result = subprocess.run(
            ["python3", "inspect_pilot_data.py", "--start", "2026-08-10", "--demo2", "--json"],
            cwd=Path(__file__).parents[1], env=environment, text=True, capture_output=True, check=True,
        )
        report = json.loads(result.stdout)
        self.assertEqual(report["real_pilots"], ["pilot-01"])
        self.assertEqual(report["demo2_status"], "NO_QUALIFYING_PATTERN")


if __name__ == "__main__":
    unittest.main()
