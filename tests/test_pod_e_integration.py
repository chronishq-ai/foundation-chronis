import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import cli
from core.update_state import update_state
from integration_log import log_event
from pod_d import get_belief_now, get_belief_then


class PodEIntegrationTests(unittest.TestCase):
    def test_pod_a_and_d_replay_uses_later_evidence_only_for_now(self):
        initial = dict(cli.STARTING_STATE)
        events = [
            {"timestamp": "2026-07-01T00:00:00+00:00", "signal": {"mood": {"value": 9, "confidence": 1}}},
            {"timestamp": "2026-07-02T00:00:00+00:00", "signal": {"mood": {"value": 1, "confidence": 1}}},
        ]
        then_state = get_belief_then(events, "2026-07-01T23:59:59+00:00", initial, update_state)
        now_state = get_belief_now(events, "2026-07-01T23:59:59+00:00", initial, update_state)
        self.assertGreater(then_state["mood"], now_state["mood"])

    def test_malformed_state_signal_is_rejected(self):
        with self.assertRaises(ValueError):
            update_state(dict(cli.STARTING_STATE), {"not_a_state": {"value": 5, "confidence": .5}})
        with self.assertRaises(TypeError):
            update_state(dict(cli.STARTING_STATE), {"mood": {"value": "high", "confidence": .5}})

    def test_deterministic_demo_does_not_call_provider_or_persist(self):
        output = io.StringIO()
        with redirect_stdout(output):
            cli.cmd_demo(None)
        report = output.getvalue()
        self.assertIn("Synthetic rehearsal only", report)
        self.assertIn("THEN:", report)
        self.assertIn("NOW:", report)

    def test_structured_log_excludes_raw_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            trace = log_event(
                pilot_id="pilot-1", input_id="case-1", component="pod-e",
                event_type="pilot_event", status="rejected", stage="ingestion",
                error_code="ValueError", message="invalid", text="private wording", path=path,
            )
            record = json.loads(path.read_text().strip())
            self.assertEqual(record["trace_id"], trace)
            self.assertNotIn("private wording", path.read_text())
            self.assertEqual(record["status"], "rejected")

    def test_empty_pilot_input_is_rejected_before_provider_use(self):
        class Args:
            text = "  "
            pilot_id = "fixture"
            input_id = "empty"
            at = None
        with self.assertRaises(SystemExit) as result:
            cli.cmd_add_event(Args())
        self.assertIn("event text must not be empty", str(result.exception))


if __name__ == "__main__":
    unittest.main()
