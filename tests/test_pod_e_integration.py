import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import analyzer
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
            self.assertIn("event_id", record)

    def test_empty_pilot_input_is_rejected_before_provider_use(self):
        class Args:
            text = "  "
            pilot_id = "fixture"
            input_id = "empty"
            at = None
        with self.assertRaises(SystemExit) as result:
            cli.cmd_add_event(Args())
        self.assertIn("event text must not be empty", str(result.exception))

    def test_provider_failure_uses_bounded_backoff_then_raises(self):
        class FailingProvider:
            def generate(self, prompt):
                raise RuntimeError("temporary provider outage")
        analyzer.set_provider(FailingProvider())
        with patch("analyzer.time.sleep") as sleep:
            with self.assertRaises(RuntimeError):
                analyzer.analyze_event("synthetic outage fixture", max_retries=3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.0, 2.0])
        analyzer.set_provider(None)

    def test_synthetic_cli_ingestion_is_labelled_and_queryable_only_explicitly(self):
        class Provider:
            def generate(self, prompt):
                return '{"signals":{"mood":{"value":8,"confidence":1}}}'
        class AddArgs:
            text = "synthetic test event"
            pilot_id = "fixture-01"
            input_id = "fixture-input-1"
            at = "2026-08-10T10:00:00Z"
            data_kind = "synthetic"
        class QueryArgs:
            date = "2026-08-10"
            pilot_id = "fixture-01"
            data_kind = "synthetic"
        with tempfile.TemporaryDirectory() as directory:
            previous_path = cli.DB_PATH
            cli.DB_PATH = str(Path(directory) / "events.db")
            analyzer.set_provider(Provider())
            try:
                cli.cmd_add_event(AddArgs())
                output = io.StringIO()
                with redirect_stdout(output):
                    cli.cmd_query(QueryArgs())
                self.assertIn('"mood": 7.4', output.getvalue())
            finally:
                analyzer.set_provider(None)
                cli.DB_PATH = previous_path


if __name__ == "__main__":
    unittest.main()
