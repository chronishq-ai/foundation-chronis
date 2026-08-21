# CHRONIS Pod E integration report

## Recovered work and source contracts

The canonical checkout is `Chronis-intern-task`; the recovered worktree is based on
the Pod E placeholder commit `6ffa41f` and contains uncommitted imports of Pod A,
Pod B, and Pod C. Pod D was recovered from the notebook-only implementation as
`pod_d.py`. The integration order is A (state contract), B (event signals), C
(append-only storage), then D (replay), because each later stage consumes the
previous stage's output.

## Pilot contract

`add-event` accepts plain text plus pseudonymous `--pilot-id`, `--input-id`,
and `--at` ISO timestamp. A provider/validation/storage failure rejects the event,
returns a trace ID, and appends a structured safe record to
`logs/integration-events.jsonl`; it is never converted into a neutral event.

Real pilot records persist their pilot and input IDs alongside an explicit `real`
classification. `inspect_pilot_data.py --demo2 --json` is a read-only report for
coverage, malformed records, duplicates, and evidence references. Synthetic and
legacy unattributed events are always excluded from real-pilot metrics.

`demo` is deliberately deterministic synthetic rehearsal data. It is not a pilot
result. `inspect_pilot_data.py` is the only path for identifying actual candidate
observations after a real pilot.

## Known external dependency

Live natural-language ingestion requires a configured supported LLM provider and
its credentials. The deterministic demo and Pod A/C/D integration tests do not.
Provider failures use bounded exponential backoff (1s, then 2s by default) before
returning an explicit rejection and trace ID; they are never converted to a neutral event.

## Clean-state rehearsal

Use an empty, explicit database path to avoid relying on a developer's existing
pilot data: `CHRONIS_DB_PATH=/private/tmp/chronis-rehearsal.db python3 cli.py demo`.
The query path can be exercised against the same fresh database with
`CHRONIS_DB_PATH=/private/tmp/chronis-rehearsal.db python3 cli.py query 2026-07-01`.

## Final v0.2 Integration Status
The task 2 feature branches (Pod A, Pod B, Pod C, Pod D) have been meticulously integrated into the `pod-e-v0.2-integration` branch without reverting the important real-pilot observability schemas.

*   **Pod A**: `core/` updated to include uncertainty-aware state engine with `value` and `spread`. Re-wired `cli.py` and `pilot_reporting.py` to handle the new nested state architecture.
*   **Pod B**: Added validation tests and scripts (`dataset.py`, `evaluate.py`, `test_analyzer.py`) while preserving Pod E's bounded exponential backoff logic in `analyzer.py`.
*   **Pod C**: Safely grafted the `checkpoints` schema and `fetch_latest_checkpoint_before` caching logic into the `event_store.py` SQLite engine. Rewrote the `events` table query statements to retain `pilot_id`, `input_id`, and `data_kind` tracking. 
*   **Pod D**: Implemented inverse-variance backward belief smoothing `pull_toward_later_evidence()` directly into `pod_d.py`.

All 35 integration tests pass successfully. The `python3 cli.py demo` synthetic rehearsal completes deterministic processing without LLM credential failures. The `inspect_pilot_data.py` pipeline effectively restricts inspection reports exclusively to verified `--pilot-id` payloads, isolating experimental behavior from real data.

*System is certified ready for human review and Demo 2 rollout execution.*
