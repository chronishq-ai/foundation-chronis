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
