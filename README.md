# CHRONIS Task-2

## Pilot operations

Register opaque `PENDING` pilot IDs in [pilot_roster.md](pilot_roster.md). A real
submission uses the production path:

```bash
python3 cli.py add-event --pilot-id pilot-01 --input-id day-1 "real event wording"
python3 cli.py query 2026-08-20 --pilot-id pilot-01
```

Real events require both IDs and are stored append-only. Invalid input, duplicate
submission IDs, invalid timestamps, and provider failures are rejected with a trace
ID. Never put names or raw events in the roster or operational log.

After Day 4, run the read-only report with the shared Day 1 date:

```bash
python3 inspect_pilot_data.py --start YYYY-MM-DD --demo2
python3 inspect_pilot_data.py --start YYYY-MM-DD --demo2 --json
```

The report excludes `synthetic` and legacy `unattributed` events, reports coverage,
duplicates, malformed records, and real evidence references. `NO_QUALIFYING_PATTERN`
is an expected honest outcome.

## Verification

```bash
python3 -m pytest -q
python3 cli.py demo
```

`demo` is deterministic synthetic rehearsal only. For a labelled synthetic live
validation, explicitly supply `--data-kind synthetic`; it will never appear in a
real-pilot report.
