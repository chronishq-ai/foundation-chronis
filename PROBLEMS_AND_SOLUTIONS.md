# Pilot rollout problems and solutions

| Problem | Resolution | Evidence / files |
|---|---|---|
| Provider outages could make ingestion unreliable. | Bounded exponential retry; explicit rejection after retry budget. No neutral fallback is created. | `analyzer.py`, `config.py`, `tests/test_pod_e_integration.py` |
| Empty input, invalid timestamps, and missing pilot identity were not uniformly rejected. | CLI validates text, timestamp, and required real-pilot IDs before storage. | `cli.py`, `event_store.py` |
| Pilot IDs were only in logs, so coverage could not be calculated safely. | Append-only event metadata now stores `pilot_id`, `input_id`, and `data_kind`. Legacy rows remain `unattributed`. | `event_store.py` |
| Synthetic validation could be confused with real evidence. | `real`, `synthetic`, and `unattributed` classifications are reported separately; only real labelled records contribute to coverage or Demo 2. | `cli.py`, `pilot_reporting.py`, `inspect_pilot_data.py` |
| Duplicate submissions were not identified. | Real pilot submissions reject a repeated pilot/input-ID pair; inspection also reports duplicate legacy records. | `event_store.py`, `pilot_reporting.py` |
| Queries could combine different users' history. | Queries require a pilot ID and filter storage to that pilot; synthetic queries require an explicit data-kind flag. | `cli.py`, `event_store.py` |
| Four-day completion was manual and unverifiable. | Read-only reporting computes COMPLETE/PARTIAL/MISSING coverage and missing dates from a declared shared Day 1. | `pilot_reporting.py`, `inspect_pilot_data.py` |
| Demo 2 could be invented from test data. | Evidence candidates contain real pilot/event references and pipeline state only; absent evidence returns `NO_QUALIFYING_PATTERN`. | `pilot_reporting.py`, `inspect_pilot_data.py` |
| Failures could expose raw pilot text. | Structured logs use trace ID, event ID, input ID, and safe input reference rather than raw wording. | `integration_log.py` |
| Rollout instructions were ambiguous. | Opaque roster, production submission command, inspection command, and synthetic-data policy are documented. | `README.md`, `PILOT_USER_GUIDE.md`, `pilot_roster.md` |

## External items deliberately not solved in code

- Recruiting 5–10 real participants.
- Four days of genuine participant submissions.
- A real-data Demo 2 candidate (or the honest absence of one).
- Independent second-engineer review before merge.

These must be completed by humans; no fixture or generated record is treated as a
real pilot or Demo 2 observation.
