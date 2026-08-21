# Edge Cases Log — Phase 3

Format: timestamp | user | raw input | what happened | handled or broke?

---

## Active Log

| Timestamp | User | Raw Input | What Happened | Handled / Broke |
|---|---|---|---|---|
| 2026-08-17T20:10 | Pod B (system) | API call to `gemini-2.5-pro` | Model deprecated (404). The `.env` had an outdated model name. | **Handled** — switched to `gemini-flash-latest` |
| 2026-08-17T20:14 | Pod B (system) | API call to `gemini-flash-latest` | 503 UNAVAILABLE (model overloaded) on 3/3 retry attempts for one event | **Handled** — fallback to neutral signal `{value: 5.0, confidence: 0.5}` for all variables |
| 2026-08-17T20:14 | Pod B (system) | 429 RESOURCE_EXHAUSTED | Free-tier rate limit exceeded mid-demo run (5 req/min) | **Handled** — next retry succeeded on its own after delay; fallback caught the one that exhausted |
| 2026-08-21T13:35 | synthetic fixture | empty input | Rejected before provider invocation; structured log includes trace ID and input hash, not raw input. | **Handled** — no event stored |
| 2026-08-21T13:35 | synthetic fixture | live provider request | DNS unavailable inside restricted sandbox; same request succeeded with approved external network. | **Handled** — no event stored during verification |
| 2026-08-21T13:37 | synthetic fixture | provider → storage → query | Live Gemini request returned 200; validated signals were appended to a disposable database and replayed successfully. | **Handled** — verified end-to-end; explicitly not pilot data |

There are currently no real-pilot entries in `events.db`; the legacy `test event storage`
record is synthetic and must not be used as pilot evidence.

---

*This log is a deliverable. Update it as pilot users generate real data during Phase 2.*
