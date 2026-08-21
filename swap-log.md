# Pod Upgrade Swap Log

| Pod Name | Date Swapped | Demo Re-run Result | Issues Found & Resolution |
|---|---|---|---|
| **Pod A: Core State Engine** | 2026-08-17 | **PASS** | Initial schema expected `value` key in signals. Enhanced `core/update_state.py` to support both absolute `value` and relative `delta` inputs, ensuring backward and forward compatibility. |
| **Pod B: Event Understanding** | 2026-08-17 | **PASS** | Default model in `.env` (`gemini-2.5-pro`) was deprecated by API provider. Updated active model to `gemini-flash-latest` and added resilient fallback handling in `cli.py` for transient API 503/429 rate limit events. |
| **Pod C: Event Storage** | 2026-08-17 | **PASS** | Swapped in append-only SQLite storage (`events.db`). Updated `cli.py` to route persistence calls through `event_store.py` with triggers preventing UPDATE/DELETE. Verified with `cli.py demo` and `cli.py add-event`. |
| **Pod D: Looking Back & Revision** | 2026-08-17 | **PASS** | Notebook-only code extracted into `pod_d.py` portable module (Colab-specific `files.upload()` and matplotlib removed). `query_then_placeholder` and `query_now_placeholder` delegated to `get_belief_then` / `get_belief_now`. THEN vs NOW diff confirmed working. |
