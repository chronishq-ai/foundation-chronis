"""
Phase 4 helper — run after pilot week to inspect events.db and search for
the 'past-belief-changing' pattern.

Usage:
    python3 inspect_pilot_data.py

Output:
    - Total events collected
    - Events per user (if user tag is present in description)
    - All events in chronological order
    - Candidate 'then vs now' pairs where mood/stress/confidence diverge most
      between THEN and NOW views
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from integration_log import log_event

DB_PATH = os.getenv("CHRONIS_DB_PATH", "events.db")
STARTING_STATE = {
    "mood": 5.0, "focus": 5.0, "stress": 5.0,
    "confidence": 5.0, "trust": 5.0,
    "motivation": 5.0, "social_engagement": 5.0,
}

def load_all_events():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, description, happened_at, change_data FROM events ORDER BY happened_at ASC"
    ).fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "text": r["description"],
            "timestamp": r["happened_at"],
            "signal": json.loads(r["change_data"]),
        }
        for r in rows
    ]

def update_state(state, signal):
    from core.update_state import update_state as real_update
    return real_update(state, signal)

def replay(events, up_to_ts=None):
    state = dict(STARTING_STATE)
    failures = []
    for e in events:
        if up_to_ts and e["timestamp"] > up_to_ts:
            break
        try:
            state = update_state(state, e["signal"])
        except Exception as exc:
            failures.append((e["id"], type(exc).__name__, str(exc)))
            log_event(pilot_id=None, input_id=f"event-{e['id']}", component="pod-e",
                      event_type="replay", status="rejected", stage="state_replay",
                      error_code=type(exc).__name__, message=str(exc), text=e["text"])
    return state, failures

def main():
    events = load_all_events()
    print(f"\n=== Pilot Data Summary ===")
    print(f"Total events in events.db: {len(events)}")

    if not events:
        print("\nNo pilot data found. Phase 2 has not been completed.")
        print("Recruit pilot users and have them run: python3 cli.py add-event \"...\"")
        return

    print(f"\nAll events (chronological):")
    for e in events:
        print(f"  [event {e['id']} at {e['timestamp']}] {e['text'][:120]}")

    print(f"\n=== Searching for Past-Belief-Changing Moments ===")
    print("(For each event date, comparing THEN belief vs NOW belief)")
    print()

    best_divergence = 0
    best_candidate = None

    for i, pivot_event in enumerate(events):
        ts = pivot_event["timestamp"]
        then_state, then_failures = replay(events, up_to_ts=ts)
        now_state, now_failures = replay(events)

        divergence = sum(
            abs(now_state.get(k, 5.0) - then_state.get(k, 5.0))
            for k in STARTING_STATE
        )

        print(f"  Pivot: [event {pivot_event['id']} at {ts}] \"{pivot_event['text'][:60]}\"")
        print(f"    THEN mood={then_state.get('mood',5):.2f}  stress={then_state.get('stress',5):.2f}  confidence={then_state.get('confidence',5):.2f}")
        print(f"    NOW  mood={now_state.get('mood',5):.2f}  stress={now_state.get('stress',5):.2f}  confidence={now_state.get('confidence',5):.2f}")
        print(f"    Total divergence score: {divergence:.2f}")
        if then_failures or now_failures:
            print(f"    Replay failures: {then_failures + now_failures}")
        print()

        if divergence > best_divergence:
            best_divergence = divergence
            best_candidate = (pivot_event, then_state, now_state)

    if best_candidate and best_divergence > 0.5:
        e, then_s, now_s = best_candidate
        print(f"=== BEST CANDIDATE for Demo Script 2 ===")
        print(f"Event:     \"{e['text']}\"")
        print(f"Timestamp: {e['timestamp']}")
        print(f"THEN: {json.dumps(then_s, indent=2)}")
        print(f"NOW:  {json.dumps(now_s, indent=2)}")
        print(f"Divergence: {best_divergence:.2f}")
        print()
        print("If this candidate is meaningful, use it for Demo Script 2.")
        print("If the divergence is too small to be compelling, say so in the rehearsal.")
    else:
        print("=== NO COMPELLING PAST-BELIEF-CHANGING MOMENT FOUND ===")
        print("The pilot data did not produce a strong then-vs-now divergence.")
        print("Per Phase 4 rules: do NOT invent one.")
        print("Report this honestly in the July 31 rehearsal.")

if __name__ == "__main__":
    main()
