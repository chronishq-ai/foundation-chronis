#!/usr/bin/env python3
"""
chronis-ai — Pod E: Personal Memory Assistant CLI

Commands:
    add-event   Add a new event (in-memory for now, real storage = Pod C)
    query       Ask what the system believed at a given point in time
    demo        Run the full scripted demo scenario automatically

Every function below that says "PLACEHOLDER" stands in for another pod's
real code. Swap the body of each one out for the real import once that
pod's piece lands — the function signature should NOT change, so nothing
else in this file needs to be touched.
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

import event_store
from analyzer import analyze_event
from core.update_state import update_state
from integration_log import log_event
from pod_d import get_belief_then, get_belief_now

# ---------------------------------------------------------------------------
# SHARED CONTRACT — fill this in with what you agreed Day 1 morning
# ---------------------------------------------------------------------------

VARIABLES = {
    # name:                {"range": (low, high), "speed": "fast" | "slow"}
    "mood":                {"range": (0, 10), "speed": "fast"},
    "focus":               {"range": (0, 10), "speed": "fast"},
    "stress":              {"range": (0, 10), "speed": "fast"},
    "confidence":          {"range": (0, 10), "speed": "slow"},
    "trust":               {"range": (0, 10), "speed": "slow"},
    "motivation":          {"range": (0, 10), "speed": "fast"},
    "social_engagement":   {"range": (0, 10), "speed": "slow"},
}

STARTING_STATE = {name: 5.0 for name in VARIABLES}

# File where events are saved between runs — stand-in for Pod C's database
# until the real one is wired in.
EVENTS_FILE = "events.json"
DB_PATH = os.getenv("CHRONIS_DB_PATH", "events.db")


def load_events():
    """Load saved events from disk. Returns an empty list if none exist yet."""
    try:
        with open(EVENTS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        print(f"Warning: {EVENTS_FILE} was corrupted or empty — starting fresh.")
        return []


def save_events(events):
    """Save the full event list to disk, overwriting the previous file."""
    with open(EVENTS_FILE, "w") as f:
        json.dump(events, f, indent=2)


# In-memory event log, loaded from disk at startup — stand-in for Pod C's
# database until it's wired in.
EVENT_LOG = load_events()


# ---------------------------------------------------------------------------
# PLACEHOLDER — Pod A: Core State Engine
# Real version: update_state(current_state, event_signal) -> new_state
# ---------------------------------------------------------------------------

def update_state_placeholder(current_state, event_signal):
    """
    Swapped in real Pod A Core State Engine.
    """
    return update_state(current_state, event_signal)


# ---------------------------------------------------------------------------
# PLACEHOLDER — Pod B: Event Understanding
# Real version: calls the Anthropic API, returns structured JSON.
# ---------------------------------------------------------------------------

def interpret_event_placeholder(event_text):
    """
    Swapped in real Pod B Event Understanding module (analyze_event).
    """
    res = analyze_event(event_text)
    if not isinstance(res, dict) or not isinstance(res.get("signals"), dict):
        raise ValueError("Pod B returned a payload without a signals object")
    return res["signals"]


# ---------------------------------------------------------------------------
# PLACEHOLDER — Pod C: Event Storage
# Real version: SQLite, append-only, add_event() / fetch_events(start, end)
# ---------------------------------------------------------------------------

def add_event_placeholder(text, signal, timestamp=None, persist=True,
                           event_list=None, pilot_id=None, input_id=None,
                           data_kind="unattributed"):
    """
    Swapped in real Pod C Event Storage module (event_store).
    """
    ts_val = timestamp or datetime.now()
    ts_str = ts_val.isoformat() if isinstance(ts_val, datetime) else str(ts_val)
    event = {
        "text": text,
        "timestamp": ts_str,
        "signal": signal,
    }
    if event_list is not None:
        event_list.append(event)
    if persist:
        conf = 0.8
        if isinstance(signal, dict) and signal:
            confs = [v.get("confidence", 0.8) for v in signal.values() if isinstance(v, dict)]
            if confs:
                conf = sum(confs) / len(confs)
        event["id"] = event_store.add_event(
            description=text,
            happened_at=ts_str,
            change_data=signal,
            confidence=conf,
            db_path=DB_PATH,
            pilot_id=pilot_id,
            input_id=input_id,
            data_kind=data_kind,
        )
    return event


def fetch_events_placeholder(start=None, end=None, event_list=None, pilot_id=None,
                             data_kind=None):
    """
    Swapped in real Pod C Event Storage module (fetch_events_between).
    """
    if event_list is not None:
        result = []
        for e in event_list:
            ts = datetime.fromisoformat(e["timestamp"]) if isinstance(e["timestamp"], str) else e["timestamp"]
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            result.append(e)
        return result
    start_str = start.isoformat() if isinstance(start, datetime) else (start or "2000-01-01T00:00:00")
    end_str = end.isoformat() if isinstance(end, datetime) else (end or "2099-12-31T23:59:59")
    db_events = event_store.fetch_events_between(
        start_str, end_str, db_path=DB_PATH, pilot_id=pilot_id, data_kind=data_kind,
    )
    return [
        {
            "text": e.description,
            "timestamp": e.happened_at,
            "signal": e.change_data,
        }
        for e in db_events
    ]


# ---------------------------------------------------------------------------
# PLACEHOLDER — Pod D: Looking Back and Reinterpreting
# Real version: "then" answer (events up to date X) vs "now" answer (all events)
# ---------------------------------------------------------------------------

def query_then_placeholder(target_date, event_list=None, pilot_id=None, data_kind=None):
    """
    Swapped in real Pod D get_belief_then.
    """
    # Normalise target_date to ISO string for timestamp comparison
    td_str = target_date.isoformat() if hasattr(target_date, "isoformat") else str(target_date)
    all_events = fetch_events_placeholder(event_list=event_list, pilot_id=pilot_id,
                                          data_kind=data_kind or ("real" if pilot_id else None))
    return get_belief_then(all_events, td_str, dict(STARTING_STATE), update_state_placeholder)


def query_now_placeholder(target_date, event_list=None, pilot_id=None, data_kind=None):
    """
    Swapped in real Pod D get_belief_now.
    """
    td_str = target_date.isoformat() if hasattr(target_date, "isoformat") else str(target_date)
    all_events = fetch_events_placeholder(event_list=event_list, pilot_id=pilot_id,
                                          data_kind=data_kind or ("real" if pilot_id else None))
    return get_belief_now(all_events, td_str, dict(STARTING_STATE), update_state_placeholder)


# ---------------------------------------------------------------------------
# CLI COMMANDS
# ---------------------------------------------------------------------------

def cmd_add_event(args):
    try:
        if not args.text or not args.text.strip():
            raise ValueError("event text must not be empty")
        if args.data_kind == "real" and (not args.pilot_id or not args.input_id):
            raise ValueError("real pilot submission requires --pilot-id and --input-id")
        timestamp = args.at or datetime.now(timezone.utc).isoformat(timespec="seconds")
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        signal = interpret_event_placeholder(args.text)
        event = add_event_placeholder(
            args.text, signal, timestamp=timestamp, pilot_id=args.pilot_id,
            input_id=args.input_id, data_kind=args.data_kind,
        )
        trace_id = log_event(
            pilot_id=args.pilot_id, input_id=args.input_id, component="pod-e",
            event_type="pilot_event", status="accepted", stage="persisted",
            message="Event understood, state-compatible, and appended to storage.", text=args.text,
            event_id=event.get("id"),
        )
        print(f"Added event: \"{event['text']}\" at {event['timestamp']}")
        print(f"Trace ID: {trace_id}")
        print(json.dumps(signal, indent=2))
    except Exception as exc:
        trace_id = log_event(
            pilot_id=args.pilot_id, input_id=args.input_id, component="pod-e",
            event_type="pilot_event", status="rejected", stage="ingestion",
            error_code=type(exc).__name__, message=str(exc), text=args.text,
        )
        raise SystemExit(f"Event was not stored. Trace ID: {trace_id}. Error: {exc}")


def cmd_query(args):
    try:
        target_date = datetime.fromisoformat(args.date.replace("Z", "+00:00"))
        if len(args.date) == 10:
            target_date = target_date.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
        then_state = query_then_placeholder(target_date, pilot_id=args.pilot_id, data_kind=args.data_kind)
        now_state = query_now_placeholder(target_date, pilot_id=args.pilot_id, data_kind=args.data_kind)
    except Exception as exc:
        trace_id = log_event(
            pilot_id=args.pilot_id, input_id=None, component="pod-e", event_type="query",
            status="rejected", stage="query", error_code=type(exc).__name__,
            message=str(exc), text=args.date,
        )
        raise SystemExit(f"Query failed. Trace ID: {trace_id}. Error: {exc}")

    trace_id = log_event(
        pilot_id=args.pilot_id, input_id=None, component="pod-e", event_type="query",
        status="accepted", stage="query", message="Then/now state replay completed.", text=args.date,
    )
    print(f"\n--- What we believed THEN (as of {args.date}) ---")
    print(json.dumps(then_state, indent=2))

    print(f"\n--- What we believe NOW about that same date ---")
    print(json.dumps(now_state, indent=2))
    print(f"Trace ID: {trace_id}")


def cmd_demo(args):
    """
    Runs a full scripted scenario end to end. Replace this with the actual
    word-for-word demo script once it's written — same shape, real content.
    """
    print("=== chronis-ai DEMO ===\n")

    demo_events = []  # isolated — never touches your real events.json

    # Explicitly synthetic fixtures: this rehearsal path never calls an LLM or
    # persists data, and must never be presented as pilot evidence.
    scenario = [
        ("synthetic: presentation went well", {"mood": {"value": 8, "confidence": .9}, "confidence": {"value": 8, "confidence": .9}}),
        ("synthetic: missed follow-up deadline", {"stress": {"value": 8, "confidence": .9}, "focus": {"value": 3, "confidence": .8}}),
        ("synthetic: later feedback revised the presentation assessment", {"mood": {"value": 2, "confidence": .9}, "confidence": {"value": 2, "confidence": .9}, "stress": {"value": 9, "confidence": .9}}),
    ]

    base_time = datetime(2026, 7, 1)
    reveal_date = None

    for i, (text, signal) in enumerate(scenario):
        ts = base_time + timedelta(days=i)
        add_event_placeholder(text, signal, timestamp=ts, persist=False,
                               event_list=demo_events)
        print(f"[Day {i+1}] Event: \"{text}\"")
        if i == 0:
            reveal_date = ts  # the date we'll re-examine later

    print("\n--- Reveal moment: what did we think about Day 1 then vs now? ---")
    then_state = query_then_placeholder(reveal_date, event_list=demo_events)
    now_state = query_now_placeholder(reveal_date, event_list=demo_events)
    print("THEN:", json.dumps(then_state, indent=2))
    print("NOW: ", json.dumps(now_state, indent=2))
    print("\nSynthetic rehearsal only: use inspect_pilot_data.py for actual pilot observations.")
    log_event(pilot_id="synthetic", input_id="demo-fixture", component="pod-e",
              event_type="rehearsal", status="accepted", stage="completed",
              message="Deterministic synthetic end-to-end rehearsal completed.", text="demo-fixture")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="chronis",
        description="chronis-ai Personal Memory Assistant CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_add = subparsers.add_parser("add-event", help="Add a new event")
    p_add.add_argument("text", help="Plain-text description of the event")
    p_add.add_argument("--pilot-id", help="Pseudonymous pilot identifier; required for real pilot data")
    p_add.add_argument("--input-id", help="Submission ID; required for real pilot data")
    p_add.add_argument("--at", help="ISO-8601 event time; defaults to current UTC time")
    p_add.add_argument("--data-kind", choices=("real", "synthetic"), default="real",
                       help="Label synthetic validation explicitly; real is the default")
    p_add.set_defaults(func=cmd_add_event)

    p_query = subparsers.add_parser(
        "query", help="Query what the system believed at a given date"
    )
    p_query.add_argument(
        "date", help="Target date, e.g. 2026-07-02 or 2026-07-02T12:00:00"
    )
    p_query.add_argument("--pilot-id", required=True,
                         help="Restrict replay to one pseudonymous real pilot")
    p_query.add_argument("--data-kind", choices=("real", "synthetic"), default="real",
                         help="Use synthetic only for explicitly labelled validation data")
    p_query.set_defaults(func=cmd_query)

    p_demo = subparsers.add_parser("demo", help="Run the full scripted demo")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
