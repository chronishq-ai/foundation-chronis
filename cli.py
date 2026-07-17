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
import random
from datetime import datetime, timedelta

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
    Fake version of Pod A's update function.
    event_signal looks like: {"mood": {"delta": 0.3, "confidence": 0.8}, ...}
    Blends current value toward a random-ish "new evidence" value, weighted
    by whether the variable is fast or slow — just enough to look real.
    """
    new_state = dict(current_state)
    for name, signal in event_signal.items():
        if name not in new_state:
            continue
        speed = VARIABLES[name]["speed"]
        blend = 0.6 if speed == "fast" else 0.15
        suggested = new_state[name] + signal.get("delta", 0)
        low, high = VARIABLES[name]["range"]
        suggested = max(low, min(high, suggested))
        new_state[name] = round(
            new_state[name] * (1 - blend) + suggested * blend, 2
        )
    return new_state


# ---------------------------------------------------------------------------
# PLACEHOLDER — Pod B: Event Understanding
# Real version: calls the Anthropic API, returns structured JSON.
# ---------------------------------------------------------------------------

def interpret_event_placeholder(event_text):
    """
    Fake version of Pod B's function.
    Real one sends event_text to Claude and gets back a structured
    {variable: {delta, confidence}} object. Here we just fake plausible
    numbers so the pipeline has something to chew on.
    """
    signal = {}
    for name in VARIABLES:
        signal[name] = {
            "delta": round(random.uniform(-1.5, 1.5), 2),
            "confidence": round(random.uniform(0.4, 0.95), 2),
        }
    return signal


# ---------------------------------------------------------------------------
# PLACEHOLDER — Pod C: Event Storage
# Real version: SQLite, append-only, add_event() / fetch_events(start, end)
# ---------------------------------------------------------------------------

def add_event_placeholder(text, signal, timestamp=None, persist=True,
                           event_list=None):
    """
    persist=True writes to events.json (real add-event usage).
    persist=False keeps it purely in memory (used by `demo`, so a demo run
    never pollutes your real saved events).
    """
    event = {
        "text": text,
        "timestamp": (timestamp or datetime.now()).isoformat(),
        "signal": signal,
    }
    target = EVENT_LOG if event_list is None else event_list
    target.append(event)
    if persist:
        save_events(EVENT_LOG)
    return event


def fetch_events_placeholder(start=None, end=None, event_list=None):
    source = EVENT_LOG if event_list is None else event_list
    if start is None and end is None:
        return list(source)
    result = []
    for e in source:
        ts = datetime.fromisoformat(e["timestamp"])
        if start and ts < start:
            continue
        if end and ts > end:
            continue
        result.append(e)
    return result


# ---------------------------------------------------------------------------
# PLACEHOLDER — Pod D: Looking Back and Reinterpreting
# Real version: "then" answer (events up to date X) vs "now" answer (all events)
# ---------------------------------------------------------------------------

def query_then_placeholder(target_date, event_list=None):
    """What the system believed using only events on/before target_date."""
    events = fetch_events_placeholder(end=target_date, event_list=event_list)
    state = dict(STARTING_STATE)
    for e in events:
        state = update_state_placeholder(state, e["signal"])
    return state


def query_now_placeholder(target_date, event_list=None):
    """What the system believes now, using ALL events, about that same date."""
    events = fetch_events_placeholder(event_list=event_list)  # everything
    state = dict(STARTING_STATE)
    for e in events:
        state = update_state_placeholder(state, e["signal"])
    return state


# ---------------------------------------------------------------------------
# CLI COMMANDS
# ---------------------------------------------------------------------------

def cmd_add_event(args):
    signal = interpret_event_placeholder(args.text)
    event = add_event_placeholder(args.text, signal)
    print(f"Added event: \"{event['text']}\" at {event['timestamp']}")
    print(json.dumps(signal, indent=2))


def cmd_query(args):
    target_date = datetime.fromisoformat(args.date)
    then_state = query_then_placeholder(target_date)
    now_state = query_now_placeholder(target_date)

    print(f"\n--- What we believed THEN (as of {args.date}) ---")
    print(json.dumps(then_state, indent=2))

    print(f"\n--- What we believe NOW about that same date ---")
    print(json.dumps(now_state, indent=2))


def cmd_demo(args):
    """
    Runs a full scripted scenario end to end. Replace this with the actual
    word-for-word demo script once it's written — same shape, real content.
    """
    print("=== chronis-ai DEMO ===\n")

    demo_events = []  # isolated — never touches your real events.json

    scenario = [
        ("felt confident presenting to the team", None),
        ("missed a deadline on the follow-up work", None),
        ("presentation actually went badly and caused ongoing stress",
         None),
    ]

    base_time = datetime(2026, 7, 1)
    reveal_date = None

    for i, (text, _) in enumerate(scenario):
        ts = base_time + timedelta(days=i)
        signal = interpret_event_placeholder(text)
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
    print(
        "\nThe 'now' answer is more trustworthy because it incorporates "
        "later events that revealed the true context behind the early one."
    )


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
    p_add.set_defaults(func=cmd_add_event)

    p_query = subparsers.add_parser(
        "query", help="Query what the system believed at a given date"
    )
    p_query.add_argument(
        "date", help="Target date, e.g. 2026-07-02 or 2026-07-02T12:00:00"
    )
    p_query.set_defaults(func=cmd_query)

    p_demo = subparsers.add_parser("demo", help="Run the full scripted demo")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
