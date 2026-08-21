"""
Pod D — Looking Back and Reinterpreting (portable module).

Extracted from POD_D_final (1).py (branch: origin/POD-D-lookback-revision-final).
Colab-specific imports and file-upload code removed; functions are standalone.
"""
from datetime import datetime, timezone


def _timestamp(value):
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise TypeError("timestamps must be ISO-8601 strings or datetime instances")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

def filter_events_up_to_date(events, target_date):
    """Keep only events whose timestamp <= target_date (string or datetime compare)."""
    relevant = []
    for event in events:
        if _timestamp(event["timestamp"]) <= _timestamp(target_date):
            relevant.append(event)
    return relevant


def sort_events_by_time(events):
    """Bubble-sort events by timestamp (ascending). Preserves original list."""
    return sorted(events, key=lambda event: _timestamp(event["timestamp"]))


def get_belief_then(events, target_date, initial_state, update_function):
    """
    What the system believed using only events on/before target_date.
    events      — list of {text, timestamp, signal} dicts
    target_date — ISO string or datetime
    initial_state — dict of variable -> float starting values
    update_function — callable(state, signal) -> new_state
    """
    relevant = filter_events_up_to_date(events, target_date)
    ordered  = sort_events_by_time(relevant)
    state = dict(initial_state)
    for e in ordered:
        state = update_function(state, e["signal"])
    return state


def get_belief_now(events, target_date, initial_state, update_function):
    """
    What the system believes NOW using ALL events (target_date is ignored
    for filtering but kept in the signature so callers are symmetric).
    """
    ordered = sort_events_by_time(events)
    state = dict(initial_state)
    for e in ordered:
        state = update_function(state, e["signal"])
    return state
