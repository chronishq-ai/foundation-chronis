"""
Pod D — Looking Back and Reinterpreting (portable module).

Extracted from POD_D_final (1).py (branch: origin/POD-D-lookback-revision-final).
Colab-specific imports and file-upload code removed; functions are standalone.
"""
from datetime import datetime, timezone
import math


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


def pull_toward_later_evidence(old_value, old_spread, new_value, new_spread):
    """
    Pulls a past estimate toward what later evidence implies it should have been.
    Uses inverse-variance weighting: weight = 1 / spread^2
    Lower spread = higher confidence = more influence on the blended result.
    """
    old_var = max(old_spread ** 2, 0.0001)
    new_var = max(new_spread ** 2, 0.0001)

    old_weight   = 1.0 / old_var
    new_weight   = 1.0 / new_var
    total_weight = old_weight + new_weight

    pulled_value  = (old_value * old_weight + new_value * new_weight) / total_weight
    pulled_spread = math.sqrt(1.0 / total_weight)

    return pulled_value, pulled_spread


def get_belief_now(events, target_date, initial_state, update_function):
    """
    Real backward smoothing -- not a forward re-run with more events.
    Step 1: Get the 'then' state (what the system believed at target_date).
    Step 2: Run later events forward from 'then' to get an 'implied' state.
    Step 3: Pull 'then' toward 'implied' using spread-weighted blending.
    """
    # Step 1: forward filter up to target_date (original function, unchanged)
    then_state = get_belief_then(events, target_date, initial_state, update_function)

    # Step 2: collect events AFTER target_date and run them forward
    later_events = []
    for event in events:
        if _timestamp(event["timestamp"]) > _timestamp(target_date):
            later_events.append(event)
    later_events = sort_events_by_time(later_events)

    implied_state = {}
    for var, data in then_state.items():
        implied_state[var] = {
            "value":  data["value"],
            "spread": initial_state[var]["spread"],  # reset to initial spread so later events move it
        }

    for event in later_events:
        implied_state = update_function(implied_state, event["signal"])

    # Step 3: pull 'then' toward 'implied' using inverse-variance weighting.
    smoothed_state = {}
    for var in then_state:
        old_val = then_state[var]["value"]
        old_spr = initial_state[var]["spread"]   # real spread as THEN uncertainty
        new_val = implied_state[var]["value"]
        new_spr = implied_state[var]["spread"]

        pulled_val, pulled_spr = pull_toward_later_evidence(old_val, old_spr, new_val, new_spr)
        smoothed_state[var] = {"value": pulled_val, "spread": pulled_spr}

    return smoothed_state
