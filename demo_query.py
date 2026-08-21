"""Fetch events between two dates, and demonstrate checkpoint lookups."""

from event_store import fetch_events_between, fetch_latest_checkpoint_before, get_state_at


def main() -> None:
    events = fetch_events_between(
        "2026-06-08T00:00:00Z",
        "2026-06-15T23:59:59Z",
    )

    print(f"Found {len(events)} event(s):")

    for event in events:
        print(
            f"[{event.happened_at}] "
            f"{event.description} | "
            f"confidence={event.confidence:.2f} | "
            f"change={event.change_data}"
        )

    print()

    as_of = "2026-06-20T00:00:00Z"
    checkpoint = fetch_latest_checkpoint_before(as_of)
    if checkpoint:
        print(
            f"Latest checkpoint before {as_of}: "
            f"checkpoint #{checkpoint.id}, covers events up to id "
            f"{checkpoint.last_event_id} ({checkpoint.event_count} events folded in), "
            f"{checkpoint.size_bytes} bytes"
        )
    else:
        print(f"No checkpoint exists before {as_of} yet.")

    print()
    print(f"Reconstructed state as of {as_of}:")
    state = get_state_at(as_of)
    for key, value in sorted(state.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
