"""Insert fictional sample events into the database and build checkpoints."""

from pathlib import Path

from event_store import (
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_DB_PATH,
    add_event,
    create_checkpoint,
    estimate_storage_for_years,
    get_storage_stats,
    initialize_database,
)


# Spread across ~7 weeks (instead of ~3) so several automatic checkpoints
# are created along the way, not just one.
SAMPLE_EVENTS = [
    ("User created a workspace", "2026-06-01T09:15:00Z",
     {"workspace_id": "ws-1001", "status": "created"}, 0.99),

    ("Notification preferences were detected", "2026-06-02T11:30:00Z",
     {"email_notifications": True, "push_notifications": False}, 0.88),

    ("First project was added", "2026-06-03T14:10:00Z",
     {"project_id": "prj-201", "project_count_change": 1}, 0.98),

    ("Project deadline was recorded", "2026-06-04T10:00:00Z",
     {"project_id": "prj-201", "deadline": "2026-06-18"}, 0.96),

    ("User completed onboarding", "2026-06-05T16:45:00Z",
     {"onboarding_status": "complete"}, 0.99),

    ("Weekly activity level increased", "2026-06-07T18:20:00Z",
     {"previous_level": "low", "new_level": "medium"}, 0.81),

    ("A task was marked complete", "2026-06-08T09:05:00Z",
     {"task_id": "task-301", "status": "completed"}, 0.99),

    ("A project risk was identified", "2026-06-09T13:40:00Z",
     {"project_id": "prj-201", "risk": "schedule delay"}, 0.84),

    ("Suggested reminder was accepted", "2026-06-10T08:50:00Z",
     {"reminder_id": "rem-401", "accepted": True}, 0.93),

    ("User added a second project", "2026-06-11T12:25:00Z",
     {"project_id": "prj-202", "project_count_change": 1}, 0.99),

    ("Preferred working time was inferred", "2026-06-12T19:10:00Z",
     {"preferred_period": "evening"}, 0.76),

    ("Project priority changed", "2026-06-14T15:00:00Z",
     {"project_id": "prj-202", "old_priority": "medium", "new_priority": "high"}, 0.91),

    ("A milestone was reached", "2026-06-15T10:35:00Z",
     {"project_id": "prj-201", "milestone": "prototype complete"}, 0.97),

    ("A collaboration pattern was detected", "2026-06-16T17:50:00Z",
     {"frequent_collaborator": "user-88", "interaction_count": 7}, 0.79),

    ("User postponed a task", "2026-06-18T09:45:00Z",
     {"task_id": "task-322", "new_date": "2026-06-21"}, 0.95),

    ("Weekly activity level increased again", "2026-06-20T20:15:00Z",
     {"previous_level": "medium", "new_level": "high"}, 0.86),

    ("A new project note was saved", "2026-06-22T11:05:00Z",
     {"project_id": "prj-202", "note_type": "decision"}, 0.99),

    ("A deadline risk was resolved", "2026-06-24T14:30:00Z",
     {"project_id": "prj-201", "status": "resolved"}, 0.94),

    ("A third project was added", "2026-06-26T09:00:00Z",
     {"project_id": "prj-203", "project_count_change": 1}, 0.98),

    ("Project budget was recorded", "2026-06-27T13:15:00Z",
     {"project_id": "prj-203", "budget_usd": 15000}, 0.9),

    ("A recurring reminder was created", "2026-06-29T08:30:00Z",
     {"reminder_id": "rem-402", "recurrence": "weekly"}, 0.92),

    ("Weekly activity level decreased", "2026-07-01T19:40:00Z",
     {"previous_level": "high", "new_level": "medium"}, 0.8),

    ("A milestone was reached", "2026-07-03T10:20:00Z",
     {"project_id": "prj-202", "milestone": "design review"}, 0.96),

    ("User archived a project", "2026-07-06T16:00:00Z",
     {"project_id": "prj-201", "status": "archived"}, 0.99),
]


def seed_database(
    db_path: str | Path = DEFAULT_DB_PATH,
    reset: bool = False,
) -> None:
    db_path = Path(db_path)

    if reset and db_path.exists():
        db_path.unlink()

    initialize_database(db_path)

    for description, happened_at, change_data, confidence in SAMPLE_EVENTS:
        add_event(
            description=description,
            happened_at=happened_at,
            change_data=change_data,
            confidence=confidence,
            db_path=db_path,
        )

    # Flush any remaining events (fewer than DEFAULT_CHECKPOINT_INTERVAL
    # since the last automatic checkpoint) into one final checkpoint, so
    # the whole history is covered.
    create_checkpoint(db_path)

    print(f"Inserted {len(SAMPLE_EVENTS)} events into {db_path}")

    stats = get_storage_stats(db_path)
    print(f"Created {stats['checkpoint_count']} checkpoints")
    print(f"Checkpoint size: {stats['avg_checkpoint_bytes']:.0f} bytes (average)")

    # Roughly one event every ~1.75 days in the sample data above;
    # project forward assuming a similar pace continues for 10 years.
    span_days = 35
    events_per_day = len(SAMPLE_EVENTS) / span_days
    projection = estimate_storage_for_years(
        years=10,
        events_per_day=events_per_day,
        checkpoint_interval=DEFAULT_CHECKPOINT_INTERVAL,
        db_path=db_path,
    )
    print(f"Estimated storage for 10 years: {projection['total_mb']:.2f} MB")


if __name__ == "__main__":
    seed_database(reset=True)
