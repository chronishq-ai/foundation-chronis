# Pilot User Guide — chronis-ai v0.2 Pilot Week

Thank you for participating. This guide is everything you need.

---

## What You Are Doing

For the next 4 days, once per day, you will type 1–3 short sentences describing
real things that happened to you that day. That's it.

- No scripts, no invented events, no "what should I write?"
- Just what actually happened to you today.
- It takes less than 2 minutes per day.

---

## How to Submit an Event

Open a terminal, navigate to the project folder, and run:

```bash
python3 cli.py add-event --pilot-id pilot-01 --input-id day-1 "your event in plain English"
```

### Examples of real events (these are just format examples — write YOUR own):
```bash
python3 cli.py add-event "had a tough code review that went on much longer than expected"
python3 cli.py add-event "finished the report I'd been stuck on for two days"
python3 cli.py add-event "argument with a teammate about project direction, still unresolved"
```

### What counts as an event:
- Something that actually happened today
- Something that affected your mood, stress, focus, confidence, trust, motivation, or social energy — even slightly
- Can be work, personal, anything real

### What does NOT count:
- Summaries of your whole week
- Things you wish had happened
- Anything invented or written to "help the demo"

---

## Schedule

| Day | Date | Action |
|---|---|---|
| Day 1 | (today) | Type at least 1 real event |
| Day 2 | (tomorrow) | Type at least 1 real event |
| Day 3 | +2 days | Type at least 1 real event |
| Day 4 | +3 days | Type at least 1 real event |

---

## If Something Breaks

If you get an error message, copy the full error and send it to the Pod E lead.
Do not retry silently — the error is useful data.

If the tool asks you something unexpected, screenshot it and send it.

---

## What Happens to Your Input

- Your exact words are stored verbatim in an append-only database (nothing is edited or deleted).
- Operational logs record a pseudonymous pilot ID, input ID, stage, status, and trace ID. They do not copy your raw wording.
- The system infers how your event affected 7 psychological state variables
- After Day 4, the data will be used to find a real instance of the "past-belief-changing" pattern for the demo

---

## Questions

Contact the Pod E lead directly.
