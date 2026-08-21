# Phase 5 — Go/No-Go Checklist (Run on Rehearsal Day)

**Instructions:** Ask each pod lead directly. Record their exact answer.
A genuine "no" = cut from the script that day. No negotiation.

---

## Pod Readiness Check

| Pod | Lead | Question Asked | Exact Answer | Go / No-Go |
|---|---|---|---|---|
| **Pod A** — Core State Engine | | "Is your piece ready and working?" | | |
| **Pod B** — Event Understanding | | "Is your piece ready and working?" | | |
| **Pod C** — Event Storage | | "Is your piece ready and working?" | | |
| **Pod D** — Looking Back | | "Is your piece ready and working?" | | |
| **Pod E** — Integration CLI | | "Is your piece ready and working?" | | |

Current code readiness (2026-08-21): Pod A/B/C/D/E integration tests and deterministic
rehearsal pass. Pilot readiness remains **NO** until 5+ real users complete the collection
period; synthetic fixtures cannot satisfy that gate.

---

## Pilot Data Gate

- [ ] At least 5 pilot users submitted events on all 4 days
- [ ] `events.db` contains real events (run: `python3 inspect_pilot_data.py`)
- [ ] `edge-cases-log.md` updated with any real user edge cases found

---

## Phase 4 Gate

- [ ] `inspect_pilot_data.py` was run on the full pilot dataset
- [ ] Demo Script 2 status is one of:
  - [ ] **Written** — real past-belief-changing moment found, cited below
  - [ ] **Honestly absent** — no compelling moment found, rehearsal will say so

*If written, cite the real event(s) here:*
> Event: ___________
> Timestamp: ___________
> THEN state: ___________
> NOW state: ___________

---

## Final Rehearsal Script (Phase 6)

Only include pods marked **Go** above.

Cut list (pods answering "no"):
- [ ] None — all pods go
- [ ] Pod(s) cut: ___________

Adjusted demo command sequence:
```bash
# Step 1 — confirm events.db has real pilot data
python3 inspect_pilot_data.py

# Step 2 — run a live query against a real pilot event date
python3 cli.py query <real_date_from_pilot_data>

# Step 3 — run the full demo command
python3 cli.py demo
```

---

## Rehearsal Result (fill in on July 31)

- Date/time rehearsal run: ___________
- Exit code: ___________
- Output matched expected: [ ] Yes / [ ] No
- Issues found: ___________
- Issues resolved before Aug 1: [ ] Yes / [ ] No / [ ] Deferred
