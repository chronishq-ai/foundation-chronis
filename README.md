# Sprint 15 — Observer Effect

Days 43–45. Claim-surfacing index + 30-day influence flags + Active
Transition planted-profile closure.

**This mitigates, does not solve, the Observer Effect**, per the Bible's
own framing (Part 7.8). Showing a claim can still change what happens
next. We only refuse to treat a change inside the 30-day window as
independent evidence for aspiration. MP-13 stays permanently open by
design; it is not a target for future closure.

## What this does

- **Queryable surfacing index** of every Level 1–3 claim that actually
  reached the user (claim id, domain, divergence type, timestamp).
- **30-day check** on behavioral attractor changes *and* narrative
  regime shifts. Flag: `potentially_claim_influenced`.
- **Read-time exclusion**: aspiration scoring consults the index. A
  missing flag on the event is not a bypass.
- **20+ planted profiles per type**, including Active Transition with a
  designed lag recovered from rate-of-change correlation. All four types
  must independently clear >75%. Logged to MLflow.
- **180-day cold-start + Mirror stage 0/1 silence** re-run with this
  logging on. The flag is never product copy.

## What this does not do

- Does not stop the person from reacting to a claim.
- Does not replace Sprint 8's real Fisher/Granger type scores (those
  engines are not in this intern zip). The planted-profile scorer is a
  stand-in that implements the *shape* of the four types and the lag
  check; it is not a claim of external validity (MP-01 / MP-05).
