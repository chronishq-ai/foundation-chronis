# Sprint 11 -- Auxiliary Intelligence Modules

Owner: Kuheli (Team DELIVERY)
Bible traceability: Part 5.12-5.20 (auxiliary modules), Part 5.19 (Inheritance Protocol),
Part 5.9 (Second Brain / Decision Replication)

## What this does

Six secondary features composed on top of existing Sprint 3/6/9 output --
no new modeling primitives, pure composition. See project-level directive
for full module descriptions.

## Status per module

| Module | Status | Notes |
|---|---|---|
| `echo_detection.py` | Core detection logic done, tested (8/8 passing) | Echo TYPE classification (conversational/behavioral-loop/situational) is BLOCKED -- needs social-context data from FOUNDRY, not yet received. Regime-label match used as a partial context proxy in the meantime. |
| `weather_forecast.py` | Not started | Needs FOUNDRY's social-context/day-of-week timestamp join. |
| `silence_map.py` | Not started | Needs turn-taking + PPG sample from FOUNDRY. |
| `behavioral_dna.py` | Done, tested (9/9 passing) | Filters on level==LEVEL_3 AND admissible AND correct user_id. Lexicon profile / social graph accepted as optional params (None until those modules exist). Signature/is_signed intentionally left unsigned -- no real signing infra exists yet in this sprint's scope. |
| `anomaly_detection.py` | Done, tested (9/9 passing) | Real bug found+fixed during review: mean-based baseline was getting dragged by the anomalies themselves; switched to median. Reuses Mansi's clinical-terminology filter via clinical_terms.py (temporary local copy, see that file's docstring). |
| `second_brain.py` | Done, tested (4/4 passing) | Deliberately UNFILTERED by design -- includes every claim regardless of level/admissibility. Only scopes to the requested user_id (data correctness, not a gate). Gating logic belongs to the constitutional-policy layer, not here. |
| `inheritance_protocol.py` | Done, tested (6/6 passing) | Reuses Mansi's generate_insight() via dependency injection (real function not importable yet -- see file docstring). Picks the most recently created admissible Level 3 claim. Raises NoEligibleClaimError for cold-start users with zero Level 3 claims, and correctly propagates (does not swallow) the real generator's near-miss-required error. Never fakes a signature. |
| `social_graph.py` | Not started | Needs vocal-fingerprint clustering sample from FOUNDRY. |
| `second_brain.py` | Not started | Scaffolding only, deliberately unfiltered per directive -- gating is the policy layer's job, not ours. |
| `inheritance_protocol.py` | Not started | Depends on `behavioral_dna.py`. Reuses Mansi's `generate_insight()` -- requires at least one `is_near_miss=True` excerpt or it raises. |

## Open upstream dependencies (see upstream_interfaces.py for details)

- FOUNDRY: social context per session (Echo Detection needs this for type classification)
- FOUNDRY: turn-taking + PPG sample format (Silence Map)
- FOUNDRY: vocal-fingerprint clustering sample format (Social Graph)
- Unclear owner: "lexicon profile" field for Behavioral DNA -- not found in
  any package received so far (Sprint 2, 7, or 9). Needs a follow-up ask.

## Definition of Done (from directive)

1. Every auxiliary module runs cleanly against the same shared surrogate
   profile (`tests/fixtures/synthetic_user_profile.py`) with no new
   modeling assumptions introduced.
2. Anomaly Detection output copy is verified free of diagnostic/medical
   language via an automated string check (reusing Mansi's
   `contains_clinical_terminology`).