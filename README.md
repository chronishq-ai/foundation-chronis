# Sprint 11 -- Auxiliary Intelligence Modules

Owner: Kuheli (Team DELIVERY)

Bible traceability: Part 5.12-5.20 (auxiliary modules), Part 5.19 (Inheritance Protocol),

Part 5.9 (Second Brain / Decision Replication)

## What this does

Eight secondary features composed on top of existing Sprint 3/6/9 output --

no new modeling primitives, pure composition. See project-level directive

for full module descriptions.

## Status per module

| Module | Status | Notes |
|---|---|---|
| `echo_detection.py` | Done, tested (8/8 passing) | Core echo detection logic is complete. Echo TYPE classification (conversational/behavioral-loop/situational) remains dependent on social-context data from FOUNDRY. Regime-label match is used as a partial context proxy in the meantime. |
| `weather_forecast.py` | Done, tested (19/19 passing) | Historical analogue forecast using `m_t` cosine similarity, same-weekday and compatible-regime matching, 45-session minimum, cold-start gating, transition-confidence reduction, and optional upstream evidence. |
| `silence_map.py` | Done, tested (10/10 passing) | Classifies silence as attentive, avoidant, or conversational using turn-taking and physiological co-signals. Includes bounded confidence and interpretable explanations. |
| `behavioral_dna.py` | Done, tested (9/9 passing) | Filters on level==LEVEL_3 AND admissible AND correct `user_id`. Lexicon profile / social graph accepted as optional params (None until those modules exist). Signature/is_signed intentionally left unsigned -- no real signing infra exists yet in this sprint's scope. |
| `anomaly_detection.py` | Done, tested (9/9 passing) | Real bug found+fixed during review: mean-based baseline was getting dragged by the anomalies themselves; switched to median. Reuses Mansi's clinical-terminology filter via `clinical_terms.py` (temporary local copy, see that file's docstring). |
| `second_brain.py` | Done, tested (4/4 passing) | Deliberately UNFILTERED by design -- includes every claim regardless of level/admissibility. Only scopes to the requested `user_id` (data correctness, not a gate). Gating logic belongs to the constitutional-policy layer, not here. |
| `inheritance_protocol.py` | Done, tested (6/6 passing) | Reuses Mansi's `generate_insight()` via dependency injection (real function not importable yet -- see file docstring). Picks the most recently created admissible Level 3 claim. Raises `NoEligibleClaimError` for cold-start users with zero Level 3 claims, and correctly propagates (does not swallow) the real generator's near-miss-required error. Never fakes a signature. |
| `social_graph.py` | Done, tested (11/11 passing) | Builds opaque, user-scoped cross-session vocal-fingerprint clusters using cosine similarity. Prevents cross-user mixing and does not infer names or identities. |

## Open upstream dependencies

See `upstream_interfaces.py` for details.

- FOUNDRY: social context per session (Echo Detection needs this for full TYPE classification)

- FOUNDRY: turn-taking + PPG sample format (Silence Map). The local Sprint 11 interface and composition logic are implemented and tested.

- FOUNDRY: vocal-fingerprint clustering sample format (Social Graph). The local Sprint 11 interface and clustering logic are implemented and tested.

- Unclear owner: "lexicon profile" field for Behavioral DNA -- not found in

  any package received so far (Sprint 2, 7, or 9). Needs a follow-up ask.

## Definition of Done

1. Every auxiliary module runs cleanly against the same shared surrogate

   profile (`tests/fixtures/synthetic_user_profile.py`) with no new

   modeling assumptions introduced.

2. Anomaly Detection output copy is verified free of diagnostic/medical

   language via an automated string check (reusing Mansi's

   `contains_clinical_terminology`).

3. Weather Forecast, Silence Map, and Social Graph have dedicated

   automated test coverage.

4. Second Brain and Inheritance Protocol retain their intentional

   policy-layer boundaries and are covered by their dedicated tests.

5. Cross-user data is not mixed by Social Graph or Weather Forecast.

6. Invalid/non-finite behavioral evidence is rejected by the relevant

   module instead of silently producing a result.

7. The complete Sprint 11 regression suite passes.

## Test status

Current local Sprint 11 regression result:

    77 passed

Module-level tests:

    Echo Detection:       8 passed
    Weather Forecast:    19 passed
    Silence Map:         10 passed
    Behavioral DNA:       9 passed
    Anomaly Detection:    9 passed
    Second Brain:         4 passed
    Inheritance Protocol: 6 passed
    Social Graph:        11 passed

Integration tests:

    Sprint 11 integration: 1 passed

Validation commands:

    python -m pytest -q

    ppython -m compileall -q .

    git diff --check

## Current Sprint 11 Result

Sprint 11 implementation and regression testing are complete for the

locally available interfaces and surrogate data.

The remaining external dependency is FOUNDRY social-context data required

for the full Echo TYPE classification and final upstream-data validation.

No new modeling primitive is introduced by the Sprint 11 auxiliary modules.