# Sprint 11 — Auxiliary Intelligence Modules

Owner: Kuheli (Team DELIVERY)

Bible traceability: Part 5.12–5.20 (auxiliary modules), Part 5.19 (Inheritance Protocol), Part 5.9 (Second Brain / Decision Replication).

## Scope

Sprint 11 composes eight auxiliary modules on top of Sprint 3/6/9 outputs. No new behavioral-model fitting primitive is introduced. The modules operate on upstream behavioral evidence and policy-owned inputs.

## Status

| Module | Status | What is covered |
|---|---|---|
| `echo_detection.py` | Complete | `m_t` cosine similarity strictly greater than 0.8, regime agreement, opaque social-context agreement, and conversation / behavioral-loop / situational echo types. Missing context does not produce an echo. |
| `weather_forecast.py` | Complete | 45-session gate, user scoping, tomorrow weekday/regime analogue matching, cosine ranking, high-focus/difficult/mixed historical focus flag, transition confidence reduction, and malformed-history rejection. |
| `silence_map.py` | Complete | Attentive, avoidant, and conversational silence using turn-taking, typing, silence duration, and physiological co-signals with finite-input validation. |
| `behavioral_dna.py` | Complete | Admissible Level 3 claims for the requested user, lexicon profile, anonymized social-graph summary validation, and Ed25519 device-key signing/verification when a caller supplies the device signer. |
| `anomaly_detection.py` | Complete | Acute, sustained, and structural anomaly scales, chronological/finite input validation, and an automated non-diagnostic copy validator using the Sprint 9 clinical-term contract. |
| `second_brain.py` | Complete | Deliberately unfiltered modeling-layer snapshot. It only enforces user data ownership; constitutional-policy gating remains outside this module. |
| `inheritance_protocol.py` | Complete | Most-recent admissible Level 3 claim, dependency-injected Sprint 9 constrained-RAG interface, user-owned candidate evidence, citation-chain validation, raw-memory exclusion from the final letter, and Ed25519 device-key signing/verification. |
| `social_graph.py` | Complete | Opaque user-internal cross-session vocal-fingerprint clustering, order-independent connected components, duplicate-session protection, finite-value validation, and no raw fingerprint export. |

## Security and boundary rules

- Device private keys are never generated, stored, or persisted by the export objects. The caller supplies a device-owned `DeviceSigner`.
- Behavioral DNA rejects identity-bearing or raw-fingerprint fields in the social-graph summary.
- Inheritance rejects candidate excerpts belonging to another user and rejects citation IDs outside the supplied evidence set.
- Second Brain performs no admissibility or constitutional-policy filtering. Its only filtering is user ownership for data correctness.
- Social Graph returns opaque cluster IDs and session IDs only; it never infers a person's name or identity.
- Anomaly copy must pass `validate_anomaly_copy()` before it can be surfaced as user-facing copy.

## Shared surrogate integration

`tests/test_sprint11_integration.py` now exercises all eight Sprint 11 modules using the same surrogate `user_id`, including signed Behavioral DNA and a signed Behavioral Letter. The integration test also verifies that the final letter does not contain raw session-excerpt text.

## Definition of Done

1. All eight auxiliary modules run against one shared surrogate profile without introducing a new modeling primitive.
2. Echo Detection requires `m_t` similarity > 0.8, matching regime, and matching opaque social context, and classifies conversation, behavioral-loop, and situational echoes.
3. Weather Forecast uses the 45-session gate and historical analogue matching for tomorrow's context, including explicit high-focus and difficult-day flags.
4. Silence Map covers attentive, avoidant, and conversational silence and rejects invalid/non-finite evidence.
5. Behavioral DNA contains active admissible Level 3 claims, an anonymized social-graph summary, a lexicon profile, and supports real Ed25519 device-key signatures.
6. Anomaly Detection covers acute, sustained, and structural scales and provides an automated check rejecting diagnostic/medical language from user-facing anomaly copy.
7. Social Graph remains opaque and user-internal, with no cross-user mixing or raw fingerprint export.
8. Second Brain remains unfiltered at the modeling layer; constitutional-policy gating is outside the module.
9. Inheritance exports the Behavioral DNA model through the constrained-RAG generator boundary, excludes raw memories from the final letter, validates citations, and supports real device-key signatures.
10. Cross-user evidence, malformed/non-finite vectors, duplicate sessions/timestamps, and invalid signing/citation boundaries have regression coverage.
11. The complete Sprint 11 regression suite passes.

## Validation

```text
python -m pytest -q
python -m compileall -q .
git diff --check
```

Current regression result: **94 passed**.

The only external integration point is the real Sprint 9 constrained-RAG generator: this repository keeps it dependency-injected so Sprint 11 does not duplicate Sprint 9 modeling/generation logic.
