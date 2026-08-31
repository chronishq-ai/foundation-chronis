# CHRONIS — Sprints 10, 11 & 12
## Phase 1 + Phase 2 Delivery Summary

**Project area:** CHRONIS AI/ML  
**Scope:** Sprints 10, 11 and 12  
**Ownership:**
- **Sprint 10 — Mayank**
- **Sprint 11 — Kuheli**
- **Sprint 12 — Mayank**

This repository contains the Sprint 10–12 implementation work, including the
Phase 1 functionality and the Phase 2 hardening/fix work completed against the
project requirements.

The goal of Phase 2 was not to redesign the system. It was to review the
existing implementation, close concrete correctness and safety gaps, preserve
the intended architecture, and add regression coverage for the fixes.

---

# 1. At a Glance

| Sprint | Owner | Main Deliverable | Phase 1 | Phase 2 |
|---|---|---|---|---|
| **Sprint 10** | **Mayank** | Cold Start Compass & Threshold Calibration II | Core D* estimation and Cold Start state-machine flow | Correct calendar-day window conversion, Stage-2 isolation, stronger HSSM validation and safer audit logging |
| **Sprint 11** | **Kuheli** | Auxiliary Intelligence Modules | Echo Detection, Behavioral DNA, Anomaly Detection, Second Brain and Inheritance Protocol | Weather Forecast, Silence Map, Social Graph, integration coverage and additional hardening |
| **Sprint 12** | **Mayank** | The Mirror / Insight Generation | Grounded daily insight pipeline, tone, feedback, archive and specificity controls | Stronger specificity blocking, mandatory citation anchoring and canonical sentence splitting |

---

# 2. Sprint 10 — Mayank

## Phase 1 — Core Cold Start & Threshold Calibration

Sprint 10 establishes the Cold Start Compass on top of the fitted behavioral
HSSM.

### Achieved

- Computes the slow-phase expected duration **D\*** from the fitted HSSM's
  log-normal duration parameters.
- Uses the exact log-normal mean:

  `D* = exp(dur_mu + dur_sigma² / 2)`

- Replaces the earlier fixed observation-window placeholder with a window
  derived from D\*.
- Maintains the staged Cold Start state machine.
- Keeps Stage 0 inference-free.
- Provides the downstream Cold Start gate used by later claim/insight layers.
- Adds MLflow-related tracking for the calibration values where permitted by
  the stage.

## Phase 2 — Correctness & Safety Hardening

The Phase 2 review closed several concrete issues in the original implementation.

### Achieved

**1. D\* unit conversion fixed**

D\* is an expected number of sessions, while Cold Start stage boundaries are
calendar days.

The observation window is therefore converted using session frequency:

`observation_window_days = 2 * D* / sessions_per_day`

This prevents users with different session frequencies from receiving the
same calendar-day window incorrectly.

**2. Stage-2 estimates are internal-only**

Stage 2 may fit the HSSM internally, but those estimates must not leak into
downstream output or external audit trails.

The implementation now:

- marks Stage 2 as `internal_estimates_only`;
- prevents the state from being safely handed downstream;
- redacts `hssm_fitted` at the Stage-2 boundary;
- skips MLflow logging for Stage-2 internal estimates;
- provides an explicit `assert_downstream_safe()` contract.

**3. HSSM input validation strengthened**

The wiring layer now checks:

- the object is an `HSSMFit`;
- at least two regimes exist;
- the slow regime exists in the fitted duration parameters;
- the fit has a valid user ID;
- the fitted model has converged before it is used.

**4. Regression coverage expanded**

Phase 2 adds tests around the corrected observation-window units,
Stage-2 isolation, and HSSM structural validation.

---

# 3. Sprint 11 — Kuheli

## Phase 1 — Auxiliary Intelligence Modules

Sprint 11 is an evidence-composition layer over the existing Sprint 3/6/9
behavioral outputs. It does not introduce a new modeling primitive.

### Achieved modules

### Echo Detection

- Detects recurring behavioral/conversational echoes.
- Uses the existing behavioral representation.
- Includes the required similarity-based core detection.
- Echo TYPE classification remains dependent on the required FOUNDRY
  social-context data where that upstream data is unavailable.

### Behavioral DNA

- Exports active Level-3 behavioral claims.
- Applies user and admissibility filtering.
- Supports the lexicon/social-graph inputs as optional upstream data.
- Does not fake cryptographic signing when real signing infrastructure is
  unavailable.

### Anomaly Detection

- Supports acute, sustained and structural anomaly scales.
- Uses the project's clinical-terminology safety filter.
- Baseline handling was reviewed and corrected to avoid anomalies
  contaminating their own baseline.

### Second Brain

- Provides the Decision Replication scaffolding.
- Remains deliberately unfiltered at the modeling layer.
- Scopes data to the requested user.
- Leaves gating to the constitutional-policy layer as required by the design.

### Inheritance Protocol

- Exports Behavioral DNA rather than raw memories.
- Selects an eligible recent Level-3 claim.
- Handles cold-start/no-eligible-claim cases explicitly.
- Reuses the insight-generation dependency rather than inventing a second
  generation path.
- Does not fabricate signatures.

## Phase 2 — Completion & Hardening

### Weather Forecast

- Historical-analogue behavioral forecast.
- Requires the 45-session minimum.
- Uses same-weekday and compatible-regime historical matching.
- Uses cosine similarity over `m_t`.
- Respects the Cold Start gate.
- Reduces confidence during regime transitions.
- Produces energy, social-engagement, stress and productivity context.
- Accepts optional upstream evidence such as PPG and social-pattern signals.
- Validates dimensions, finite values and zero vectors.

### Silence Map

Classifies silence into:

- **attentive**
- **avoidant**
- **conversational**

The classification combines turn-taking state with physiological
co-signals and exposes bounded confidence plus an interpretable explanation.

### Social Graph

- Performs cross-session vocal-fingerprint clustering.
- Uses cosine similarity.
- Keeps graph data strictly scoped to the requested user.
- Prevents cross-user mixing.
- Does not infer names or real-world identities.
- Keeps the representation opaque/user-internal.

### Integration & Regression Coverage

Sprint 11 includes dedicated tests for the new modules and a Sprint 11
integration test.

Current local Sprint 11 regression result during Phase 2 review:

`77 passed`

The remaining external limitation is the availability of the real FOUNDRY
social-context / fingerprint data required for full upstream validation of
the affected interfaces.

---

# 4. Sprint 12 — Mayank

## Phase 1 — The Mirror

Sprint 12 provides the daily insight-generation experience on top of the
Claims Engine and Cold Start gate.

### Achieved

### Insight Generation

- Generates grounded, second-person daily insights.
- Uses admitted behavioral claims and supporting session excerpts.
- Requires the grounded-generation evidence structure.
- Maintains citation-chain information for generated sentences.

### Cold Start Integration

- Respects the Sprint 10 Cold Start state.
- Prevents Mirror output during the silent early stages.
- Keeps downstream generation behind the appropriate evidence gates.

### Specificity Linter

- Blocks generic coach-speak and unsupported generic emotional statements.
- Enforces the insight word-count range.
- Checks sentence-level evidence anchoring.

### Tone Calibration

Supports the defined tone modes:

- DIRECT
- REFLECTIVE
- WARM

Tone instructions are applied at generation time rather than by silently
rewriting the generated text afterward.

### Feedback Loop

- Incorporates user feedback into adaptive thresholds.
- Supports NOT_YET / TOO_SOON behavior.
- Supports domain-level suppression where required.

### Archive

- Provides an append-only insight archive.
- Prevents duplicate insertion.
- Supports retrieval/search over archived insight records.

### TTS

- Uses the defined synthesized/stub interface.
- Does not pretend that generated audio is a recording of the user.

## Phase 2 — Production Hardening

### Specificity rules strengthened

The generic-output blocker was expanded to catch additional bypass patterns,
including:

- indirect emotional attribution;
- additional appearance/state verbs;
- implicit wellness recommendations;
- vague difficulty references;
- additional unanchored evaluations;
- unsolicited care directives.

### Citation anchoring made mandatory

The specificity linter no longer permits callers to omit the citation chain.

Every sentence must resolve to a citation anchor.

This closes the bypass where an uncited insight could otherwise reach the
linter without being checked.

### Canonical sentence splitting

Duplicate sentence-tokenization logic was removed from the linter.

The linter now uses the canonical sentence-splitting implementation already
used by the grounded-generation citation chain.

This keeps sentence indexing consistent across generation, citation and
specificity validation.

---

# 5. Cross-Sprint Architecture

The three sprints fit together as a staged pipeline:

```text
Sprint 10
Cold Start Compass
        │
        │  evidence/stage gate
        ▼
Sprint 11
Auxiliary Intelligence
        │
        │  behavioral evidence / supporting context
        ▼
Sprint 12
The Mirror
        │
        ▼
Grounded, cited insight output
```

The implementation intentionally keeps responsibilities separated:

- **Sprint 10** controls when inference and claims are allowed to progress.
- **Sprint 11** composes additional behavioral evidence and auxiliary
  intelligence.
- **Sprint 12** turns admissible evidence into the user-facing Mirror
  experience.

---

# 6. Testing & Validation

## Sprint 10

Phase 1 and Phase 2 include dedicated Cold Start and wiring tests covering:

- D\* calculation;
- observation-window calculation;
- Cold Start stages;
- Stage-2 internal-only behavior;
- downstream safety;
- HSSM structural validation;
- wiring behavior.

The Sprint 10 phase2 README records:

`57/57 Sprint 10 tests passing`

## Sprint 11

Current Phase 2 regression result:

`77 passed`

Coverage includes:

- Echo Detection;
- Weather Forecast;
- Silence Map;
- Behavioral DNA;
- Anomaly Detection;
- Second Brain;
- Inheritance Protocol;
- Social Graph;
- Sprint 11 integration.

## Sprint 12

The Phase 1 repository records:

`178/178 tests passing`

including the Sprint 12 Mirror suite and its upstream Sprint 9/10 coverage.

Phase 2 additionally adds/updates tests around the strengthened specificity
and citation contracts.

---

# 7. Phase 2 Review Principles

The Phase 2 work follows four simple rules:

1. **Preserve the existing architecture.**
   Fixes are targeted rather than replacing the original design.

2. **Enforce important requirements in code.**
   Safety and correctness requirements are not left only as README
   statements.

3. **Test the failure paths, not only the happy path.**
   Invalid dimensions, invalid model objects, stage leakage, missing
   citations, generic output bypasses and cross-user mixing are explicitly
   considered.

4. **Do not fabricate unavailable upstream infrastructure.**
   Where FOUNDRY data, signing infrastructure, or another cross-sprint
   dependency is genuinely unavailable, the boundary is documented rather
   than simulated as complete.

---

# 8. Ownership Summary

| Work | Owner |
|---|---|
| Sprint 10 — Cold Start Compass & Threshold Calibration II | **Mayank** |
| Sprint 11 — Auxiliary Intelligence Modules | **Kuheli** |
| Sprint 12 — The Mirror | **Mayank** |

This folder therefore represents the combined Sprint 10–12 delivery, while
keeping ownership and module boundaries explicit.

---

# 9. Final Status

### Sprint 10 — Mayank
**Phase 1:** Core Cold Start and D\* calibration completed.  
**Phase 2:** Unit conversion, Stage-2 isolation, validation and audit-path
hardening completed.

### Sprint 11 — Kuheli
**Phase 1:** Core auxiliary modules completed.  
**Phase 2:** Weather Forecast, Silence Map, Social Graph, integration
coverage and hardening completed.  
**Local regression:** `77 passed`.

### Sprint 12 — Mayank
**Phase 1:** The Mirror pipeline and supporting components completed.  
**Phase 2:** Specificity, citation and sentence-tokenization safeguards
strengthened.

## Overall

The repository contains the Phase 1 implementations and the corresponding
Phase 2 fixes/hardening for Sprints 10, 11 and 12, with module ownership
clearly separated between **Mayank (Sprints 10 & 12)** and **Kuheli (Sprint
11)**.

Where Phase 2 requirements depend on unavailable upstream systems or
cross-sprint infrastructure, those dependencies remain explicitly identified
instead of being represented as falsely completed.
