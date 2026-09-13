# Sprint 8 + Sprint 9 — Divergence Engine & Claims Engine

Standalone reference implementation of the CHRONIS AI/ML directive's Sprint 8
(The Divergence Engine, Days 22-24) and Sprint 9 (Claims Engine & Grounded
Generation, Days 25-27), built against **mocked upstream interfaces**
(`upstream_interfaces.py`) since Sprints 1-7's real codebase wasn't available.

**This is a starting point, not a signed-off deliverable.** Per the directive's
own mandatory AI-assistant policy: every module here must be run and
independently tested by an engineer against real/surrogate data before anyone
marks Sprint 8 or Sprint 9 done. Nothing in this repo should be treated as
verified just because it runs.

## What this implements

### Sprint 8 — `divergence_engine/`
- `state.py` — `DivergenceState` (append-only), `TypeScores` with the 0.15
  ambiguity rule (`AMBIGUITY_THRESHOLD`, MP-05, explicitly provisional).
- `cooccupancy.py` — Condition 1: windowed contingency table + Fisher's exact
  test, Bonferroni-corrected (reused pattern from Sprint 6 Day 17).
- `granger.py` — Condition 2: within-regime Granger predictability + the hard
  **20-session-per-regime power gate (MP-09)**. ⚠️ Uses statsmodels' AIC-lag VAR
  Granger test as a stand-in for the spec's Bayesian MS-VAR
  (Droumaguet/Warne/Wozniak 2017) — statsmodels doesn't ship that estimator.
  Flagged explicitly in the module docstring; swap in a real Bayesian MS-VAR
  implementation before production use if your team needs the joint estimation,
  not pre-segmented-by-regime approximation.
- `engine.py` — wires both conditions into the four type-score evidence
  formulas. **The evidence-to-type weighting in `compute_divergence_state` is
  a first-pass approximation of Bible Part 5.5-5.7's formal math** — confirm
  the exact coefficients against the Bible before shipping.

### Sprint 9 — `claims_engine/`
- `claim_levels.py` — Level 0-3 gates. Level 3 is a hard AND across all five
  conditions; failing even one means nothing surfaces (no softened version).
- `surfacing_policy.py` — surface / UNCLEAR / withhold-entirely decision logic.
- `grounded_generation.py` — constrained-RAG pipeline: 3 supporting excerpts +
  1 mandatory near-miss, constrained system prompt, citation-chain logging,
  clinical-terminology filter, 6-month mandatory Level-3 human review.
  `LLMClient` is a `Protocol` — wire in your actual **self-hosted** inference
  client; this module never calls a third-party API.

### `synthetic/planted_profiles.py`
Sprint 8 Day 24 / Sprint 15 Day 44 validation harness: generates 20+ planted
profiles per divergence type (Ignorance, Aspiration, Self-Protection, Active
Transition) and reports per-type accuracy/precision/recall. Current run:
all four types clear the >75% bar (see `tests/` for the pinned assertion).

## What is NOT done here (explicitly out of scope for this pass)

- No real Sprint 1-7 integration — `upstream_interfaces.py` defines the shapes
  Sprint 8/9 expect; swap in real imports from your HSSM/NSSM/Attractor/Domain
  modules.
- No real Bayesian MS-VAR — see `granger.py` flag above.
- No MLflow logging wiring for thresholds/hyperparameters (Global Standard
  item 7) — the values that should be logged are computed and returned in
  `Provenance` objects; wire the actual `mlflow.log_param`/`log_metric` calls
  at your integration point.
- No constitutional-policy-engine wiring (Global Standard item 6) —
  `SurfacingContext.constitutional_restriction_active` is a bool input your
  policy engine should populate; not computed here.
- No real per-user model registry / namespace isolation — out of scope for
  Sprint 8/9 specifically (that's Sprint 13).
- The `>80% of user-domain pairs produce a DivergenceState` and `>90%
  shared-latent-driver test success rate` DoD numbers require a real surrogate
  population (TILES-2018/GLOBEM) run — not fabricable from this standalone
  harness. Flagged rather than faked.

## Running it

```bash
pip install numpy scipy statsmodels pytest --break-system-packages
cd chronis_sprints_8_9
PYTHONPATH=. python3 -m pytest tests/ -v
PYTHONPATH=. python3 synthetic/planted_profiles.py
```

## Open items to escalate before sign-off

1. Confirm Bible Part 5.5-5.7's exact type-score coefficients against
   `engine.py`'s evidence-weighting heuristics — current weights are tuned to
   pass synthetic validation, not derived from the Bible's formal equations.
2. Decide whether the Bayesian MS-VAR gap in `granger.py` is acceptable for
   this milestone or must be closed before Sprint 8 sign-off.
3. Run the full suite against real TILES-2018/GLOBEM-derived HSSM/NSSM output
   once Sprint 1-7 code is available, not just synthetic planted profiles.
