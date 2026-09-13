# Sprint 10 — Cold Start Compass & Threshold Calibration II

**Status:** ✅ 57/57 Sprint 10 tests passing | 99/99 total passing

---

## Architecture

```
Sprint 3 backbone (GaussianHSMM)
         │
         │  hssm_fit_from_backbone()
         ▼
     HSSMFit ─────────────────────────────────────┐
  (duration_parameters: {regime_id: {dur_mu, dur_sigma}})   │
         │                                         │
         │  estimate_slow_phase_duration()          │
         ▼                                         │
        D* = exp(dur_mu + dur_sigma²/2)             │ (log-normal mean)
         │                                         │
         │  compute_observation_window()            │
         ▼                                         │
  window = 2 * D*                                  │
         │                                         │
         │  run_cold_start_pipeline()               │
         ▼                                         │
  ColdStartStateMachine                            │
    Stage 0 (Days  1– 7): zero inference ◄─────────┘ (skips D* entirely)
    Stage 1 (Days  8–29): tentative internal
    Stage 2 (Days 30–59): first HSSM fit, Level-1 only
    Stage 3 (Days 60–89): divergence accumulation
    Stage 4 (Day  90+):   claims (evidence-gated)
         │
         ▼
   ColdStartState { stage, can_surface_claims, user_facing_message, ... }
```

---

## D* Derivation (the fix)

**Old (wrong):**
```python
# Derived persistence probability from regime_posterior[:, slow_regime] — a proxy, not duration
D* = 1 / (1 - p_slow)
```

**New (correct):**
```python
# Uses actual EM-fitted log-normal dwell-time parameters from Sprint 3 backbone
dur_mu    = fitted_hssm.duration_parameters[slow_regime_id]["dur_mu"]
dur_sigma = fitted_hssm.duration_parameters[slow_regime_id]["dur_sigma"]
D* = exp(dur_mu + dur_sigma**2 / 2)   # log-normal mean — exact formula
```

The Sprint 3 backbone (`backbone/hssm/model.py`) uses an **explicit log-normal duration prior**. After the EM M-step, `dur_mu[k]` and `dur_sigma[k]` are the log-space parameters of the fitted dwell-time distribution for regime `k`. The mean of `X ~ LogNormal(μ, σ)` is `exp(μ + σ²/2)`.

---

## File Map

| File | Role |
|------|------|
| `cold_start.py` | D* estimator, state machine, observation window |
| `cold_start_pipeline.py` | Full orchestration (stage-first flow) |
| `cold_start_wiring.py` | Adapter for real Sprint 3/8 types |
| `upstream_interfaces.py` | `HSSMFit`, `RegimeObservation`, `hssm_fit_from_backbone()` |
| `tests/test_cold_start.py` | 44 unit/regression tests |
| `tests/test_cold_start_wiring.py` | 13 wiring/integration tests |

---

## Teammate API

**Question:** *"What is the function/interface for your Cold Start stage checker?"*

**Answer:**

```python
from upstream_interfaces import HSSMFit, hssm_fit_from_backbone
from cold_start import evaluate_cold_start, ColdStartStage

# 1. Convert Sprint 3 backbone GaussianHSMM → HSSMFit contract
fit = hssm_fit_from_backbone(
    backbone_model,         # fitted GaussianHSMM after canonicalize_labels()
    user_id="u_001",
    fit_id="2026-08-23T00:00:00Z_s0",
)

# 2. Evaluate cold-start for a given day
state = evaluate_cold_start(
    day=45,
    fitted_hssm=fit,
    divergence_state=divergence_engine.get_state(user_id),  # or None
)

# 3. Gate all claims behind state.can_surface_claims
if state.can_surface_claims:     # True only at Stage 4 + evidence gate
    yield_claims(state)
else:
    show_user(state.user_facing_message)   # specific, never "coming soon"

# Check stage directly
assert state.stage in ColdStartStage          # STAGE_0 … STAGE_4
assert state.stage == ColdStartStage.STAGE_2  # example
```

**What D* depends on:**

```python
# The ONLY valid source of D* is the fitted HSSM duration parameters.
# You MUST pass a real HSSMFit from Sprint 3 — no regime_posterior proxies.
fit.duration_parameters[0]   # {"dur_mu": float, "dur_sigma": float}
                              # regime 0 = slow regime (always, after canonicalize_labels)
```

---

## Submit checklist

- [x] `estimate_slow_phase_duration` uses real HSSM `dur_mu`/`dur_sigma` — not posteriors
- [x] D* = exp(dur_mu + dur_sigma²/2) — log-normal mean, exact formula
- [x] `upstream_interfaces.py` updated with `HSSMFit`, `RegimeObservation`, `hssm_fit_from_backbone`
- [x] Stage 0 short-circuits before any inference runs
- [x] MLflow logging: D* and window logged per user, per fit
- [x] 57/57 Sprint 10 tests passing
- [x] 99/99 total tests passing (all sprints)
- [x] `mlflow` is a lazy import (no hard dep at module collection time)
- [x] `requirements.txt` includes `mlflow>=2.0`
