# Chronis Foundation — Hardener Branch (Sprints 13–15)

> **Branch:** `hridhani/hardener-sprint-13-15`  
> **Scope:** Per-User Model Isolation · Constitutional Policy Engine · Observer-Effect Safeguard

---

## Overview

This branch delivers three consecutive hardening sprints on top of the Chronis AI foundation pipeline. Each sprint adds a distinct security and correctness layer — they are designed to compose, not replace, prior sprint deliverables.

| Sprint | Focus | Key Deliverable |
|--------|-------|-----------------|
| **13** | Per-User Model Isolation | `chronis_ml/` — isolated adapters, IsolationError guardrail, MLflow-gated registry |
| **14** | Constitutional Policy Engine | `policy_engine/` + `integration/` — single choke-point authorization, audit-once guarantee |
| **15** | Observer-Effect Safeguard | `observer_effect/` — 30-day surfacing index, aspiration evidence zero-out at read time |

---

## Repository Structure

```
foundation-chronis-hridhani-hardener-sprint-13-15/
│
├── chronis_ml/                  # Sprint 13 — Per-user model store & fine-tune stubs
│   ├── store.py                 # IsolatedModelStore — raises IsolationError on cross-user writes
│   ├── train.py                 # Personal LM fine-tune harness (base-checkpoint enforced)
│   └── ops.py                   # MLflow registry ops with gated validation
│
├── policy_engine/               # Sprint 14 — Constitutional layer (single choke point)
│   ├── consent.py               # ConsentRecord, ConsentTier enum, check_inference_consent()
│   ├── policy_rule.py           # PolicyRule — min_consent_tier NOW enforced in covers()
│   ├── principal.py             # ModelPrincipal.check() — audit-once, raise-on-deny
│   ├── audit_log.py             # Append-only, hash-chained audit log
│   └── errors.py                # ConsentTierError, ModeCBlocked, PolicyDenied hierarchy
│
├── integration/                 # Sprint 14 — Gated ML I/O wrappers
│   ├── gated_store.py           # GatedModelStore — routes all reads/writes through principal
│   ├── gated_registry.py        # GatedRegistry — REGISTRY_REGISTER requires explicit rule
│   ├── gated_claims.py          # evaluate_claim_access() — constitutional claim surfacing
│   └── gated_divergence.py      # gated_compute_divergence_state() — gated divergence engine
│
├── e2e/                         # Sprint 14 — End-to-end TILES pipeline runner
│   ├── pipeline_runner.py       # Stage orchestrator (MP-18 pre-gate gap documented)
│   ├── tiles_loader.py          # TILES feature matrix loader (post-NaN-handling entry point)
│   └── timing.py                # Pipeline timing and stage-count metrics
│
├── observer_effect/             # Sprint 15 — Observer-effect mitigation
│   ├── index.py                 # SurfacingIndex — 30-day inclusive window, would_flag()
│   ├── safeguard.py             # aspiration_evidence_weight() — zeros flagged evidence at read
│   ├── observer.py              # Observer coordinating module
│   ├── profiles.py              # Validation harness — routes through real pipeline (NOT toy sim)
│   ├── regression.py            # Regression suite for observer-effect invariants
│   └── README.md                # Observer-effect design writeup
│
├── tests/                       # Full regression & boundary test suite
│   ├── test_policy_boundary_cases.py   # 100+ parametrized policy-boundary cases (incl. T1/T2/T3)
│   ├── test_audit_tamper.py            # Append-only + hash-chain tamper resistance
│   ├── test_e2e_pipeline.py            # End-to-end pipeline + MP-18 known-gap documentation
│   ├── test_consent_tier_gate.py       # ConsentTier floor enforcement
│   ├── test_mode_c_block.py            # Mode C hard-block (construction-time, not runtime)
│   ├── test_g1_g4_review.py            # G1–G4 structural review
│   └── test_sprint13.py / test_sprint15.py
│
├── docs/
│   └── g1_g4_signoff.md        # G1–G4 production-readiness sign-off (named owner, dated)
│
├── mp_registry.json             # Master Problem Registry — MP-13 and MP-18 with owners
├── upstream_interfaces.py       # Dataclass contracts (AttractorRecord, Domain) for pipeline
├── requirements.txt             # Pinned dependencies
├── licenses.json                # License manifest for all pinned dependencies
└── pytest.ini                   # Test discovery configuration
```

---

## Sprint 13 — Per-User Model Isolation

**Goal:** Every personal adapter is isolated to its owner. No cross-user read or write is possible through normal code paths.

### Key Design Decisions

- `IsolatedModelStore` raises `IsolationError` immediately if a caller attempts to write an adapter to a path that doesn't belong to them — checked by path ownership, not by caller-supplied boolean.
- `promote_to_global()` unconditionally raises — sharing is architecturally impossible, not just policy-blocked.
- `train.py` enforces a shared `chronis-base-v1` checkpoint via `ensure_base()` before every per-user fine-tune.

> **Known open item (S13.1):** `train.py` uses a hashed-vector adapter as a stand-in. Real LoRA/PEFT implementation is marked open and requires Senior ML Lead approval before any fine-tune logic changes.

---

## Sprint 14 — Constitutional Policy Engine

**Goal:** Route every ML data read/write through a single authorization choke point. No bypass path exists, including for legitimate-seeming retries.

### Key Design Decisions

- `ModelPrincipal.check()` is the **only** entry point. It either returns `None` (granted) or raises `PolicyDenied` (or a subclass). Every code path — success or denial — produces **exactly one audit entry** before returning.
- `PolicyRule.min_consent_tier` is now **actively enforced** in `covers()` against the requester's actual consent tier (audit finding S14.1 fix).
- Mode C (`OperationalMode.MODE_C`) is rejected at rule **construction time** in `__post_init__` — it cannot appear in `allowed_modes` for any rule, making it structurally impossible to grant.
- The audit log is append-only and hash-chained (`audit_log.py`) — no `update`, `edit`, `delete`, or `purge` methods exist.

### Running the Policy Boundary Tests

```bash
python -m pytest tests/test_policy_boundary_cases.py -v
```

> Requires the legacy upstream packages (`claims_engine`, `divergence_engine`) from Sprints 7–9. Without them, test collection will fail at import — this accurately reflects the codebase's dependency state.

### Known Open Item — MP-18

HSSM fit and attractor detection (Stages 2–3 of `e2e/pipeline_runner.py`) run **before** any `ModelPrincipal.check()` call. This pre-gate compute gap is documented in `test_e2e_pipeline.py::test_KNOWN_GAP_hssm_and_attractor_stages_run_before_any_gate_check` and tracked in `mp_registry.json`. **Owner: Senior ML Lead.**

---

## Sprint 15 — Observer-Effect Safeguard

**Goal:** Mitigate the observer effect — the risk that surfacing a divergence claim to a user changes the very behavioral patterns the claim was derived from.

### Key Design Decisions

- `SurfacingIndex.would_flag()` implements a **30-day inclusive window** (day+15 flagged, day+45 not flagged — verified against the worked example).
- `aspiration_evidence_weight()` **zeros out flagged evidence at read time**, regardless of whether the caller remembered to set the flag — deliberately more robust than a caller-trusted boolean.
- `product_copy()` is tested to **never leak** the internal flag name to user-facing text.

### Validation Harness

`observer_effect/profiles.py` routes planted-profile trajectories through the
**real package path**: `backbone.hssm.fit_hssm` → `nssm_pipeline.fit_nssm` →
`divergence_engine.engine.compute_divergence_state` → `DivergenceState.type_scores`.

Accuracy is measured end-to-end through that path. The Granger estimator behind
the available divergence package remains **OLS-VAR-limited (S79.1 open)** —
named explicitly in `mp_registry.json`, MLflow tags, and `KNOWN_LIMITATIONS.md`.
This is **not** a claim that Bayesian MS-VAR is complete.

The old hand-invented formula is quarantined in
`observer_effect/scratch_type_scores.py` and is never used on the live scoring path.

### Known Open Item — MP-13

The observer-effect mitigation is **permanently open by design** — mitigation is not closure. The surfacing index and evidence zeroing prevent the worst-case feedback loop, but the underlying measurement-affect-measurement dynamic cannot be fully resolved without architectural changes outside this sprint's scope.

---

## G1–G4 Production-Readiness Sign-off

See [`docs/g1_g4_signoff.md`](docs/g1_g4_signoff.md) for the full review.

| Guarantee | Status | Owner |
|-----------|--------|-------|
| G1 — (text not located in uploaded materials) | UNVERIFIED | Senior ML Lead |
| G2 — ML pipeline never writes back to Layer 0 | **PASS** (Sprint 14 scope) | Senior ML Lead |
| G3 — NULL/missing states are typed, non-imputed | NOT INDEPENDENTLY VERIFIED (Sprint 1 scope) | Senior ML Lead |
| G4 — No bypass path for any data-access grant | **PASS** (Sprint 14 scope) | Senior ML Lead |

---

## What This Branch Does NOT Implement

The following are explicitly **out of scope** for Sprints 13–15 and are documented here to prevent future confusion:

- **Sprint 16:** Isolated microVM/gVisor processing container, Argon2id key derivation, VoicePrivacy EER unlinkability harness, bystander biometric TTL — confirmed absent, assigned as a separate senior-led ground-up build.
- **Real LoRA/PEFT fine-tuning (S13.1):** `train.py` uses a hashed-vector adapter stand-in. Approved real PEFT implementation is a Senior ML Lead deliverable.
- **Full Sprint 3/7/8 scientific remediations:** this zip includes wiring-compatible upstream packages so Sprint 14/15 tests install and run. S79.1 (Bayesian MS-VAR) and related research tickets remain owned by those teams.

---

## Running Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full test suite (from this directory)
python -m pytest tests/ -v

# Run only the constitutional policy boundary cases (Sprint 14)
python -m pytest tests/test_policy_boundary_cases.py -v

# Run only Sprint 15 (observer + S15.1 harness)
python -m pytest tests/test_sprint15.py -v
```

See also `KNOWN_LIMITATIONS.md`.

---

## Dependencies

All dependencies are pinned in `requirements.txt` and cross-referenced in `licenses.json`. No dependency has an incompatible license for this project's usage pattern.

---

## Scope Disclaimer

This branch implements Sprint 13 (per-user model isolation) + Sprint 14 (constitutional policy engine, gated I/O, TILES e2e) + Sprint 15 (observer-effect safeguard).

It does **not** implement the isolated processing container (microVM/gVisor, 24h session-key TTL, RAM zero-fill) — that is Sprint 16 (Bible 5.24).
