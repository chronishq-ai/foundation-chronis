# CHRONIS AI/ML Core Build: Sprints 7, 8 & 9

### Narrative State-Space Model (NSSM), Divergence Engine, and Claims Engine

**Status:** Synthetically validated and structurally complete.
**Bible Traceability:** Part 9.4 (Narrative-Semantic Model); Part 5.11 (Person-Calibration Doctrine); Part 5.5–5.7 (Divergence Engine); Part 5.9 (Claims Engine).



## 1. Pipeline Overview

This repository executes the core CHRONIS mathematical premise: comparing a user's lived behavior (System A) against their self-narrative (System B).

* **Sprint 7 (`nssm_pipeline/`)** takes raw session transcripts and uses Weak Supervision to fit a calibrated Narrative State-Space Model (NSSM), giving System B the exact same mathematical shape as System A.
* **Sprint 8 (`divergence_engine/`)** takes the NSSM output and runs strict statistical tests (Fisher's exact, Granger causality) to calculate Divergence Type Scores between the systems.
* **Sprint 9 (`claims_engine/`)** acts as the safety and generation layer, forcing Divergence States through a hierarchy of admissibility gates (Levels 0-3) before passing them to a constrained, self-hosted RAG pipeline for grounded text generation.

---

## 2. Module Breakdown

### Sprint 7 — `nssm_pipeline/` (Narrative State-Space Model)

* `weak_supervision_label_layer.py`: Employs a Dawid-Skene EM label model to aggregate noisy labeling functions across 8 narrative dimensions without manual ground-truth data. Abstention is explicitly modeled to prevent symmetric-saddle-point failure. Calculates a heteroskedastic measurement uncertainty term (`sigma_t`) from posterior entropy. **Self-hosted LLM constraint strictly enforced.**
* `nssm_calibration.py`: Turns soft labels into a fitted NSSM. Implements rolling z-score idiolect normalization and split-conformal calibration (targeting a stated marginal coverage guarantee). Fits the discrete narrative regime (`q_t`) and continuous state (`n_t`) using numerical MLE and a log-normal duration prior. Regime count `J` is strictly selected via BIC.
* `cross_system_wiring.py`: The gating and preparation layer. Includes the person-calibrated narrative-density gate, and prepares the windowed arrays for Sprint 8.

### Sprint 8 — `divergence_engine/`

* `state.py`: Defines the append-only `DivergenceState` and `TypeScores`. Implements the 0.15 ambiguity rule (`AMBIGUITY_THRESHOLD`, MP-05).
* `cooccupancy.py`: Condition 1. Evaluates windowed contingency tables using a Bonferroni-corrected Fisher's exact test.
* `granger.py`: Condition 2. Evaluates within-regime predictability. **Strictly enforces the MP-09 20-session-per-regime power gate.**
* `engine.py`: Maps the statistical evidence into the four divergence type scores (Ignorance, Aspiration, Self-Protection, Active Transition) using an approximation of the Bible Part 5.5-5.7 math.

### Sprint 9 — `claims_engine/`

* `claim_levels.py`: Enforces the Level 0-3 gating hierarchy. **Level 3 is a hard AND across all five conditions; failing even one means nothing surfaces.**
* `surfacing_policy.py`: Implements `SURFACE`, `WITHHOLD`, and `UNCLEAR` logic, including hard stops for acute trauma markers without therapeutic context and unresolved Sprint 20 conflict records.
* `grounded_generation.py`: The constrained RAG pipeline. Mandates 3 supporting excerpts + 1 deliberate near-miss counter-example. Enforces a strict clinical-terminology block (diagnoses route immediately to human review) and a standing 6-month mandatory human-review requirement for all Level 3 text.

---

## 3. Validation & Testing

Run the included synthetic validation suite to verify the mathematical soundness of the pipeline:

```bash
pip install numpy scipy statsmodels pandas pytest --break-system-packages

# Run automated test assertions
PYTHONPATH=. python3 -m pytest tests/ -v

# Run the end-to-end regime-recovery and divergence suite
PYTHONPATH=. python3 synthetic/planted_profiles.py

```

*Current Status:* The end-to-end synthetic run (which plants sustained agentic, passive, and oscillating patterns) cleanly bypasses label-switching using the Hungarian algorithm and successfully clears the hard `>75%` regime-recovery accuracy gate (currently operating at 100.0% best-permutation accuracy on synthetic data).