# CHRONIS — AI/ML Production Build
## Team 4 (INVENTORS) — Sprint 7, Days 19–21
### Module: Narrative State-Space Model (NSSM) — Weak Supervision, Calibration & Fitting, Cross-System Wiring

**Status:** Complete, synthetically validated, handoff-eligible for Sprint 8 (Days 22–24)
**Owners:** Abhinav (Lead), Anuj, Mansi
**Bible traceability:** Part 9.4 (Narrative-Semantic Model, Layer 3); Part 5.11 (person-calibration doctrine); Part 5.5–5.7 (Divergence Engine formal math, consumed downstream); Part 5.9 (Claims Engine, consumed downstream)
**Files in this module:**

| File | Sprint Day | Purpose |
|---|---|---|
| `weak_supervision_label_layer.py` | Day 19 | Weak-Supervision Label Layer (WSL) |
| `nssm_calibration.py` | Day 20 | Idiolect normalization, conformal calibration, NSSM fitting |
| `cross_system_wiring.py` | Day 21 | Narrative-density gate, Sprint 8 input wiring, synthetic validation |

---

## 1. Module Overview

This module gives **System B (the narrative self-story)** the same mathematical object type as **System A (behavior, Sprint 3's HSSM)**: a fitted, person-calibrated regime-switching process. Before this module existed, the original Sprint 6/7 design compared a real statistical behavioral model against a rough topic-tagger for narrative — an architecture-audit-flagged flaw, because it meant Chronis's core claim ("here is the gap between how you live and the story you tell about it") was never actually measuring two comparable things.

This module closes that gap in three stages:

1. **Day 19** turns raw session transcripts into calibrated, uncertainty-aware soft labels across 8 narrative dimensions, using weak supervision (no hand-labeled ground truth required).
2. **Day 20** turns those soft labels into a fitted **Narrative State-Space Model (NSSM)** — a slow discrete narrative regime `q_t` with a log-normal duration prior, and a fast continuous narrative state `n_t` — using the identical fitting discipline Sprint 3 established for the behavioral HSSM.
3. **Day 21** gates which sessions are trustworthy enough to fit on, prepares the exact inputs Sprint 8's Divergence Engine needs, and proves — on synthetic data with known ground truth — that the whole pipeline actually recovers narrative regimes correctly before anything downstream is allowed to build on it.

Everything in this module implements **Bible Part 9.4 — Narrative-Semantic Model, Layer 3**.

---

## 2. Day-by-Day Functionality

### Day 19 — Weak-Supervision Label Layer (`weak_supervision_label_layer.py`)

**What it does:** Produces, per session and per narrative dimension, a soft class distribution plus a learned measurement-uncertainty term (`sigma_t`) — with zero hand-labeled ground truth.

- **8 narrative dimensions**, each with 2–4 independently-noisy labeling functions (LFs) with deliberately different failure modes: keyword/regex heuristics, syntactic-pattern heuristics, and one self-hosted-LLM-based LF per dimension. LFs vote a class index or `ABSTAIN`.
- **`DawidSkeneLabelModel`** aggregates LF votes via Expectation-Maximization, learning each LF's reliability purely from inter-LF agreement patterns — never from ground truth.
  - **Abstention is modeled as an explicit `k+1`-th outcome**, not silently dropped. Each LF's confusion matrix is `(true_class, outcome)` where `outcome ∈ {0, ..., k-1, ABSTAIN}`. This matters because most of our LFs are one-sided detectors (vote one specific class, or stay silent) — an LF's *abstention rate* conditioned on true class is frequently the only signal that distinguishes "reliable detector" from "noise," and skipping abstaining votes throws that signal away entirely. See **§4 — Testing Notes** for the concrete failure this fix corrects.
  - Fitting is seeded with a **data-driven majority-vote initialization** (not a symmetric uniform prior) to avoid a separate label-symmetry saddle point in EM.
- **`SelfHostedLLMClient`** enforces, at every call site (not just construction), that the labeling LLM never targets a non-local host — `SelfHostedOnlyError` is raised at the connector level, not documented as a policy and hoped for.
- **`sigma_t`** (heteroskedastic measurement uncertainty) is derived from the **normalized entropy of the label model's posterior distribution** — confident, peaked posteriors yield low `sigma_t`; ambiguous ones yield high `sigma_t`. This is a learned, per-session, per-dimension quantity, never a fixed constant.

```python
output: Dict[str, Dict[str, DimensionOutput]] = wsl.transform(sessions)
# output[session_id][dimension_name] -> DimensionOutput(distribution, sigma_t, dominant_class)
```

### Day 20 — Person-Specific Calibration & NSSM Fitting (`nssm_calibration.py`)

**What it does:** Turns Day 19's soft labels into a fitted, person-calibrated NSSM.

- **`IdiolectNormalizer`** rolling-z-scores every observation channel against a person's own strictly-past trailing window (never population statistics, never future sessions). Returns `NaN` — not a fabricated value — until `min_sessions_for_baseline` is met.
- **`ConformalCalibrator`** implements split conformal prediction: computes a nonconformity score (`1 − P̂(true class)`) over sparse per-person "gold checks," derives a single quantile `qhat`, and builds prediction sets with a **stated marginal coverage guarantee** (e.g. 90%). Falls back to a population-level calibration pool when a person doesn't yet have enough gold checks, and reports that fallback explicitly (`person_specific: bool`) rather than hiding it.
- **The NSSM itself**: a slow discrete narrative regime `q_t` with a **log-normal duration prior** (`LogNormalDurationPrior` — hazard-based, not the constant/geometric hazard implicit in a plain HMM), and a fast continuous narrative state `n_t`, fit via a **duration-augmented Kim (1994) filter** (`DurationAugmentedKimFilter`). The emission model loads `n_t` onto all 8 narrative-dimension channels using a per-regime loading vector, with **heteroskedastic emission noise set directly from Day 19's `sigma_t`** at every timestep.
- **Fitting harness (`fit_nssm_for_j` / `fit_nssm`)**: ≥10 random restarts (default), one of which is a **statsmodels-**(`MarkovRegression`)-informed warm start; keeps the run with the highest log-likelihood — never the "best-looking" one. `J` (regime count) is selected by **BIC over `{2, 3, 4}`**, never by inspection. Label-switching is resolved post-hoc by canonical regime sorting (`canonical_sort_regimes`), reusing Sprint 3 MP-02's exact convention.
- **Honest engineering note documented in-code:** the log-normal duration prior has no closed-form M-step, so the "EM fitting harness" is implemented as direct numerical MLE against the Kim filter's likelihood surface (`scipy.optimize.minimize`), while preserving every other piece of Sprint 3's fitting discipline (many restarts, keep-best-likelihood, never prettiest-looking regimes). This is a standard, defensible substitute in the duration-HSMM literature, not a shortcut — flagged for team-lead visibility, not buried.

### Day 21 — Narrative-Density Gate, Cross-System Wiring, Synthetic Validation (`cross_system_wiring.py`)

**What it does:** Gates which sessions are fit-worthy, prepares Sprint 8's exact inputs, and proves regime recovery on synthetic data before handoff.

- **`NarrativeDensityGate`**: counts "wearer-attributed, agentive first-person clauses" (`count_agentive_first_person_clauses`) per session; a session only enters the fit set `S` once it clears a **person-calibrated** minimum. Below-gate sessions are excluded from `S`, not zero-filled or imputed.
- **Cross-system wiring for Sprint 8**:
  - `run_fisher_alignment` builds a **windowed contingency table** between the behavioral regime `p_t` and narrative regime `q_t` for every pair, tests with **Fisher's exact test**, and **Bonferroni-corrects across every pair tested** (`k_count × j_count`).
  - `prepare_granger_inputs` / `run_granger_ms_var` prepare `(m_t, n_t)` series for a **Bayesian MS-VAR Granger-causality test** and expose the exact interface Sprint 8 will call through.
- **The hard 20-observation-per-regime power gate (MP-09)** (`power_gate_ok`): below 20 observations in a regime pair, the Granger test **does not run at all**, and `prepare_granger_inputs` returns `None` rather than an underpowered result — enforced as a single boolean, no override path.
- **Synthetic validation suite**: plants 3 known narrative regime patterns (sustained agentic, sustained passive, oscillating/ambivalent) into synthetic sessions, runs the **full Day 19 → Day 20 → Day 21 pipeline blind** (the NSSM never sees planted labels), decodes the NSSM's regime assignment per session, and scores recovery accuracy against the planted ground truth using the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) to find the best-possible regime-index-to-pattern alignment before scoring — correctly handling label-switching rather than assuming index alignment.
- **`assert_regime_recovery_accuracy`** is a hard assertion (`> 75%`), not a warning: a failing run stops the handoff to Sprint 8, per the Global Standard's "a green checkbox that isn't actually true is worse than a red one."

```
Confusion matrix (rows=planted, cols=decoded regime index):
[[0 0 6]
 [0 6 0]
 [6 0 0]]
Best-permutation accuracy: 100.0%
PASSED the 75% regime-recovery bar — eligible for handoff to Sprint 8.
```

---

## 3. Testing Notes — A Correctness Fix Found During Integration

Per the program's AI-assistance policy, this code was independently run and verified, not assumed correct because an assistant produced it. That testing surfaced one real defect in the originally-delivered Day 19 `DawidSkeneLabelModel`:

- **Defect:** abstaining LF votes were silently skipped when building confusion matrices. For LFs that only ever cast one specific vote (the majority shape of the LFs in this module — e.g. an LF that fires `low_tolerance` or stays silent, never anything else), this made the model unable to distinguish "reliable one-sided detector" from "noise" — the label model would converge to a near-uniform posterior regardless of what actually voted. This reproduced cleanly on an isolated two-LF, two-class case even at 120 sessions and 200 EM iterations — not a small-sample artifact.
- **Fix applied:** abstention is now modeled as an explicit `k+1`-th outcome with its own learned probability per true class (matching Snorkel's own generative label model), combined with a majority-vote EM initialization to avoid a related symmetric-saddle-point failure mode.
- **Verification:** re-ran Day 19's own smoke test (output confidence improved, no regressions) and Day 21's synthetic validation suite (regime recovery went from a failing 37.5–75.0% range to a clean 100% once this fix was in place, confirming the earlier low numbers were a label-model defect, not a fundamentally unrecoverable synthetic design).

**Action for team leads:** this fix is already applied in the delivered `weak_supervision_label_layer.py`. Anyone who pulled an earlier copy of this file should re-pull before running further validation.

---

## 4. Cross-Sprint Dependencies (Open / Provisional Items)

**This is the critical section for handoff.** Every item below is a working, independently-testable standalone implementation that stands in for a module another team owns. Each is marked in-code with the exact tag listed here — grep for the tag to find every call site before wiring in the real dependency.

| Tag | Location | What it stands in for | Swap-in owner |
|---|---|---|---|
| `[REQUIRES SPRINT 2 PROSODY EXTRACTOR]` | `weak_supervision_label_layer.py` → `lf_self_role_prosody_cross_check` | Sprint 2 Day 5's per-session z-scored prosody output (F0 contour, energy envelope, etc.). Currently reads `session.prosody_features` if present and **honestly abstains** (never guesses) when it's `None`. | FOUNDRY (Sprint 1) / Team 1's prosody pipeline (Sprint 2) |
| `[REQUIRES SPRINT 2 Z-SCORING UTILITIES]` | `nssm_calibration.py` → `IdiolectNormalizer.rolling_zscore` | FOUNDRY's per-person rolling z-scoring utility (Sprint 1 Day 3's "never normalize across the population" rule). Same-contract standalone implementation ships in its place. | FOUNDRY (Sprint 1) |
| `[REQUIRES SPRINT 3 KIM FILTER]` | `nssm_calibration.py` → `DurationAugmentedKimFilter` | BACKBONE's production Kim (1994) filter + EM fitting engine (Sprint 3), which the NSSM is supposed to reuse as "identical model class to System A." A standalone duration-augmented equivalent ships here so Sprint 7 wasn't blocked on Sprint 3's internals being exposed as a shared library. | BACKBONE (Sprint 3, Team 2) |
| `[REQUIRES SPRINT 4 CALIBRATION HARNESS]` | `cross_system_wiring.py` → `NarrativeDensityGate.calibrate` | Sprint 4 Day 11's synthetic-trajectory grid-search calibration harness (N/T thresholds vs. false-positive/false-negative rate, targeting a precision bar). Same-shape standalone stand-in ships here. | BACKBONE (Sprint 4, Team 2) |
| `[REQUIRES SPRINT 6 FISHER'S EXACT]` | `cross_system_wiring.py` → `run_fisher_alignment` | Sprint 6 Day 17's production windowed-contingency-table + Bonferroni-corrected Fisher's-exact alignment procedure. Same-math standalone equivalent ships here. | CHRONOS (Sprint 6, Team 3) |
| `[REQUIRES SPRINT 8 GRANGER MS-VAR]` | `cross_system_wiring.py` → `run_granger_ms_var` | The real Bayesian Markov-Switching VAR Granger-causality test (Droumaguet, Warne & Wozniak 2017), AIC-selected lag order — Sprint 8 Day 23's own deliverable. **This is a wiring-level stub only** (plain OLS Granger test): do not wire its p-values or direction output into any Level 2 claim. | INVENTORS (Sprint 8, this team) |

### Handoff checklist

- [ ] Swap `lf_self_role_prosody_cross_check`'s prosody-feature contract for Sprint 2's real extractor output; re-run Day 19's smoke test to confirm the LF now votes (rather than universally abstaining) on real data.
- [ ] Swap `IdiolectNormalizer.rolling_zscore`'s body for a direct call into FOUNDRY's z-scoring utility; confirm output is bit-for-bit consistent with the standalone version on a shared synthetic input.
- [ ] Replace `DurationAugmentedKimFilter` with a call into BACKBONE's Kim filter, if/when Sprint 3 exposes a generic custom-transition-matrix hook; otherwise keep this module's standalone filter but re-validate against Sprint 3's convergence-rate bar (≥90% across random-init runs on ≥3 surrogate users).
- [ ] Swap `NarrativeDensityGate.calibrate`'s grid search for Sprint 4's real harness; re-calibrate thresholds against real per-user session history, not synthetic data.
- [ ] Swap `run_fisher_alignment`'s Fisher's-exact logic for a direct import from Sprint 6's module; confirm Bonferroni correction denominators match exactly (same `k_count × j_count` convention).
- [ ] Replace `run_granger_ms_var`'s OLS stub with Sprint 8's real Bayesian MS-VAR test **before any Level 2 claim is generated from its output** — this is a hard blocker, not a nice-to-have.

---

## 5. Known Limitations (Provisional, Not Yet Closed)

- **Ordinal-encoding simplification:** the NSSM's continuous state `n_t` loads onto all 8 narrative dimensions via a fixed ordinal encoding per dimension (`ORDINAL_ENCODING` in `nssm_calibration.py`), collapsing each dimension's soft class distribution to one scalar. This is a real modeling choice made to keep the emission model tractable for a single continuous state, not an implied Bible requirement — worth revisiting if a richer, multi-dimensional narrative emission model is wanted downstream.
- **0.15 ambiguity threshold (MP-05):** not touched by this module; remains explicitly provisional per Sprint 9/10, pending a ground-truth-labeled divergence-type dataset that doesn't exist yet.
- **Synthetic validation scale:** the smoke tests in this module's `if __name__ == "__main__"` blocks intentionally use reduced settings (fewer sessions, smaller duration caps, fewer random initializations) purely to finish quickly in a sandboxed run. A production sign-off run should use each module's actual defaults (`n_random_inits=10`, `j_candidates=(2, 3, 4)`) against a larger planted-profile suite, matching Sprint 8 Day 24's own ≥20-profiles-per-type bar.
- **`d_max` (duration cap) must exceed the longest expected regime block** — discovered directly during this module's own validation (see `cross_system_wiring.py`'s `__main__`, `d_max=6` comment). Anyone extending the synthetic validation suite should size `d_max` to the longest planted block first.

---

*This README satisfies Global Standard requirement 8: what this module does, what Bible section it implements, and what remains open or provisional. Team leads: enforce the handoff checklist in §4 in code review before any Sprint 8 pull request merges against this module's stubs.*
