# Chronos — Sprint 5-6: Phase Transition Detection & Domain Emergence

Implements the Bible's phase-transition gate (Part 5.3 / 5.23, Module 4.11) and
domain-emergence engine (Part 5.8), per the Sprint 5-6 directive (Part 13.4
Phase 3, MP-12, Risks 3-A/3-B/3-C).

> **Remediation status:** this package has been through the Intern Remediation
> & Test Pack pass (findings S56.1–S56.10). All ten findings have a fix and a
> test in place — see the per-module table below for what changed.
> **S56.1, S56.4, and S56.5 ship without the Mandatory senior sign-off the
> pack requires for those three (statistical-estimator design choices) — do
> not merge those paths to production without that review.**

## Layout

```
bocd/                   Vendored Adams & MacKay (2007) BOCPD reference impl
                         (gwgundersen/bocd, BSD-3-Clause — see bocd/LICENSE)
phase_transition/        Sprint 5 — phase transition gate + rupture detector
domain_emergence/        Sprint 6 — domain emergence engine
tests/                   Unit + regression tests for both (140 tests)
tests/fixtures/          Relocated synthetic-HSSM fixture (test-only, see S56.6)
```

## phase_transition/ (Sprint 5, Days 13-15)

| Module | Role |
|---|---|
| `bocpd.py` | Wraps the vendored BOCPD lib. **Its output is condition 1 of 3 only** — never a declaration on its own. Also exposes `hazard_sensitivity_sweep` (**S56.3**): sweeps a fixed hazard-rate set over the same data/seed and reports the resulting candidate-changepoint count/timing per hazard, so a single arbitrary hazard's influence on timing is visible rather than assumed. Diagnostic report only, no pass/fail gate. |
| `degradation.py` | Condition 2: fits a Gaussian on the pre-boundary window, scores its log predictive likelihood against the post-boundary window. Sharp degradation = evidence of a real regime change. **S56.2:** the generic scalar-Gaussian model (`PredictiveFitDegradation`) is left in place, documented as not the same claim as testing the actual fitted behavioral generative model. `evaluate_generative_model_degradation` adds a model-agnostic held-out-likelihood harness (pluggable `fit_fn`/`predict_ll_fn`/optional `null_baseline_ll_fn`) that records exact pre/post window boundaries for the reproducibility manifest — see `hssm_degradation.py` for a best-effort regime-conditional implementation built on top of it. |
| `hssm_degradation.py` | **New (S56.2/S56.6).** A best-effort regime-conditional `fit_fn`/`predict_ll_fn` pair for the harness above, built from HSSM-shaped output (`regime_sequence` + `observations`). Narrower claim than genuine forward-predictive HSSM likelihood (see its HONESTY FLAG docstring) since it consumes already-decoded post-window regime labels rather than forward-filtering into the future — closing that gap needs Sprint 3-4's `backbone` package. Opt-in only; wired into the gate as an alternate path, not the default. |
| `stability.py` | Condition 3: tracks post-candidate regime posterior variance for ≥14 calendar days (`min_days`) — **S56.1 fix:** the window is now selected by real calendar-day span via an optional `timestamps` argument (falls back to sample-offset behavior when omitted, backward compatible), not sample count, so sparse/dense users get the same 14-day window. Splits the post-candidate window in half; second-half variance must drop below `drop_ratio × first-half` to count as "stabilizing." `regime_posterior_entropy` and `is_stabilizing_entropy` implement the doctrine-correct entropy-of-regime-posterior metric (the original raw-variance metric was the wrong statistical object) — exercised only against synthetic known-distribution fixtures pending real HSSM posteriors from Sprint 3-4; **ships without the Mandatory senior sign-off the metric-swap decision requires.** |
| `gate.py` | `PhaseTransitionGate` — hard AND of all 3 conditions. Rupture evidence from `bifurcation_log.py` is **ORed into condition 2** (additional evidence), never a substitute for the degradation-score check. |
| `rupture.py` | Module 4.11. `RuptureDetector` declares a rupture only when **all four** hold simultaneously: voice energy >3σ above personal mean, PPG HR >40% above baseline, CSE salience at L5 for >10 min, and significant IMU motion disruption. Hard AND, never a weighted score. Deliberately does *not* catch slow/gradual shifts — acute-event detector only. |
| `bifurcation_log.py` | Append-only log for declared ruptures (`BifurcationEvent`, frozen dataclass). No delete/overwrite methods — separate from the ordinary recurrence pipeline. |

**Test coverage:** BOCPD wrapper + hazard-sensitivity sweep, degradation
scoring (generic and regime-conditional harnesses), stability/reset behavior
in both raw-variance and entropy forms, calendar-day vs. sparse/dense-user
windowing, rupture's 4-condition AND (including each condition's independent
failure mode), the composed gate, and the append-only log — including the
two Definition-of-Done cases: a synthetic transient noise spike does *not*
trigger a declaration, and a synthetic genuine transition (all 3 conditions
met) is declared within the expected latency window.

## domain_emergence/ (Sprint 6, Days 16-18)

| Module | Role |
|---|---|
| `hssm_adapter.py` | **New (S56.6).** Real Sprint 3-4 HSSM adapter — lazily imports the canonical `backbone.hssm.fit_hssm` entry point (S34.7) and extracts `regime_sequence`/`observations` for everything downstream. Raises `BackboneHSSMUnavailableError` (not a silent fallback) if `backbone` isn't installed, and a clear `AttributeError` if the real `HSSMResult` is missing an expected field. **This is now the only production path into HSSM output** — the old `synthetic_hssm.py` stand-in has been removed from the runtime path entirely (relocated to `tests/fixtures/synthetic_hssm_fixture.py`, test-only; zero production imports remain, verified by grep). |
| `context_signature.py` | Turns a regime sequence + observations into one feature vector per contiguous dwell episode. Missing sessions inside an episode are excluded from mean/std, never imputed. |
| `context_clustering.py` | Day 16 — HDBSCAN over episode signatures (chosen because domain count is unknown a priori and noise is expected, not an error). Outputs *raw candidate* clusters only; label `-1` (noise) is kept, never forced into a cluster. |
| `synthetic_transcripts.py` | Mocks the Audio Transcription Pipeline's per-episode text output, with a configurable regime↔topic correlation plus deliberate noise (silent episodes, independent topics) so the alignment step has real signal *and* the two edge cases to find. |
| `narrative_topics.py` | Day 17 — online/streaming topic modeling. `create_topic_model()` now **defaults to `BERTopicWrapper`** (**S56.9**, flipped per explicit instruction) — the real `bertopic` path Bible Part 5.8 specifies, wired against BERTopic's own documented "Online Topic Modeling" example (the class had a real, now-fixed bug: `from bertopic.cluster import River` never existed; replaced with a hand-rolled `_RiverClusterWrapper`). BERTopicWrapper fires a loud `UserWarning` on every instantiation flagging it as unverified end-to-end in this dev sandbox (bertopic SIGBUS-crashes on import / needs blocked network here); pass `use_bertopic=False` for `NarrativeTopicModel`, the lightweight hashing-trick + `river.cluster.DBSTREAM` implementation, fully tested and the previous default. |
| `domain_alignment.py` | Day 17 — Fisher's exact test per (behavioral cluster, narrative topic) pair, Bonferroni-corrected across all pairs tested, p<0.05. Three-way outcome: significant co-occurrence → joint domain; behavioral cluster with no narrative partner → HIGH IGNORANCE PRIOR; narrative cluster with no behavioral partner → ASPIRATIONAL-OR-HYPOTHETICAL. Noise labels are never treated as candidates on either side. **S56.7:** optional `episode_ids` immutable join key — raises `AlignmentKeyMismatchError` (not a bare `assert`) on length mismatch or duplicate IDs, instead of relying on caller discipline that the label arrays are already aligned; omitted = unchanged backward-compatible behavior. **S56.5:** optional `subject_ids` — when supplied, the decision path (`joint_domains`) uses a subject-level cluster-bootstrap p-value (resampling whole subjects, not episodes) instead of naive per-episode Fisher's exact, fixing the repeated-same-person-window independence violation; naive p is still returned in `naive_pvalues` for comparison. **Ships without the Mandatory senior sign-off the dependence-correction method choice requires.** **S56.10:** Benjamini-Hochberg now runs alongside Bonferroni on every call (`log_correction_comparison=True` by default) and both are logged side by side; the decision path itself is unchanged (still Bonferroni per doctrine). |
| `domain_confidence.py` | Day 18 — weighted score from observation count, persistence duration, cross-phase survival (highest weight — doctrine's "strongest signal of true stability"), and behavioral-narrative coherence. **S56.4:** coherence is now a bootstrap-percentile-CI lower bound on the observed per-episode co-occurrence rate (`co_occurrence_indicator`) — a genuine effect-size/confidence-in-magnitude statement — replacing the statistically invalid `1 − p` treated as a probability. The legacy `1 − p` path remains for callers without per-episode indicators wired through yet, but now emits a `RuntimeWarning` every call flagging it as not a valid effect-size estimate. `compare_confidence_formulations` reports all three (naive, bootstrap-stability-fraction, CI-lower) side by side. Below `MIN_CONFIDENCE_THRESHOLD` (0.5) → "candidate" status only. **Ships without the Mandatory senior sign-off this metric-swap requires.** |
| `domain_lifecycle.py` | Day 18 — split/merge. **Split:** **S56.8 fix** — "sustained" now requires the *longest contiguous run* of within>between-variance windows to meet `min_sustained_windows`, not scattered qualifying windows anywhere across the whole history. Parent kept, marked inactive, never deleted; children inherit history and get fresh `DivergenceState`; parent exposes `pre_split_hold` for a future claims-engine hook. **Merge:** sustained rising co-occurrence in *both* behavioral and narrative space, both above threshold. `DomainRegistry` is append-only, same doctrine as the bifurcation log. |
| `multiple_comparisons.py` | Day 18 patched validation task — Bonferroni vs Benjamini-Hochberg (FDR) on the same raw p-values, for a direct, measured comparison (Risk 3-C: Bonferroni can under-produce domains under heavy multiple-testing load). Now actually invoked from `domain_alignment.py` on every call (S56.10), not left as an unused standalone alternative; `domain_alignment.py` still defaults to Bonferroni per doctrine. |

**Test coverage:** signature extraction, clustering (including noise
handling), synthetic transcript generation, streaming topic model (both the
lightweight default-path and the BERTopicWrapper construction/warning path),
Fisher's-exact alignment and all three outcome branches, subject-level
cluster-bootstrap vs. naive p-value comparison, immutable-join-key mismatch
handling, confidence scoring (legacy and bootstrap-CI paths), contiguous-run
split/merge lifecycle (append-only, parent preservation), Bonferroni-vs-BH
comparison logging, and the real HSSM adapter's error paths — 140 tests total
across both packages.

## Running tests

```bash
pip install pytest numpy scipy hdbscan river --break-system-packages
pytest tests/ -q
```
(`pyproject.toml` sets `pythonpath = ["."]` so `phase_transition`/
`domain_emergence` imports resolve without installing the package.)

`bertopic` is optional — install it (`pip install bertopic sentence-transformers
river --break-system-packages`) to exercise `BERTopicWrapper` end-to-end;
without it, `create_topic_model()`'s default path raises a clear
`ImportError` and the lightweight `NarrativeTopicModel` path
(`use_bertopic=False`) remains fully covered regardless.