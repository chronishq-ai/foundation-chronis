# Chronos — Sprint 5-6: Phase Transition Detection & Domain Emergence

Implements the Bible's phase-transition gate (Part 5.3 / 5.23, Module 4.11) and
domain-emergence engine (Part 5.8), per the Sprint 5-6 directive (Part 13.4
Phase 3, MP-12, Risks 3-A/3-B/3-C).

## Layout

```
bocd/                   Vendored Adams & MacKay (2007) BOCPD reference impl
                         (gwgundersen/bocd, BSD-3-Clause — see bocd/LICENSE)
phase_transition/       Sprint 5 — phase transition gate + rupture detector
domain_emergence/       Sprint 6 — domain emergence engine
tests/                  Unit + regression tests for both (890 lines, 84 tests)
```

## phase_transition/ (Sprint 5, Days 13-15)

| Module | Role |
|---|---|
| `bocpd.py` | Wraps the vendored BOCPD lib. **Its output is condition 1 of 3 only** — never a declaration on its own. |
| `degradation.py` | Condition 2: fits a Gaussian on the pre-boundary window, scores its log predictive likelihood against the post-boundary window. Sharp degradation = evidence of a real regime change. |
| `stability.py` | Condition 3: tracks post-candidate regime posterior variance for ≥14 days (`min_days`). Splits the post-candidate window in half; second-half variance must drop below `drop_ratio × first-half` to count as "stabilizing." High-and-not-decreasing after the window = transient noise → reset, don't declare. |
| `gate.py` | `PhaseTransitionGate` — hard AND of all 3 conditions. Rupture evidence from `bifurcation_log.py` is **ORed into condition 2** (additional evidence), never a substitute for the degradation-score check. |
| `rupture.py` | Module 4.11. `RuptureDetector` declares a rupture only when **all four** hold simultaneously: voice energy >3σ above personal mean, PPG HR >40% above baseline, CSE salience at L5 for >10 min, and significant IMU motion disruption. Hard AND, never a weighted score. Deliberately does *not* catch slow/gradual shifts — acute-event detector only. |
| `bifurcation_log.py` | Append-only log for declared ruptures (`BifurcationEvent`, frozen dataclass). No delete/overwrite methods — separate from the ordinary recurrence pipeline. |

**Test coverage:** BOCPD wrapper, degradation scoring, stability/reset behavior,
rupture's 4-condition AND (including each condition's independent failure
mode), the composed gate, and the append-only log — including the two
Definition-of-Done cases: a synthetic transient noise spike does *not*
trigger a declaration, and a synthetic genuine transition (all 3 conditions
met) is declared within the expected latency window.

## domain_emergence/ (Sprint 6, Days 16-18)

| Module | Role |
|---|---|
| `synthetic_hssm.py` | Standalone synthetic generator matching the real `GaussianHSMM` output shape (regime sequence, transition matrix, emission params, session-indexed, NaN for missing sessions — never imputed). Lets this sprint build/test independently of Team 2's real HSSM; swap only the loader later. |
| `context_signature.py` | Turns a regime sequence + observations into one feature vector per contiguous dwell episode. Missing sessions inside an episode are excluded from mean/std, never imputed. |
| `context_clustering.py` | Day 16 — HDBSCAN over episode signatures (chosen because domain count is unknown a priori and noise is expected, not an error). Outputs *raw candidate* clusters only; label `-1` (noise) is kept, never forced into a cluster. |
| `synthetic_transcripts.py` | Mocks the Audio Transcription Pipeline's per-episode text output, with a configurable regime↔topic correlation plus deliberate noise (silent episodes, independent topics) so the alignment step has real signal *and* the two edge cases to find. |
| `narrative_topics.py` | Day 17 — online/streaming topic modeling. Real `bertopic` cannot import in this sandbox (binary crash + blocked model download), so this implements the same algorithmic shape without the dependency: hashing-trick embeddings → `river.cluster.DBSTREAM` (genuinely online, same spirit as BERTopic's default) → c-TF-IDF topic representation. A `BERTopicWrapper` with the same `.partial_fit()`/`.topics_` interface is included at the bottom of the file for swapping in on a real BERTopic install — **structurally reviewed only, never executed; run its own smoke test before trusting it.** |
| `domain_alignment.py` | Day 17 — Fisher's exact test per (behavioral cluster, narrative topic) pair, Bonferroni-corrected across all pairs tested, p<0.05. Three-way outcome: significant co-occurrence → joint domain; behavioral cluster with no narrative partner → HIGH IGNORANCE PRIOR; narrative cluster with no behavioral partner → ASPIRATIONAL-OR-HYPOTHETICAL. Noise labels are never treated as candidates on either side. Requires caller-aligned, equal-length label arrays. |
| `domain_confidence.py` | Day 18 — weighted score from observation count, persistence duration, cross-phase survival (highest weight — doctrine's "strongest signal of true stability"), and behavioral-narrative coherence (inverted alignment p-value). Below `MIN_CONFIDENCE_THRESHOLD` (0.5) → "candidate" status only. |
| `domain_lifecycle.py` | Day 18 — split/merge. **Split:** sustained within->between variance excess across ≥2 windows; parent kept, marked inactive, never deleted; children inherit history and get fresh `DivergenceState`; parent exposes `pre_split_hold` for a future claims-engine hook. **Merge:** sustained rising co-occurrence in *both* behavioral and narrative space, both above threshold. `DomainRegistry` is append-only, same doctrine as the bifurcation log. |
| `multiple_comparisons.py` | Day 18 patched validation task — Bonferroni vs Benjamini-Hochberg (FDR) on the same raw p-values, for a direct, measured comparison (Risk 3-C: Bonferroni can under-produce domains under heavy multiple-testing load). Diagnostic only; `domain_alignment.py` still defaults to Bonferroni per doctrine. |

**Test coverage:** signature extraction, clustering (including noise handling),
synthetic transcript generation, streaming topic model, Fisher's-exact
alignment and all three outcome branches, confidence scoring, split/merge
lifecycle (append-only, parent preservation), and the Bonferroni-vs-BH
comparison — 59 tests across the domain-emergence modules.


## Running tests

```bash
pip install pytest numpy scipy hdbscan river --break-system-packages
pytest tests/ -q
```
(`pyproject.toml` sets `pythonpath = ["."]` so `phase_transition`/
`domain_emergence` imports resolve without installing the package.)