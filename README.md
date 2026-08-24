# CHRONIS AI/ML Core Build: Sprints 9, 10 & 12

### Claims Engine · Cold Start Compass · The Mirror — Insight Generation Engine

**Primary submission: Sprint 12 — The Mirror**
**Status:** 178/178 tests passing (79 Sprint 12 · 57 Sprint 10 · 42 Sprint 9/8/7)
**Bible Traceability:** Part 5.21 (The Mirror, Module 4.10) · Part 5.10 (Cold Start Protocol) · Part 5.9 (Claims Engine)

---

## 1. What This Repository Contains

This repository delivers **The Mirror** (Sprint 12), the primary daily insight experience for CHRONIS. It is built on top of three completed upstream sprints:

| Sprint | Module | Role in this submission |
|--------|--------|------------------------|
| Sprint 9 | `claims_engine/` | Level 0–3 admissibility gates + grounded generation (dependency) |
| Sprint 10 | `cold_start/` | 5-stage Cold Start gate — Mirror silence enforced for Stage 0/1 (dependency) |
| **Sprint 12** | **`mirror/`** | **The Mirror: daily grounded insight generator (primary deliverable)** |

Sprints 7 and 8 (`nssm_pipeline/`, `divergence_engine/`) are present as transitive dependencies. They are not the subject of this submission.

---

## 2. Sprint 12 — The Mirror

**Days 34–36 | Phase VI: Production Hardening**

The Mirror generates a 100–200 word, second-person, evidence-grounded daily insight drawn from Level 1–3 claims. Every sentence is traceable to a specific `SessionExcerpt` via a citation chain. Generic coach-speak is hard-blocked before output ships.

### Module map

```
mirror/
├── insight_generator.py   Day 34 — core generator: LLM call, user vocab, citation chain
├── specificity_linter.py  Day 35 — automated quality gate (hard-blocks generic output)
├── tone_calibration.py    Day 35 — DIRECT/REFLECTIVE/WARM tone + TTS stub
├── feedback_loop.py       Day 36 — adaptive threshold (HELPFUL/NOT_YET/TOO_SOON)
├── archive.py             Day 36 — append-only InsightArchive with full-text + tag search
└── mirror_pipeline.py     orchestration (stage-first cold-start gate → generate → archive)
```

### Pipeline

```
ColdStartState.stage ∈ {STAGE_0, STAGE_1}
        ↓ True  → return None (zero output, zero LLM calls — code-enforced)

ColdStartState.can_surface_claims == False
        ↓ True  → return None (evidence gate not passed)

FeedbackLoop.is_domain_suppressed(user_id, domain_id)
        ↓ True  → return None (TOO_SOON suppression, 30 days)

divergence_state.confidence < user_adaptive_threshold
        ↓ True  → return None (adaptive bar not met)

MirrorInsightGenerator.generate()
    ├── select_excerpts()          (Sprint 9 contract: 3 supporting + 1 near-miss)
    ├── extract_user_vocabulary()  (user's own cadence injected into system prompt)
    ├── tone_system_prompt()       (DIRECT/REFLECTIVE/WARM prepended at generation time)
    ├── llm_client.generate()      (self-hosted only — never third-party API)
    ├── contains_clinical_terminology()   (Sprint 9 clinical filter — triggers human review)
    ├── _build_citation_chain()    (every sentence must attribute to a SessionExcerpt)
    └── lint_insight()             (SpecificityLinter — hard blocks if FAIL)

InsightRecord.new()  →  InsightArchive.append()
        ↓
return InsightRecord
```

### Key constraints (all code-enforced)

| Constraint | Enforcement |
|-----------|------------|
| Stage 0/1 = zero output | `_SILENT_STAGES` check before any inference runs |
| No generic coach-speak | `SpecificityLinter` 5-pattern catalogue; hard FAIL = no archive |
| Every sentence cited | `_build_citation_chain()` raises `ValueError` on unattributed sentence |
| Clinical terms → human review | `routed_to_human_review=True` + hard stop before archive (no unchecked clinical output ships) |
| Tone at generation time | System-prompt modifier prepended, not post-hoc rewriting |
| Adaptive feedback | `AdaptiveFeedbackStore`: NOT_YET +5 pp, TOO_SOON +10 pp + domain suppression |
| Append-only archive | `InsightArchive.append()` raises on duplicate (G2-compliant) |
| TTS = synthesised stub | `synthesize_voice_stub()` returns `is_stub=True`; never a recording of the user |

---

## 3. API / Interface

```python
from cold_start import evaluate_cold_start_gated, ColdStartStage
from mirror import run_mirror_pipeline, ToneMode, AdaptiveFeedbackStore, InsightArchive
from upstream_interfaces import hssm_fit_from_backbone

# 1. Get Cold Start state (Sprint 10)
cold_state = evaluate_cold_start_gated(
    day=95,
    user_id="u_001",
    n_present_sessions=45,
    fitted_hssm=hssm_fit_from_backbone(backbone_model, user_id="u_001", fit_id="..."),
    divergence_state=divergence_engine.get_state("u_001"),
)

# 2. Run The Mirror
archive        = InsightArchive()
feedback_store = AdaptiveFeedbackStore()

record = run_mirror_pipeline(
    user_id          = "u_001",
    cold_start_state = cold_state,        # gate checked first — silent if Stage 0/1
    claims           = claims_engine.get_admitted_claims("u_001"),
    candidate_excerpts = session_store.get_excerpts("u_001"),
    divergence_state = divergence_state,
    llm_client       = my_self_hosted_client,   # LLMClient Protocol — never third-party
    archive          = archive,
    feedback_store   = feedback_store,
    tone             = ToneMode.REFLECTIVE,
    domain_id        = "work",
)

if record is None:
    show_user(cold_state.user_facing_message)   # specific, never "coming soon"
else:
    show_insight(record.text)                   # 100–200 words, grounded, second person

# 3. Record feedback (updates per-user threshold adaptively)
from mirror import FeedbackRating
feedback_store.record_feedback("u_001", record.insight_id, FeedbackRating.NOT_YET, domain_id="work")

# 4. Search archive
results = archive.search("u_001", query="sustained focus", tags=["tone:reflective"])
```

---

## 4. Known Limitations

### Lexical citation attribution (explicit, not hidden)
The current implementation attributes each generated sentence to its source `SessionExcerpt` using deterministic **lexical overlap** (`_naive_attribute_sentence_to_excerpt`):

```python
# Current: token-overlap heuristic
overlap = len(sentence_tokens & excerpt_tokens)
```

This is sufficient for synthetic Sprint 12 validation — it ensures the citation-chain mechanism is testable end-to-end without a production LLM. The code and docstring explicitly flag this:

> *"In production this should be replaced by whatever citation mechanism your self-hosted LLM client supports natively (e.g. structured output with inline source tags)."*

**Before production:** replace with structured source attribution from the self-hosted generation layer.

### TTS voice stub
`synthesize_voice_stub()` returns a placeholder (`is_stub=True`), not real audio. Per Sprint 12 Day 35 spec, this is a "prosody-calibrated TTS interface stub." The production constraint — **synthesised voice, never a recording of the user** — is already enforced as a hard rule in the stub's notice field and must be preserved in any production TTS integration.

### Clinical routing
Clinical terminology detection triggers `routed_to_human_review=True` on the `InsightRecord` and blocks the insight from appearing in product output until reviewed. This follows the Sprint 9 standing 6-month mandatory human-review requirement. The routing flag is metadata; the actual review queue integration is handled by the product layer, not this module.

### LLM client
`LLMClient` is a `Protocol` (structural typing). Wire your actual **self-hosted** inference client. This module never imports or calls a third-party API.

---

## 5. Validation Results

```
tests/test_sprint12_mirror.py    79 tests  ✅ all pass
tests/test_cold_start.py         44 tests  ✅ all pass
tests/test_cold_start_wiring.py  13 tests  ✅ all pass
tests/test_sprint9_claims_engine.py        ✅ all pass
tests/test_sprint8_divergence_engine.py    ✅ all pass
tests/test_sprint7_nssm_calibration.py     ✅ all pass
─────────────────────────────────────────────────────
TOTAL                           178/178    ✅
```

### Sprint 12 DoD checklist

| Requirement | Status |
|------------|--------|
| 20 synthetic insight variants pass specificity linter regression | ✅ `test_linter_20_sample_pass` |
| 20-run end-to-end pipeline integration test (all gates, all tones) | ✅ `test_mirror_20_run_end_to_end` |
| Mirror produces zero output for Stage 0/1 — code-enforced | ✅ 6 parametrised cases |
| Every generated sentence resolves to a citation | ✅ `test_citation_chain_full_coverage` |
| NOT_YET ratings adaptively raise per-user threshold | ✅ `test_repeated_not_yet_ratchets_threshold` |
| Insight archive full-text + tag search | ✅ `test_archive_full_text_search`, `test_archive_tag_search` |
| No feedback flag surfaced as product copy | ✅ enforced by design — `FeedbackRecord` is internal-only |
| TTS stub: synthesised, not a user recording | ✅ `test_tts_stub_notice_says_synthesised` |

---

## 6. Running Tests

```bash
pip install numpy scipy statsmodels mlflow pytest --break-system-packages

cd task/
PYTHONPATH=. python3 -m pytest tests/ -v
# Expected: 178 passed
```

---

## 7. Module Dependency Graph

```
sprint 7: nssm_pipeline/          ← narrative regime process (transitive dep)
sprint 8: divergence_engine/      ← DivergenceState, TypeScores (dep)
sprint 9: claims_engine/          ← ClaimLevel, LLMClient, citation chain (dep)
sprint 10: cold_start/            ← ColdStartStage, can_surface_claims gate (dep)
                    ↓
sprint 12: mirror/                ← PRIMARY DELIVERABLE
```