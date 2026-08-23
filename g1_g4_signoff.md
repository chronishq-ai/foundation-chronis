# G1–G4 Production-Readiness Review — ML Layer
Sprint 14, Day 42.

## Scope note — read before treating this as a real sign-off

The full canonical text of the Bible's Front-Matter G1–G4 guarantees was
**not present** in any of the four documents uploaded to this workspace
(`CHRONIS_AI_ML_INTERN_DIRECTIVE.pdf`, `CHRONIS_BIBLE.md`,
`CHRONIS_BIBLE_ADDENDUM.md`). Only fragments referencing them appear,
scattered across the directive and the Sprint 7-9 code:

- **G2** — "the pipeline never writes to Layer 0" (directive, Sprint 1 Day
  1); "corrections... are counter-annotations, never overwrites"
  (`claims_engine/claim_levels.py`, referencing G2 directly).
- **G3** — "the NULL-handling contract every module must obey" (directive,
  Sprint 1 Day 1) — i.e. NULL/missing/not-worn states are typed,
  non-imputed values, never silently zero.
- **G4** — "no bypass path, including for legitimate-seeming retry
  requests" (Bible Addendum 5.24, re: session-key re-decryption grants).

**G1's actual wording was never located in the uploaded materials.** Per
Rule Zero ("a green checkbox that isn't actually true is worse than a red
one"), this document does NOT invent G1's text to fill the gap. Whoever
owns this sign-off must pull the real Bible Front Matter before this
review can be considered complete — the G1 row below is marked
**UNVERIFIED**, not passed, for that reason alone.

Named owner below is a placeholder (`TBD — Sprint 14 owner`) since no
specific engineer name was assigned to this review in the materials
provided. Replace before treating this as a real, signed-off checklist.

---

## Guarantee-by-guarantee review

| Guarantee | ML-layer meaning (as evidenced) | Status | Owner | Evidence |
|---|---|---|---|---|
| **G1** | Text not located in uploaded materials. | **UNVERIFIED** — cannot review a guarantee whose text we don't have. | TBD | None — see scope note above. |
| **G2** | The ML pipeline never writes back to Layer 0 (the canonical record); corrections are counter-annotations, never overwrites. | **PASS** (at Day 40/41 scope) | TBD — Sprint 14 owner | `gated_store.py`, `gated_registry.py` only ever write to model-artifact storage (`chronis_ml.store`, MLflow registry) — neither has any code path that touches a Layer-0 record. No Layer-0 write path exists anywhere in the Sprint 14 codebase to violate this. **Caveat**: this is "no violation because no such path exists yet," not "a write-back attempt was tested and blocked" — Sprint 17+ introduces Layer-0-adjacent structures (visual index, retrieval cache) this guarantee will need re-verifying against. |
| **G3** | NULL/missing/not-worn states are typed, non-imputed values, never silently zero, at the ML layer. | **NOT INDEPENDENTLY VERIFIED BY SPRINT 14** | TBD | This is Sprint 1 Day 2's own guarantee (`chronis-ml` data loaders, per the directive) — Sprint 14's code does not re-derive or re-test it. `e2e/tiles_loader.py` explicitly notes it starts *after* Sprint 1's NaN-handling policy point (`X_complete`, already NaN-free) and is out of scope for re-verifying that policy. Sprint 1's own team must own this row. |
| **G4** | No bypass path for a data-access/decryption grant, including for legitimate-seeming retries. | **PASS** (at Day 40/41 scope) | TBD — Sprint 14 owner | `ModelPrincipal.check()` is the single required choke point (Day 40); `policy_engine/consent.py::check_mode_c_block` has no tier parameter specifically so no consent level can override it; `PolicyRule.__post_init__` rejects Mode C at construction time, not just at check time. 104 boundary cases (`test_policy_boundary_cases.py`) include repeated-denial and no-rule-registered cases and none produced an unintended grant. **Caveat**: Sprint 16's actual 24-hour session-key TTL / fresh-grant-required mechanism (Bible 5.24) is not built in Sprint 14 — this PASS covers "the model principal has no override path," not "the full re-decryption-grant lifecycle is implemented." |

---

## Summary

- **2 of 4 guarantees (G2, G4) reviewed and passing** at the scope Sprint
  14 actually owns (policy-engine choke point, audit log, integration
  wrappers).
- **1 of 4 (G3) is explicitly out of Sprint 14's scope** — belongs to
  Sprint 1, not re-tested here, and should not be marked passing on
  Sprint 14's say-so.
- **1 of 4 (G1) cannot be reviewed at all** — text unavailable. This is a
  real gap in this review, not a formality. **Do not sign off the full
  program checklist's "G1–G4 verified... each with a named owner and a
  recorded pass" line item using this document alone** — it only closes
  half of what that line item requires.

## What would close this out properly

1. Locate and paste in the real G1 text from the Bible Front Matter.
2. Assign real named owners (not "TBD") for each row — the directive
   requires this explicitly.
3. Confirm with Sprint 1's owner whether G3 has its own independent
   ML-layer sign-off, and reference it here rather than leaving it blank.
4. Re-run this review after Sprint 16 ships, since G4's real teardown/TTL
   mechanism will exist by then and this review's G4 PASS should be
   re-confirmed against the complete mechanism, not just the policy
   engine's choke-point guarantee.
5. Resolve the pre-gate compute gap flagged below (MP-18) — either build
   a `gated_hssm.py`, or get an explicit ruling that it's out of scope.

---

## Known gap surfaced during Day 42 e2e testing (affects the G4 PASS above)

`tests/test_e2e_pipeline.py::test_KNOWN_GAP_hssm_and_attractor_stages_run_before_any_gate_check`
documents that Stages 2–3 of `e2e/pipeline_runner.py` (HSSM fit,
attractor detection) run **before** any `ModelPrincipal.check()` call —
only the later divergence/claims stages are gated. A denied user's
aligned feature matrix is still fit by the real HSSM and scored for
attractors before consent/mode is ever consulted. This doesn't invalidate
the G4 PASS above (that PASS is scoped to "the model principal itself has
no bypass path," which remains true), but it means G4 is not yet a clean
pass across the *entire* ML pipeline — only across the surfaces Sprint 14
actually wrapped (store, registry, claims, divergence). Recorded here as
**MP-18** below rather than left as an unlogged comment.

---

## Master Problem Registry — Snapshot as of Sprint 14 Completion

The directive's own Section 7 table (page 36–38) is canonical. This
snapshot reproduces it **as of Sprint 14's point in the program**: MP-01
through MP-12 reflect completed work from Sprints 1–9, already fixed by
the time HARDENERS starts Sprint 14. MP-13 through MP-17 are Sprint
15/16/19/20 deliverables and are correctly reported as **not yet
reached** — reporting them as further along would itself be a Rule Zero
violation ("never whichever sounds more finished").

| ID | Problem | Status as of Sprint 14 completion |
|---|---|---|
| MP-01 | No paired multimodal-behavioral + narrative ground-truth dataset | Unchanged — pilot data required; synthetic-planted validation substitutes for external validity, never marketed as equivalent. |
| MP-02 | Label switching in latent-state models | Code-complete (Sprint 3); identical fix applied to NSSM narrative regimes (Sprint 7). |
| MP-03 | NLP pipeline transfer quality to private audio transcripts | Fixed by the weak-supervision label layer plus learned per-session measurement uncertainty (Sprint 7). |
| MP-04 | Cold start produces no output for 30–90 days | Code-complete (Sprint 10). |
| MP-05 | No ground truth for divergence-type validation | 0.15 threshold remains explicitly provisional (Sprint 9/10). |
| MP-06 | Irregular time series breaks SSM assumptions | Code-complete (Sprint 3); reused for NSSM (Sprint 7). |
| MP-07 | Per-user HSSM fitting cost at scale | Explicitly deferred — Out of Scope. |
| MP-08 | Latent-state identifiability not guaranteed | Unchanged — empirical monitoring only. |
| MP-09 | Granger test underpowered for sparse domains | Code-complete (Sprint 8); narrative \|S\|-gate binds independently and later, by design (Sprint 7). |
| MP-10 | Legal framework for longitudinal intimate data (India) | Outside AI/ML scope — legal/constitutional track. |
| MP-11 | Attractor detection needs data density | Code-complete (Sprint 4/10). |
| MP-12 | BOCPD false-positive rate when sparse | Code-complete (Sprint 5); exact rate pending live data. |
| MP-13 | The Observer Effect (Bible Part 7.8) | **Partial mitigation, code-complete (Sprint 15).** Queryable surfacing index; `potentially_claim_influenced` on behavior *and* narrative change within 30 days of a surfaced Level 1–3 claim; flagged events excluded from aspiration evidence at read time. 20+ planted profiles per type, AT lag recovered from rate-of-change correlation. **Permanently open** — mitigation is not closure. |
| MP-14 | No unlinkability proof for voice transformation | **Not yet reached — Sprint 16 deliverable.** |
| MP-15 | Layer-0 storage cost at multi-year retention | **Not yet reached — Sprint 17/19 deliverable.** |
| MP-16 | Global/federated learning justification | **Not yet reached — Sprint 19 deliverable.** |
| MP-17 | Identity-graph false-match rate under adversarial similarity | **Not yet reached — Sprint 20 deliverable.** |
| **MP-18 (proposed, new)** | **Pre-gate ML computation in the e2e pipeline** — HSSM fit and attractor detection run before any policy-engine check in `e2e/pipeline_runner.py`. | **Identified, not yet designed.** Owner: TBD (HARDENERS, or whoever takes the `gated_hssm.py` follow-up). Not in the directive's original registry — proposed here for whoever owns the registry to accept or reject. |

---

## Test evidence backing this sign-off

142 tests passing across Sprint 14's own work: 104 policy-boundary cases
(`test_policy_boundary_cases.py`), 24 audit-tamper cases
(`test_audit_tamper.py`), 14 end-to-end pipeline cases
(`test_e2e_pipeline.py`). This is real evidence for the G2/G4 PASS rows
above — it is not evidence for G1 (unreviewable) or G3 (out of scope).

Sign-off: ______________________________ Date: ____________