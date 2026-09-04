# FAD-001 — Audio & Voice Data Access Policy

**Status:** Approved by founder, engineering mechanism built and tested
(T4A/T4B). **Not yet reconciled against `CHRONIS_BIBLE.md`** — see
"Required follow-up" below. Do not treat the behavior described here as
fully canonical until that reconciliation happens.

## Context

The Sprint 1 remediation audit's S1.1/T4 ticket required OMSignal data
to be structurally distinguishable from audio data, and a separate,
explicit classification path for audio-derived artifacts. Resolving
that ticket required an actual product/policy decision — how should
Chronis handle raw audio, derived voice features, and voice
fingerprints — which had been an open question, not a settled spec.

That decision has now been made by the founder and is recorded here.

## Decision

1. **Derived voice/prosody features** (pitch, speaking rate, pauses,
   energy, voice quality, etc.) — **may be used** for AI inference when
   the user's selected mode/consent permits the relevant audio
   processing.

2. **Raw recorded audio** — **may be stored and processed** when the
   user's selected mode/consent permits raw-audio capture/retention.

3. **Voice fingerprints / recurring-speaker signatures** — **may be
   used internally only**, to recognize that the same opaque speaker
   appears across sessions. Requirements: internal use only; opaque
   identity by default; no automatic real-world name; no public/
   family-facing identity inference merely because a fingerprint
   matches; user-assigned identity is a separate, later product
   decision.

4. **Third-party access** — **deny by default.** Partner/family/other
   third parties receive none of the above (raw audio, derived voice
   data, voice fingerprints, transcripts) automatically. Access
   requires an explicit user grant through the constitutional
   permission system.

## Why this needed a formal decision record, not a silent code change

Decision #2 directly contradicts existing wording in
`CHRONIS_BIBLE.md` stating the model layer does not retain raw data.
Per explicit instruction accompanying this decision: *"Do not silently
implement the contradiction. Record a formal Founder Architecture
Decision and update/reconcile the Bible/policy specification before
treating the new behavior as canonical."* This document is that
record. The Bible itself has not been edited as part of this change —
see required follow-up.

## What was built against this decision (T4A / T4B)

- `src/chronis_ml/schema/classification.py` — `DataClassification`,
  `DataSource`, `ObjectType`, `Representation`. Structurally prevents
  invalid combinations (e.g. OMSignal classified as audio) at
  construction time, both for new objects and on reload from storage.
- `src/chronis_ml/schema/policy.py` — `evaluate_access()`, encoding
  the 4 rules above as a testable policy table, with a `ConsentContext`
  Protocol documenting the exact interface the real constitutional/
  consent system needs to satisfy to plug in.
- Full test coverage: 23 tests on the classification schema (including
  every required negative test and a persistence round-trip), 11 tests
  on the policy rules (one group per founder rule).

## What was explicitly NOT done (honesty, not oversight)

- **The real constitutional/consent system was not touched or wired
  in.** I do not have visibility into that system's actual code in
  this repository or conversation. `policy.py`'s `ConsentContext`
  Protocol defines the exact shape it needs; connecting a real
  implementation is the remaining integration step, and needs someone
  with access to that system.
- **`CHRONIS_BIBLE.md` was not edited.** I have never seen this
  document's full content — only excerpts quoted inside audit PDFs.
  Editing a document I cannot read in full risks introducing an
  inaccurate diff. Whoever owns that file needs to reconcile Decision
  #2 against its current "no raw retention" wording directly.
- **T4C (real TILES field mapping)** remains blocked on lawful data
  access, unaffected by this decision.

## Required follow-up (not closeable by engineering alone)

1. Whoever owns `CHRONIS_BIBLE.md` reconciles Decision #2 against its
   current wording.
2. Whoever owns the real constitutional/consent system implements
   `ConsentContext` against `evaluate_access()` and confirms — per the
   resolved direction — that this "correctly enforces the
   founder-approved policy," not whether the policy itself is right
   (that's already decided).
3. Once both of the above land, T4B can move from "mechanism built and
   tested" to fully `CLOSED`.
