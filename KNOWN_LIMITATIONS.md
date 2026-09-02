# KNOWN_LIMITATIONS.md

## Sprint 13 (S13.1)

`chronis_ml/train.py` is a **prototype adapter / isolation fixture; not a
foundation-model fine-tuning implementation.**

It is not LoRA, QLoRA, or PEFT. Real fine-tuning, checkpoint provenance beyond
the shared `chronis-base-v1` base gate, and adapter deletion semantics are
**Senior ML Lead** owned. Do not mark Sprint 13 "complete" on PEFT grounds.

## Sprint 14

- **S14.1 closed:** `PolicyRule.min_consent_tier` is enforced in `covers()`.
- **MP-18 open (tracked):** HSSM fit + attractor stages in
  `e2e/pipeline_runner.py` still run before any `ModelPrincipal.check()`.
  Owner: Senior ML Lead. Documented in `mp_registry.json` and
  `test_e2e_pipeline.py::test_KNOWN_GAP_...`.

## Sprint 15

- Observer-effect flags **mitigate** MP-13. They do **not** close it.
- **S15.1:** planted-profile accuracy is measured through
  `backbone.hssm` + `nssm_pipeline` + `divergence_engine.engine`.
  The Granger path is still **OLS-VAR-limited (S79.1)** — named in
  `mp_registry.json` and MLflow tags. Scratch formula is in
  `observer_effect/scratch_type_scores.py` only.
- Upstream packages bundled here are the **wiring-compatible stand-ins**
  required for this branch to install and run. Full Sprint 3/7/8 scientific
  remediations remain owned by those teams.

## Sprint 16

Out of scope. Not implemented (microVM/gVisor, Argon2id, EER unlinkability,
bystander TTL). Senior-owned ground-up build.
