# Phase 1 Checklist

## Repository & Development Setup

- [x] Python 3.11 declared
- [x] Poetry configured
- [x] pytest configured
- [x] mypy configured
- [x] pre-commit configured
- [x] GitHub Actions CI configured (lint, format check, mypy, pytest, pre-commit)
- [x] Basic package/test added

## Data Architecture

- [x] Canonical Chronis data schema (`ChronisDataset`, `FeatureRecord`,
      `FeatureMetadata`)
- [x] Dataset-loader interface (`DatasetLoader` protocol, `LoaderConfig`,
      `ExampleLoader` reference implementation)
- [x] Feature naming/normalization convention (`normalize_feature_name`,
      snake_case validation)

## Dataset Loaders

- [x] GLOBEM loader — implemented and tested, including edge cases for
      non-identifier column names, missing required columns, and
      malformed (non-numeric) values
- [x] TILES-2018 loader mechanics — per-participant file loading,
      timezone handling, multi-day gap detection, configurable column
      mapping — implemented and tested against synthetic fixtures
- [ ] ~~TILES-2018 loader against real data~~ — **superseded.** Real
      TILES-2018 access is not available to Chronis as a commercial
      startup (research-only license/DUA). Replaced by **Sprint 1B**
      (see `sprint1b-synthetic-harness.md`): a fully self-generated
      synthetic test harness with no real third-party data anywhere in
      the repo or CI.

## Sprint 1B — Synthetic Data Test Harness

Full details: [`sprint1b-synthetic-harness.md`](./sprint1b-synthetic-harness.md).

- [x] 9 deterministic, seeded, config-driven stream generators
- [x] 6 of 8 corruption modes as record/file-level transforms;
      `participant_dropout` at the roster level; `leaky_fixture` as a
      permanent regression fixture (not a probabilistic mode, by design)
- [x] Stress-config CI run on a daily schedule
- [x] `tiles_participant_index.py` — discovery, probing, reproducible
      manifest, cache invalidation (proven with a modified-source test)
- [x] Canonical adapters — 4 distinct record kinds (point/interval/
      event/snippet), never collapsed into one shape
- [x] Full isolation test suite (Section 8), including the permanent
      `leaky_fixture` substring-collision regression case
- [x] `_generator_manifest.json` self-description per run
- [ ] Open items — honestly tracked, not silently resolved: no
      per-cell typed-missing record generation yet; `proximity`
      beacon-beacon/device-beacon variants unbuilt; `environment.device`
      assumes 1 device per participant; EMA/HR stress correlation is
      one-directional. See the doc above for full detail on each.

## Data Quality & Missing Data

- [x] Typed missing-data model (`MeasurementStatus`, `MissingReason`)
- [x] Validation rules (observed/missing consistency, non-empty user_id,
      tz-aware timestamps, snake_case naming)
- [x] Malformed-data validation (non-numeric values raise, not coerced)
- [x] `not_worn`/`audio_paused` decision table (`classify_missing_reason`)
      specified and tested in `schema/validation.py`
- [x] Dataset tests across schema, loaders, and the Sprint 1B synthetic
      harness
- [ ] `not_worn`/`audio_paused` wired into a real loader's data flow —
      explicitly out of Sprint 1 scope per the audit (Sprint 2 work)

## Documentation

- [x] Repository architecture (`repository.md`)
- [x] Dataset format & loader interfaces (`dataset-loaders.md`)
- [x] Missing-data rules (`missing-data.md`)
- [x] Developer setup and usage instructions (`README.md`,
      `development.md`)
- [x] Sprint 1B synthetic harness architecture, acceptance-checklist
      verification, and real-data transition plan
      (`sprint1b-synthetic-harness.md`)
