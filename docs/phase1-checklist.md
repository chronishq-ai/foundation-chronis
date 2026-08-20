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

- [ ] TILES-2018 loader — interface and negative-path tests exist;
      field mapping not yet implemented
- [x] GLOBEM loader — implemented and tested, including edge cases for
      non-identifier column names, missing required columns, and
      malformed (non-numeric) values

## Data Quality & Missing Data

- [x] Typed missing-data model (`MeasurementStatus`, `MissingReason`)
- [x] Validation rules (observed/missing consistency, non-empty user_id,
      tz-aware timestamps, snake_case naming)
- [x] Malformed-data validation (non-numeric values raise, not coerced)
- [x] Dataset tests (`tests/schema/test_models.py`,
      `tests/loaders/test_globem.py`, `tests/loaders/test_utils.py`,
      `tests/loaders/test_base.py`, `tests/loaders/test_tiles.py`)
- [ ] `not_worn` reason wired to a real loader signal
- [ ] `audio_paused` reason wired to a real loader signal

## Documentation

- [x] Repository architecture (`repository.md`)
- [x] Dataset format & loader interfaces (`dataset-loaders.md`)
- [x] Missing-data rules (`missing-data.md`)
- [x] Developer setup and usage instructions (`README.md`,
      `development.md`)

## Remaining before Phase 1 is fully closed

- [ ] Implement TILES-2018 field mapping
- [ ] Wire `not_worn` / `audio_paused` to real detection logic once the
      relevant source signals are identified
- [ ] Confirm `.gitignore` / `.env.example` contents are complete
