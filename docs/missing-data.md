# Missing-Data Rules

## Principle

Missing data is never represented as zero, an empty string, or any other
sentinel numeric value. A missing measurement is a distinct, typed state
in the canonical schema.

## Representation

Every `FeatureRecord` carries a `status: MeasurementStatus`:

- `MeasurementStatus.OBSERVED` — the record has a real `value` and no
  `missing_reason`.
- `MeasurementStatus.MISSING` — the record has `value = None` and a
  required, typed `missing_reason`.

These two states are mutually exclusive and enforced by
`chronis_ml.schema.validation.validate_record`:

- An observed record with `value is None` is rejected.
- An observed record with a `missing_reason` set is rejected.
- A missing record with a non-`None` value is rejected.
- A missing record without a `missing_reason` is rejected.

This is validated automatically whenever `validate_dataset` runs, which
every loader calls before returning a `ChronisDataset`.

## Typed reasons (`MissingReason`)

| Reason | Meaning | Currently assigned by |
|---|---|---|
| `sensor_failure` | The source recorded a null/NaN value for this feature at this timestamp. | GLOBEM loader (any `NaN` cell) |
| `not_worn` | The wearable/device was not being worn, so no reading was possible. | Not yet wired to any loader |
| `audio_paused` | Audio capture was intentionally paused (e.g. for privacy), so no reading exists. | Not yet wired to any loader |

**Current status:** only `sensor_failure` is actually assigned today, and
only by the GLOBEM loader, which maps every `NaN` cell in a GLOBEM feature
file to `sensor_failure` unconditionally. `not_worn` and `audio_paused`
exist in the schema and are covered by schema-level tests
(`tests/schema/test_models.py`), but no loader currently has the
source-data signal needed to distinguish them from a generic sensor
failure.

Wiring these up requires dataset-specific detection logic once the
relevant source fields are identified, for example:

- A GLOBEM/TILES wear-status or device-attachment column, if present in
  the raw feature files, would justify `not_worn`.
- An explicit audio-pause flag or gap marker in an audio-derived modality
  would justify `audio_paused`.

This is tracked as open Part 1 work, not yet implemented.

## Malformed vs. missing

A value that is *present* but cannot be interpreted correctly (wrong
type, unparseable string, etc.) is **malformed data**, not missing data.
Loaders must raise a clear `ValueError` in this case rather than:

- silently coercing it to `0` or `None`, or
- silently dropping the row.

The GLOBEM loader enforces this in `_coerce_numeric`, which raises with
the filename, column, `pid`, and `date` on any non-numeric, non-`NaN`
cell.

## Adding a new missing reason

1. Add the value to `MissingReason` in `schema/models.py`.
2. Confirm the schema-level validation rules in `schema/validation.py`
   still hold (no changes should be required — the rules are reason-agnostic).
3. Add loader-specific detection logic and pass the correct
   `MissingReason` into `build_missing_record`.
4. Add a loader-level test proving the reason is assigned correctly, in
   addition to the existing schema-level tests.
