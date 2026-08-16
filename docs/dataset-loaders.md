# Dataset Loaders

## Purpose

Dataset loaders convert source-specific datasets into the canonical
Chronis data representation.

## Supported datasets

- TILES-2018
- GLOBEM

## Contract

Every loader implements:

    load(config) -> ChronisDataset

## Loader responsibilities

A loader must:

1. Read approved source data.
2. Preserve participant identifiers.
3. Parse timestamps.
4. Normalize feature names.
5. Preserve source modality.
6. Preserve missingness.
7. Validate canonical output.
8. Return ChronisDataset.

## Missing data

Missing values must never be silently converted to:

- zero
- interpolated values
- fabricated measurements

Missing data must retain an explicit missing status and typed reason.

## TILES-2018

TILES field mappings must be based on the approved dataset release and
its documented schema.

The loader must not guess undocumented source fields.

## GLOBEM

GLOBEM FeatureData is converted into the canonical Chronis representation.

## Testing

Dataset loaders are tested using synthetic fixtures before real
surrogate data are processed.

## Security

Raw datasets must not be committed to Git.

Generated or downloaded datasets must remain outside the source tree
or inside ignored data directories.
