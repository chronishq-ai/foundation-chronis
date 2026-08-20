# Dataset Loaders

## Purpose

Dataset loaders convert source-specific datasets into the canonical
Chronis data representation.

The loader layer provides a common interface for converting different
dataset formats into a consistent representation that can be consumed
by downstream Chronis ML components.

## Supported datasets

- TILES-2018 — loader interface defined, field mapping not yet
  implemented (`chronis_ml.loaders.tiles.TilesLoader` raises
  `NotImplementedError`).
- GLOBEM — implemented. Reads `rapids.csv`, `location.csv`, `screen.csv`,
  `call.csv`, `bluetooth.csv`, `steps.csv`, `sleep.csv`, `wifi.csv` from a
  `FeatureData` directory and converts each row/column into canonical
  `FeatureRecord`s.

See [missing-data.md](./missing-data.md) for how each loader represents
missing values.

## Repository architecture

The repository is organized around a canonical Chronis schema and
dataset-specific loaders.

```text
chronis-ml/
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   └── chronis_ml/
│       ├── loaders/
│       │   ├── base.py
│       │   ├── example.py
│       │   ├── globem.py
│       │   ├── tiles.py
│       │   └── utils.py
│       └── schema/
│           ├── models.py
│           └── validation.py
├── tests/
│   ├── loaders/
│   └── schema/
├── docs/
│   ├── dataset-loaders.md
│   ├── development.md
│   ├── missing-data.md
│   ├── phase1-checklist.md
│   └── repository.md
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
└── pyproject.toml
```
