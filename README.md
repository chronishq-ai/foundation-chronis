# Chronis ML

ML infrastructure and data foundation for Chronis.

## Phase 1: Repository & Development Setup

This repository uses:

- Python 3.11
- Poetry
- pytest
- mypy
- pre-commit
- Ruff
- GitHub Actions

## Local setup

### 1. Verify Python

```bash
python --version
```

It must report Python 3.11.x.

### 2. Install dependencies

```bash
poetry install
```

### 3. Activate the environment

```bash
poetry shell
```

If your Poetry version does not provide `poetry shell`, use:

```bash
poetry run <command>
```

### 4. Run tests

```bash
poetry run pytest
```

### 5. Run type checking

```bash
poetry run mypy
```

### 6. Run linting

```bash
poetry run ruff check .
```

### 7. Run pre-commit

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

## Documentation

Further documentation lives in [`docs/`](./docs):

- [`docs/repository.md`](./docs/repository.md) — repository architecture and layout
- [`docs/dataset-loaders.md`](./docs/dataset-loaders.md) — dataset format and loader interfaces
- [`docs/missing-data.md`](./docs/missing-data.md) — typed missing-data rules and status
- [`docs/development.md`](./docs/development.md) — development workflow and data-safety rules
- [`docs/phase1-checklist.md`](./docs/phase1-checklist.md) — Phase 1 scope tracking

## Development rules

- Work on a feature branch.
- Do not self-merge into `main`.
- Every change must pass CI.
- Add tests for new behavior.
- Do not silently replace missing data with zero.
- Do not write raw/decrypted sensor data to disk.
- Do not modify Layer 0.
- Keep data access behind the approved policy boundary.
- Document important design decisions and unresolved questions.

## Current scope

Phase 1 establishes the development foundation. Dataset schemas/loaders,
missing-data implementation, MLflow, TimescaleDB, and feature extraction
are separate work items unless explicitly assigned to this branch.
