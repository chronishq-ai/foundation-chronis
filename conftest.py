"""Pytest path bootstrap — package root must be importable without PYTHONPATH."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# MLflow 3.x file store is opt-in; Sprint 13/15 tests use a local tracking URI.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# Foundation extras that need Postgres are out of Sprint 13–15 hardener scope.
collect_ignore = [
    "test_feature_store.py",
    "test_integration.py",
]
