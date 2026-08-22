import sys
import types
from pathlib import Path
import numpy as np
import pytest

from domain_emergence.hssm_adapter import (
    get_hssm_output, BackboneHSSMUnavailableError, HSSMAdapterOutput,
)


def test_raises_clear_error_when_backbone_missing():
    # backbone package is not part of this delivered zip -- must fail
    # loudly and specifically, not silently fall back to synthetic data.
    with pytest.raises(BackboneHSSMUnavailableError):
        get_hssm_output(np.zeros((10, 3)))


def test_adapter_calls_real_fit_hssm_no_local_reimplementation(monkeypatch):
    """S56.6 Test Sheet T1: returns the expected HSSMResult-shaped object
    via a real call to backbone.hssm.fit_hssm, no local re-implementation."""
    calls = []

    class FakeHSSMResult:
        def __init__(self):
            self.regime_sequence = np.array([0, 0, 1, 1])
            self.observations = np.array([[1.0], [2.0], [3.0], [4.0]])

    def fake_fit_hssm(matrix):
        calls.append(matrix)
        return FakeHSSMResult()

    fake_backbone = types.ModuleType("backbone")
    fake_hssm = types.ModuleType("backbone.hssm")
    fake_hssm.fit_hssm = fake_fit_hssm
    fake_backbone.hssm = fake_hssm
    monkeypatch.setitem(sys.modules, "backbone", fake_backbone)
    monkeypatch.setitem(sys.modules, "backbone.hssm", fake_hssm)

    matrix = np.ones((4, 2))
    out = get_hssm_output(matrix)

    assert len(calls) == 1
    assert np.array_equal(calls[0], matrix)
    assert isinstance(out, HSSMAdapterOutput)
    assert np.array_equal(out.regime_sequence, [0, 0, 1, 1])
    assert out.observations.shape == (4, 1)


def test_missing_field_raises_attribute_error(monkeypatch):
    class BrokenHSSMResult:
        def __init__(self):
            self.regime_sequence = np.array([0, 1])
            # observations deliberately missing

    fake_backbone = types.ModuleType("backbone")
    fake_hssm = types.ModuleType("backbone.hssm")
    fake_hssm.fit_hssm = lambda matrix: BrokenHSSMResult()
    fake_backbone.hssm = fake_hssm
    monkeypatch.setitem(sys.modules, "backbone", fake_backbone)
    monkeypatch.setitem(sys.modules, "backbone.hssm", fake_hssm)

    with pytest.raises(AttributeError):
        get_hssm_output(np.zeros((2, 1)))


def test_no_production_import_of_synthetic_stand_in():
    """S56.6 Test Sheet T3: static scan of production IMPORT paths for
    the retired synthetic_hssm stand-in -- zero import references outside
    test files. (Doc/comment mentions pointing readers at the relocated
    fixture, e.g. in hssm_adapter.py's own docstring, are fine and are
    not import statements.)

    Pure-Python pathlib/re scan, not a shelled-out `grep` subprocess call:
    the original used `__file__.rsplit("/tests/", 1)` (POSIX-slash-only,
    breaks on Windows backslash paths) and depended on a `grep` binary
    being on PATH (not guaranteed on Windows). Neither is a statement
    about the production code itself -- both are portability bugs in the
    test harness, fixed here without changing what T3 actually checks."""
    import re

    repo_root = Path(__file__).resolve().parent.parent
    pattern = re.compile(r"^\s*(from|import)\s+.*synthetic_hssm")

    hits = []
    for pkg in ("domain_emergence", "phase_transition"):
        for py_file in (repo_root / pkg).rglob("*.py"):
            for lineno, line in enumerate(
                py_file.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if pattern.match(line):
                    hits.append(f"{py_file.relative_to(repo_root)}:{lineno}: {line}")

    assert hits == [], f"production code still imports synthetic_hssm: {hits}"
