import sys
import types
from pathlib import Path
import numpy as np
import pytest

from domain_emergence.hssm_adapter import (
    get_hssm_output, BackboneHSSMUnavailableError, HSSMAdapterOutput,
)


def _backbone_importable() -> bool:
    try:
        import backbone.hssm  # noqa: F401
        return True
    except ImportError:
        return False


def test_raises_clear_error_when_backbone_missing(monkeypatch):
    """Explicitly simulate backbone.hssm being unimportable, rather than
    relying on the ambient test environment happening to not have it
    installed. Force-simulating the missing-package condition (instead of
    depending on sandbox ambient state) keeps this a true test of that
    failure mode regardless of whether backbone is actually installed.
    """
    import builtins
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "backbone" or name.startswith("backbone."):
            raise ImportError(f"simulated: {name} not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(BackboneHSSMUnavailableError):
        get_hssm_output(np.zeros((10, 3)))


@pytest.mark.skipif(
    not _backbone_importable(),
    reason="requires real backbone.hssm installed alongside this package "
           "(Sprint 3-4 zip) -- not available in every sandbox",
)
def test_real_backbone_integration_maps_p_t_and_echoes_matrix():
    """R2-S56.1 fix verification: against the REAL installed
    backbone.hssm.fit_hssm (no mock), the adapter must now succeed --
    not raise AttributeError -- because regime_sequence is sourced from
    HSSMResult.p_t and observations is the echoed input matrix. This
    replaces the old test of the same name that pinned the *bug*
    (AttributeError on every real call); that bug is what this pass
    fixes. The p_t->regime_sequence / echoed-matrix->observations
    mapping is flagged in hssm_adapter.py as pending formal Research ML
    sign-off (S34.7 ownership) even though it is applied here.
    """
    matrix = np.random.randn(20, 2)
    out = get_hssm_output(matrix)

    assert isinstance(out, HSSMAdapterOutput)
    assert out.regime_sequence.shape == (20,)
    assert out.observations.shape == (20, 2)
    assert np.array_equal(out.observations, np.asarray(matrix, dtype=float))


def test_adapter_calls_real_fit_hssm_no_local_reimplementation(monkeypatch):
    """S56.6 Test Sheet T1: returns the expected HSSMResult-shaped object
    via a real call to backbone.hssm.fit_hssm, no local re-implementation.
    Fake result exposes p_t (the real HSSMResult's field per S34.7), not
    the old made-up regime_sequence/observations names -- observations
    comes from the adapter echoing its own input, not from the result."""
    calls = []

    class FakeHSSMResult:
        def __init__(self):
            self.p_t = np.array([0, 0, 1, 1])

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
    assert out.observations.shape == (4, 2)
    assert np.array_equal(out.observations, matrix)


def test_missing_p_t_field_raises_attribute_error(monkeypatch):
    """Contract-mismatch safety net: if a future/broken HSSMResult lacks
    p_t entirely, the adapter must fail loudly (AttributeError naming the
    field) rather than silently substituting synthetic data."""
    class BrokenHSSMResult:
        pass  # no p_t at all

    fake_backbone = types.ModuleType("backbone")
    fake_hssm = types.ModuleType("backbone.hssm")
    fake_hssm.fit_hssm = lambda matrix: BrokenHSSMResult()
    fake_backbone.hssm = fake_hssm
    monkeypatch.setitem(sys.modules, "backbone", fake_backbone)
    monkeypatch.setitem(sys.modules, "backbone.hssm", fake_hssm)

    with pytest.raises(AttributeError, match="p_t"):
        get_hssm_output(np.zeros((2, 1)))


def test_none_p_t_field_raises_attribute_error(monkeypatch):
    """p_t present but None (HSSMResult allows `p_t: np.ndarray | None`)
    must also fail loudly, not be silently treated as an empty sequence."""
    class NonePTResult:
        def __init__(self):
            self.p_t = None

    fake_backbone = types.ModuleType("backbone")
    fake_hssm = types.ModuleType("backbone.hssm")
    fake_hssm.fit_hssm = lambda matrix: NonePTResult()
    fake_backbone.hssm = fake_hssm
    monkeypatch.setitem(sys.modules, "backbone", fake_backbone)
    monkeypatch.setitem(sys.modules, "backbone.hssm", fake_hssm)

    with pytest.raises(AttributeError, match="p_t"):
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