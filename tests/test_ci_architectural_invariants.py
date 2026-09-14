"""XCUT-2 / R2-XINT.2 -- tests for ci/check_architectural_invariants.py
itself (re-added this pass, by explicit request, after being removed
at the user's request in a prior pass -- same script, not a new
design; see SPRINT5-6_Changes_Final.md item 8)."""
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_SCRIPT = REPO_ROOT / "ci" / "check_architectural_invariants.py"


def _load_ci_module():
    spec = importlib.util.spec_from_file_location("check_architectural_invariants", CI_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_synthetic_hssm_imports_passes_on_current_tree():
    mod = _load_ci_module()
    ok, violations = mod.check_no_synthetic_hssm_imports()
    assert ok, violations


def test_no_legacy_one_minus_p_passes_on_current_domain_confidence():
    mod = _load_ci_module()
    ok, violations = mod.check_no_legacy_one_minus_p()
    assert ok, violations


def test_no_legacy_one_minus_p_ignores_docstring_mentions(tmp_path):
    mod = _load_ci_module()
    fake_pkg = tmp_path / "domain_emergence"
    fake_pkg.mkdir()
    (fake_pkg / "domain_confidence.py").write_text(
        '"""Docs mention 1 - p here for context, on purpose."""\n'
        "def f():\n"
        "    return 1  # not the pattern\n"
    )
    orig_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        ok, violations = mod.check_no_legacy_one_minus_p()
        assert ok, violations
    finally:
        mod.REPO_ROOT = orig_root


def test_no_legacy_one_minus_p_catches_real_code_pattern(tmp_path):
    mod = _load_ci_module()
    fake_pkg = tmp_path / "domain_emergence"
    fake_pkg.mkdir()
    (fake_pkg / "domain_confidence.py").write_text(
        "def confidence(fisher_p_value):\n"
        "    return 1 - fisher_p_value\n"
    )
    orig_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        ok, violations = mod.check_no_legacy_one_minus_p()
        assert not ok
        assert violations
    finally:
        mod.REPO_ROOT = orig_root


def test_raw_variance_check_correctly_reports_known_open_item3():
    """Doc's documented prior run: this check FAILs (non-blocking) as
    long as item 3's metric-swap hasn't shipped -- proves the check
    isn't silently green on a real open item."""
    mod = _load_ci_module()
    ok, violations = mod.check_no_raw_variance_production_default()
    assert not ok
    assert violations


def test_no_bare_assert_passes_on_current_tree():
    mod = _load_ci_module()
    ok, violations = mod.check_no_bare_assert_on_production_invariants()
    assert ok, violations


def test_no_bare_assert_catches_real_bare_assert(tmp_path):
    mod = _load_ci_module()
    fake_pkg = tmp_path / "phase_transition"
    fake_pkg.mkdir()
    (fake_pkg / "something.py").write_text(
        "def f(x):\n"
        "    assert x > 0\n"
        "    return x\n"
    )
    orig_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        ok, violations = mod.check_no_bare_assert_on_production_invariants()
        assert not ok
        assert violations
    finally:
        mod.REPO_ROOT = orig_root


def test_main_exits_zero_despite_known_open_check3(capsys):
    mod = _load_ci_module()
    exit_code = mod.main()
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "KNOWN OPEN" in out