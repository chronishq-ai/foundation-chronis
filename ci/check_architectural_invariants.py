#!/usr/bin/env python3
"""XCUT-2 / R2-XINT.2 -- CI enforcement of the 4 architectural invariants
this pack's closure gate has relied on manual review for.

RE-ADDED (this pass, by explicit request): this script + the paired
`.github/workflows/ci.yml` were built and verified once before (3/4
PASS, raw-variance check correctly FAILing on item 3), then removed at
the user's request in that same pass. Re-adding now does not change
any of the 4 checks' semantics or thresholds from that prior run --
this is the same script, not a new design.

Checks (run against `phase_transition/`, `domain_emergence/`, `bocd/`
only -- production packages, never `tests/`):

  1. No `synthetic_hssm` import in production code. The synthetic HSSM
     generator is test-fixture-only (`tests/fixtures/synthetic_hssm_fixture.py`,
     relocated there per S56.6) and must never be importable from a
     production module, even transitively, or a "real" HSSM path could
     silently be running on fake data.
  2. No legacy `1 - p` / `one_minus_p` domain-confidence formula in
     `domain_emergence/domain_confidence.py` specifically (item 5).
     Detected via `tokenize`, not a plain string grep, so a mention in
     a comment or docstring does not trip it -- only code tokens count.
     `domain_emergence/legacy_diagnostic.py` is the explicit quarantine
     module for this formula and is exempted by design.
  3. No raw-variance metric used as the ONLY/DEFAULT production
     stability path (item 3's still-open metric-swap decision). This
     check is EXPECTED TO FAIL until item 3's Senior-owned metric swap
     ships -- it exists to make that open item visible in CI, not to
     block unrelated work. It passes only once
     `RegimeStability.is_stabilizing`'s raw-variance metric is no
     longer reachable as a production default (e.g. entropy becomes
     the only path, or the raw-variance path is moved under
     `diagnostics/` the way item 4's scalar-Gaussian model was).
  4. No bare `assert` statement on a production-path invariant in
     `phase_transition/`, `domain_emergence/`, `bocd/` (PH0.2). Bare
     asserts are stripped under `python -O` and silently vanish in
     production, unlike a real raised exception.

Exit code is nonzero if any check other than #3 fails, so CI blocks
merges on regressions to checks 1/2/4 while still surfacing check 3's
known-open status every run instead of hiding it.
"""
from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_DIRS = ["phase_transition", "domain_emergence", "bocd"]

# item 5's quarantine module is explicitly exempt from check 2 -- it
# exists SPECIFICALLY to hold the old formula for migration comparison.
LEGACY_DIAGNOSTIC_MODULE = "domain_emergence/legacy_diagnostic.py"

# item 4's diagnostic scalar-Gaussian model is explicitly exempt from
# check 3's raw-variance language where it overlaps in wording -- it is
# a different (already-quarantined, tagged) model, not the stability
# raw-variance path check 3 is about. Nothing to exempt today since
# check 3 only inspects phase_transition/stability.py and gate.py.


def _iter_production_py_files():
    for d in PRODUCTION_DIRS:
        base = REPO_ROOT / d
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def check_no_synthetic_hssm_imports() -> tuple[bool, list[str]]:
    violations = []
    for path in _iter_production_py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as e:
            violations.append(f"{path}: SyntaxError while parsing: {e}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "synthetic_hssm" in alias.name:
                        violations.append(f"{path}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if "synthetic_hssm" in mod:
                    violations.append(f"{path}:{node.lineno}: from {mod} import ...")
    return (len(violations) == 0, violations)


def check_no_legacy_one_minus_p() -> tuple[bool, list[str]]:
    """Tokenize-based: only NAME/OP/NUMBER tokens count, so a mention
    inside a comment or docstring can never satisfy or trip this."""
    target = REPO_ROOT / "domain_emergence" / "domain_confidence.py"
    if not target.exists():
        return (True, [])  # nothing to check is not a failure

    source = target.read_text(encoding="utf-8")
    tokens = [
        tok for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                             tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING,
                             tokenize.ENDMARKER)
        and tok.type != tokenize.STRING  # excludes docstrings/string literals
    ]
    violations = []
    for i in range(len(tokens) - 2):
        a, b, c = tokens[i], tokens[i + 1], tokens[i + 2]
        # pattern: NUMBER(1) OP(-) NAME(p / fisher_p_value / ...)
        if (a.string == "1" and b.string == "-"
                and c.type == tokenize.NAME and "p" in c.string.lower()
                and ("p_value" in c.string.lower() or c.string.lower() == "p")):
            violations.append(f"{target}:{a.start[0]}: literal '1 - {c.string}' pattern")
    for tok in tokens:
        if tok.type == tokenize.NAME and tok.string == "one_minus_p":
            violations.append(f"{target}:{tok.start[0]}: 'one_minus_p' identifier")
    return (len(violations) == 0, violations)


def check_no_raw_variance_production_default() -> tuple[bool, list[str]]:
    """EXPECTED TO FAIL until item 3's metric-swap ships (Senior-owned,
    stability.py's own HONESTY FLAG says 'DO NOT MERGE without that
    review'). Reports status, does not silently mark it green."""
    stability = REPO_ROOT / "phase_transition" / "stability.py"
    gate = REPO_ROOT / "phase_transition" / "gate.py"
    violations = []
    if stability.exists():
        src = stability.read_text(encoding="utf-8")
        if "def is_stabilizing(" in src:
            violations.append(
                f"{stability}: raw-variance `is_stabilizing` is still defined "
                "and reachable as a production method (item 3 metric-swap "
                "not yet done -- this is a KNOWN OPEN item, not a regression)."
            )
    if gate.exists():
        src = gate.read_text(encoding="utf-8")
        if "require_regime_probabilities: bool = False" in src:
            violations.append(
                f"{gate}: PhaseTransitionGate defaults to allowing the "
                "raw-variance fallback (`require_regime_probabilities=False` "
                "default) -- entropy is opt-in, not the only production "
                "path (item 3, KNOWN OPEN)."
            )
    return (len(violations) == 0, violations)


def check_no_bare_assert_on_production_invariants() -> tuple[bool, list[str]]:
    violations = []
    for path in _iter_production_py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as e:
            violations.append(f"{path}: SyntaxError while parsing: {e}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                violations.append(f"{path}:{node.lineno}: bare `assert` statement")
    return (len(violations) == 0, violations)


CHECKS = [
    ("1. no synthetic_hssm imports in production", check_no_synthetic_hssm_imports, True),
    ("2. no legacy 1-p domain-confidence path", check_no_legacy_one_minus_p, True),
    ("3. no raw-variance production stability default", check_no_raw_variance_production_default, False),
    ("4. no bare assert on production invariants", check_no_bare_assert_on_production_invariants, True),
]


def main() -> int:
    overall_blocking_ok = True
    for name, fn, blocking in CHECKS:
        ok, violations = fn()
        status = "PASS" if ok else "FAIL"
        blocking_tag = "" if blocking else "  [KNOWN OPEN -- non-blocking, item 3]"
        print(f"[{status}] {name}{blocking_tag}")
        for v in violations:
            print(f"    - {v}")
        if not ok and blocking:
            overall_blocking_ok = False
    print()
    if overall_blocking_ok:
        print("All blocking invariants hold. (Check 3 status reported above regardless.)")
    else:
        print("One or more BLOCKING invariants failed.")
    return 0 if overall_blocking_ok else 1


if __name__ == "__main__":
    sys.exit(main())