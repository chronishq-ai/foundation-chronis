"""Production-invariant static scanner (Endgame pack, Phase A4).

Implements the four A4 CI gates as one script so `ci.yml` only needs a
single new step:

  1. bare-assert-outside-tests   (closes VF-01's failure mode)
  2. duplicate-module-basename   (closes the §1.5 duplicate-package class)
  3. retrieval-bypass-import     (protects VF-07's correct implementation)
  4. identity-fail-open (Q11)    (closes VF-09 / VF-10 / VF-11's pattern)

Design notes / known scope limits (read before extending):

* Checks 1-2-4 only see this one branch's tree. A0 (canonical tree
  consolidation) has not happened yet, so this cannot yet be run
  against the full merged bundle the pack describes. Re-run this
  script against the consolidated tree once A0 lands and re-baseline
  the allowlists at that point.
* Check 2 (duplicate module basenames) is intentionally narrow: two
  files that happen to share a filename in *different* packages (e.g.
  `loaders/base.py` and `generators/base.py`, each a small unrelated
  Protocol) are completely normal Python and are not, by themselves,
  the bug the pack describes. The bug is the same *import path*
  ending up satisfied by two different implementations after a merge.
  Since that literally cannot happen inside a single already-checked-
  out tree, this check instead flags same-basename files with
  *diverging content* as a forced human decision: either the name
  collision is coincidental and fine (add to the allowlist with a
  one-line reason) or it is an early instance of the real problem.
  This deliberately fails on first run for any unreviewed collision —
  see `allowlist_duplicate_modules.txt`.
* Check 3 (retrieval bypass) has nothing to check yet on this branch
  (`visual_memory` / `transcript_search` / `central_retrieval_core`
  live on `feat/frontier-sprints-17-18-19-20`, not here). It is wired
  in now, scanning `src/` and `synthetic/`, so that it is already
  active the moment that code merges into this tree instead of being
  added retroactively.
* Check 4 (identity fail-open) is a heuristic AST match for the exact
  shape VF-09/VF-10/VF-11 shared: `if <id> and <id> != ...:` /
  `if <id> is not None and <id> != ...:` guarding a deny/raise branch
  with no `else: raise` for the falsy/None case. It cannot prove the
  absence of every fail-open bug, only this one recurring shape, plus
  the shared/base-write-without-uid-parameter shape.

Exit status: nonzero if any check finds an unallowlisted violation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = ["src", "synthetic"]
TESTS_DIR_NAME = "tests"

# Modules that must only ever be imported from the canonical retrieval
# core. Empty today on this branch (see module docstring); kept here so
# the check activates automatically once the frontier code merges in.
RETRIEVAL_PRIMITIVES = {"visual_memory", "transcript_search"}
RETRIEVAL_CORE_SUFFIX = "central_retrieval_core.py"

IDENTITY_NAME_HINTS = ("user_id", "uid", "owner", "requesting_user")
SHARED_WRITE_NAME_HINTS = ("put_base", "shared_")


@dataclass
class Finding:
    check: str
    path: str
    line: int
    message: str


@dataclass
class ScanResult:
    findings: list[Finding] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def fail(self, check: str, path: Path, line: int, message: str) -> None:
        self.findings.append(Finding(check, path.relative_to(REPO_ROOT).as_posix(), line, message))


def _iter_py_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if TESTS_DIR_NAME in path.relative_to(REPO_ROOT).parts:
                continue
            files.append(path)
    return files


def _load_allowlist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    entries: set[str] = set()
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            entries.add(line)
    return entries


# --------------------------------------------------------------------------
# Check 1 — bare assert outside tests/
# --------------------------------------------------------------------------


def check_bare_assert(files: list[Path], allowlist: set[str], result: ScanResult) -> None:
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in allowlist:
            continue
        try:
            tree = ast.parse(path.read_text(), filename=rel)
        except SyntaxError as exc:  # pragma: no cover - defensive
            result.fail("bare-assert", path, exc.lineno or 0, f"file failed to parse: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                result.fail(
                    "bare-assert",
                    path,
                    node.lineno,
                    "bare `assert` on a production path outside tests/ — use a typed "
                    "failure (NotFittedError, ValueError, ContractError, PermissionError, "
                    "CrossUserEvidenceError, CorruptRecordError, ConvergenceError, ...) "
                    "per A3, or add this file to allowlist_bare_assert.txt with a reason "
                    "if it is genuinely test-harness code that must live outside tests/.",
                )


# --------------------------------------------------------------------------
# Check 2 — duplicate module basename with diverging content
# --------------------------------------------------------------------------


def check_duplicate_modules(files: list[Path], allowlist: set[str], result: ScanResult) -> None:
    by_basename: dict[str, list[Path]] = {}
    for path in files:
        if path.name == "__init__.py":
            continue
        by_basename.setdefault(path.name, []).append(path)

    for basename, paths in sorted(by_basename.items()):
        if len(paths) < 2:
            continue
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        distinct = set(hashes.values())
        rels = sorted(p.relative_to(REPO_ROOT).as_posix() for p in paths)
        pair_key = f"{basename}: {', '.join(rels)}"
        if len(distinct) == 1:
            # Identical content copy-pasted in two places — not the
            # branch-divergence bug, but still worth a human's attention.
            result.info.append(f"duplicate-module (identical content, not failing): {pair_key}")
            continue
        if pair_key in allowlist or basename in allowlist:
            continue
        for path in paths:
            this_rel = path.relative_to(REPO_ROOT).as_posix()
            other_rels = ", ".join(r for r in rels if r != this_rel)
            result.fail(
                "duplicate-module",
                path,
                1,
                f"basename '{basename}' is shared with diverging content by: "
                f"{other_rels}. "
                "If this is an intentional, unrelated same-named module (e.g. two "
                "small per-package Protocol files), add "
                f"'{pair_key}' to allowlist_duplicate_modules.txt with a one-line reason. "
                "If it's the real §1.5 pattern (two implementations competing to be "
                "'the' module), that must be resolved, not allowlisted.",
            )


# --------------------------------------------------------------------------
# Check 3 — retrieval-bypass import
# --------------------------------------------------------------------------


def check_retrieval_bypass(files: list[Path], result: ScanResult) -> None:
    for path in files:
        if path.name == RETRIEVAL_CORE_SUFFIX:
            continue
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:  # pragma: no cover - defensive, covered by check 1
            continue
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                segments = set(module.split("."))
                hit = segments & RETRIEVAL_PRIMITIVES
                if hit:
                    result.fail(
                        "retrieval-bypass",
                        path,
                        node.lineno,
                        f"imports {sorted(hit)} directly instead of going through "
                        "frontier.central_retrieval_core — this bypasses VF-07's "
                        "ownership-checked read path.",
                    )


# --------------------------------------------------------------------------
# Check 4 — identity fail-open (Q11)
# --------------------------------------------------------------------------


def _is_identity_like(node: ast.expr) -> bool:
    name = ""
    if isinstance(node, ast.Name):
        name = node.id
    elif isinstance(node, ast.Attribute):
        name = node.attr
    return any(hint in name for hint in IDENTITY_NAME_HINTS)


def _guard_has_else_raise(node: ast.If) -> bool:
    orelse = node.orelse
    if not orelse:
        return False
    return any(isinstance(stmt, ast.Raise) for stmt in orelse) or (
        len(orelse) == 1 and isinstance(orelse[0], ast.If) and _guard_has_else_raise(orelse[0])
    )


def _body_denies(node: ast.If) -> bool:
    return any(isinstance(stmt, ast.Raise) for stmt in node.body) or any(
        isinstance(stmt, ast.Continue) for stmt in node.body
    )


def check_identity_fail_open(files: list[Path], result: ScanResult) -> None:
    for path in files:
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:  # pragma: no cover - defensive, covered by check 1
            continue

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.BoolOp)
                and isinstance(node.test.op, ast.And)
            ):
                values = node.test.values
                if len(values) != 2:
                    continue
                first, second = values
                is_not_none_guard = (
                    isinstance(first, ast.Compare)
                    and any(isinstance(op, ast.IsNot) for op in first.ops)
                    and _is_identity_like(first.left)
                )
                bare_truthy_guard = _is_identity_like(first)
                inequality_check = (
                    isinstance(second, ast.Compare)
                    and any(isinstance(op, ast.NotEq) for op in second.ops)
                    and _is_identity_like(second.left)
                )
                if (bare_truthy_guard or is_not_none_guard) and inequality_check:
                    if _body_denies(node) and not _guard_has_else_raise(node):
                        result.fail(
                            "identity-fail-open",
                            path,
                            node.lineno,
                            "conditional matches the VF-09/VF-10/VF-11 shape: "
                            "`if <identity> and <identity> != ...:` denies on mismatch "
                            "but has no `else: raise` for the falsy/None case, so a "
                            "missing identity silently passes instead of failing closed.",
                        )

            if isinstance(node, ast.FunctionDef):
                name_matches_hint = any(
                    node.name.startswith(hint) or hint.rstrip("_") == node.name
                    for hint in SHARED_WRITE_NAME_HINTS
                )
                if name_matches_hint:
                    arg_names = {a.arg for a in node.args.args + node.args.kwonlyargs}
                    if not any(hint in a for a in arg_names for hint in IDENTITY_NAME_HINTS):
                        result.fail(
                            "identity-fail-open",
                            path,
                            node.lineno,
                            f"function '{node.name}' implies write access to a shared/base "
                            "resource but has no uid/requesting_user_id parameter at all "
                            "(Q11).",
                        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    args = parser.parse_args(argv)

    checks_dir = Path(__file__).resolve().parent
    bare_assert_allowlist = _load_allowlist(checks_dir / "allowlist_bare_assert.txt")
    duplicate_allowlist = _load_allowlist(checks_dir / "allowlist_duplicate_modules.txt")

    roots = [REPO_ROOT / r for r in args.roots]
    files = _iter_py_files(roots)

    result = ScanResult()
    check_bare_assert(files, bare_assert_allowlist, result)
    check_duplicate_modules(files, duplicate_allowlist, result)
    check_retrieval_bypass(files, result)
    check_identity_fail_open(files, result)

    for line in result.info:
        print(f"INFO: {line}")

    if not result.findings:
        print(f"prod_invariant_scan: OK ({len(files)} files scanned, 0 violations)")
        return 0

    by_check: dict[str, int] = {}
    for finding in result.findings:
        by_check[finding.check] = by_check.get(finding.check, 0) + 1
        print(f"FAIL [{finding.check}] {finding.path}:{finding.line}: {finding.message}")

    print(
        f"\nprod_invariant_scan: {len(result.findings)} violation(s) across "
        f"{len(by_check)} check(s): {by_check}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
