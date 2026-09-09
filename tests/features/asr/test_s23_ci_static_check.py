"""S2.3 T1 — CI static check.

Test Sheet - S2.3 T1: "CI static check grep for `import openai` / `from
openai` under features/ -> Build fails if found."

This runs as a normal pytest test so it executes on every CI run, not
just as a separate manual script.
"""

from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN_IMPORT_PATTERN = re.compile(r"^\s*(import openai|from openai\b)")


def test_s23_t1_no_openai_import_anywhere_under_features() -> None:
    """Whisper must be self-hosted per the ticket's REQUIRED FIX
    ('self-hosted Whisper') — a third-party OpenAI API call anywhere
    under features/ would silently ship raw audio content off-device
    to an external API, which is exactly what self-hosting is meant to
    prevent. This check fails the build if such an import ever
    appears."""

    features_dir = Path(__file__).resolve().parents[2] / "src" / "chronis_ml" / "features"

    violations = []

    for path in features_dir.rglob("*.py"):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if FORBIDDEN_IMPORT_PATTERN.match(line):
                violations.append(f"{path}:{line_number}: {line.strip()}")

    assert not violations, (
        "forbidden `openai` import found under features/ (Whisper must be "
        "self-hosted, not called via a third-party API):\n" + "\n".join(violations)
    )


def test_s23_t1_check_actually_detects_a_real_violation(tmp_path: Path) -> None:
    """Positive control: a check that can never fail proves nothing.
    This proves the pattern genuinely catches a real violation, rather
    than the real codebase merely happening to be clean right now."""

    violating_file = tmp_path / "bad_module.py"
    violating_file.write_text("import openai\n\ndef f() -> None:\n    pass\n")

    violations = []
    file_content = violating_file.read_text(encoding="utf-8")
    for line_number, line in enumerate(file_content.splitlines(), start=1):
        if FORBIDDEN_IMPORT_PATTERN.match(line):
            violations.append(f"{violating_file}:{line_number}: {line.strip()}")

    assert violations


def test_s23_t1_check_does_not_false_positive_on_a_comment_mentioning_openai() -> None:
    """The pattern must only match real import statements, not a
    comment or string that merely mentions "openai"."""

    benign_line = "# We deliberately do NOT use openai's hosted API here."

    assert FORBIDDEN_IMPORT_PATTERN.match(benign_line) is None
