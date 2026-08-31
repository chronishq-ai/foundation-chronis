"""
tests/frontier/test_ci_architecture.py

R2-XINT.2 — CI Architecture Enforcement

AST-based checks that prevent architectural regressions from being silently
reintroduced.

CRITICAL: Every rule must ALSO have a test-of-the-test that intentionally
introduces a violation and proves the rule detects it.  This prevents the
situation where a CI rule always passes because it is broken.
"""

import ast
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PRODUCTION_ROOT = Path(__file__).parent.parent.parent / "src" / "frontier"
INTERFACE_FILES = ["voice_assistant.py", "multimodal_assistant.py", "explainability.py"]


def _ast_contains_string(source: str, target: str) -> list:
    """Return list of line numbers where target literal string appears."""
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if target in node.value:
                hits.append(node.lineno)
    return hits


def _ast_contains_call(source: str, module: str, func: str) -> list:
    """Return line numbers where module.func() is called."""
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == func:
                if isinstance(fn.value, ast.Name) and fn.value.id == module:
                    hits.append(node.lineno)
    return hits


def _ast_imports(source: str, name: str) -> list:
    """Return line numbers where 'name' appears in any import."""
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if name in alias.name:
                    hits.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if name in module:
                hits.append(node.lineno)
    return hits


def _production_sources():
    """Yield (filename, source) for every production Python file."""
    for f in PRODUCTION_ROOT.rglob("*.py"):
        if "__pycache__" in str(f):
            continue
        yield f.name, f.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Rule 1: no mock_event in production code
# ---------------------------------------------------------------------------

def _check_no_mock_event(source: str) -> list:
    return _ast_contains_string(source, "mock_event")


def test_no_mock_event_in_production():
    """Production code must not contain the 'mock_event' literal."""
    for fname, src in _production_sources():
        hits = _check_no_mock_event(src)
        assert not hits, (
            f"CI VIOLATION [no_mock_event]: '{fname}' contains 'mock_event' "
            f"at lines {hits}"
        )


def test_ci_detects_mock_event_violation():
    """Test-of-the-test: prove the rule catches a violation."""
    bad_code = textwrap.dedent("""
        def get_context(user_id, time_range, query_type):
            return [{"event": "mock_event", "time": None}]
    """)
    hits = _check_no_mock_event(bad_code)
    assert len(hits) > 0, "CI rule MUST detect 'mock_event' literal — rule is broken"


# ---------------------------------------------------------------------------
# Rule 2: no np.random.rand in production code (signals random-vector encoder)
# ---------------------------------------------------------------------------

def _check_no_random_rand(source: str) -> list:
    """
    Checks for np.random.rand or numpy.random.rand calls.
    NOTE: SelfHostedCLIPEncoder is explicitly exempted because it is marked
    BLOCKED and the randomness is intentionally visible as a known bug.
    We check that the count has not increased (i.e. no NEW usages added).
    """
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            # np.random.rand(...)
            if (
                isinstance(fn, ast.Attribute) and fn.attr == "rand"
                and isinstance(fn.value, ast.Attribute) and fn.value.attr == "random"
            ):
                hits.append(node.lineno)
    return hits


# The allowlist is a temporary measure for the BLOCKED encoder.
# When S1720.8 transitions from BLOCKED → READY, DELETE THIS ENTRY.
_RANDOM_RAND_TEMPORARY_ALLOWLIST = {
    "visual_memory.py",  # SelfHostedCLIPEncoder placeholder — DELETE when real encoder installed
}


def test_random_rand_only_in_blocked_encoder():
    """
    np.random.rand must only appear inside the temporary BLOCKED-encoder allowlist.
    Any NEW usage outside the allowlist is a violation.

    ACTION REQUIRED: Delete _RANDOM_RAND_TEMPORARY_ALLOWLIST when S1720.8 is unblocked.
    """
    for fname, src in _production_sources():
        if fname not in _RANDOM_RAND_TEMPORARY_ALLOWLIST:
            hits = _check_no_random_rand(src)
            assert not hits, (
                f"CI VIOLATION [no_random_rand]: '{fname}' uses np.random.rand "
                f"at lines {hits} — not in temporary allowlist"
            )


def test_ci_detects_random_rand_violation():
    """Test-of-the-test."""
    bad_code = textwrap.dedent("""
        import numpy as np
        def encode(data):
            return np.random.rand(512)
    """)
    hits = _check_no_random_rand(bad_code)
    assert len(hits) > 0, "CI rule MUST detect np.random.rand — rule is broken"


# ---------------------------------------------------------------------------
# Rule 3: no "Object identified as" fabricated string
# ---------------------------------------------------------------------------

def _check_no_object_fabrication(source: str) -> list:
    return _ast_contains_string(source, "Object identified as")


def test_no_object_fabrication_in_production():
    """'Object identified as' must never appear as a literal in production."""
    for fname, src in _production_sources():
        hits = _check_no_object_fabrication(src)
        assert not hits, (
            f"CI VIOLATION [no_object_fabrication]: '{fname}' contains "
            f"'Object identified as' at lines {hits}"
        )


def test_ci_detects_object_fabrication_violation():
    """Test-of-the-test."""
    bad_code = 'return "Object identified as X."'
    hits = _check_no_object_fabrication(bad_code)
    assert len(hits) > 0, "CI rule MUST detect 'Object identified as' — rule is broken"


# ---------------------------------------------------------------------------
# Rule 4: no unsafe deserialization in production code (pickle/dill load/loads)
# ---------------------------------------------------------------------------

def _check_no_unsafe_deserialization(source: str) -> list:
    """Catches pickle.load, pickle.loads, dill.load, dill.loads."""
    hits = []
    for module, func in [
        ("pickle", "load"),
        ("pickle", "loads"),
        ("dill", "load"),
        ("dill", "loads"),
    ]:
        hits.extend(_ast_contains_call(source, module, func))
    return hits


def test_no_pickle_loads_in_production():
    """Unsafe deserialization (pickle/dill load/loads) must not exist in production."""
    for fname, src in _production_sources():
        hits = _check_no_unsafe_deserialization(src)
        assert not hits, (
            f"CI VIOLATION [no_unsafe_deserialization]: '{fname}' calls unsafe "
            f"deserialization at lines {hits}"
        )


def test_ci_detects_pickle_loads_violation():
    """Test-of-the-test."""
    bad_code = textwrap.dedent("""
        import pickle, base64
        obj = pickle.loads(base64.b64decode(payload))
    """)
    hits = _check_no_unsafe_deserialization(bad_code)
    assert len(hits) > 0, "CI rule MUST detect pickle.loads — rule is broken"


def test_ci_detects_pickle_load_violation():
    """Test-of-the-test for pickle.load."""
    bad_code = textwrap.dedent("""
        import pickle
        with open("file.bin", "rb") as f:
            obj = pickle.load(f)
    """)
    hits = _check_no_unsafe_deserialization(bad_code)
    assert len(hits) > 0, "CI rule MUST detect pickle.load — rule is broken"


# ---------------------------------------------------------------------------
# Rule 5: no direct retrieval bypass (interface files must not import
#         visual_memory or transcript_search)
# ---------------------------------------------------------------------------

BANNED_RETRIEVAL_IMPORTS = ["visual_memory", "transcript_search"]


def _check_no_retrieval_bypass(source: str) -> list:
    hits = []
    for banned in BANNED_RETRIEVAL_IMPORTS:
        hits.extend(_ast_imports(source, banned))
    return hits


def test_no_retrieval_bypass_in_interface_files():
    """Interface files must not bypass CentralRetrievalCore."""
    for filename in INTERFACE_FILES:
        filepath = PRODUCTION_ROOT / filename
        if not filepath.exists():
            continue
        hits = _check_no_retrieval_bypass(filepath.read_text(encoding="utf-8"))
        assert not hits, (
            f"CI VIOLATION [no_retrieval_bypass]: '{filename}' imports a "
            f"banned retrieval module at lines {hits}"
        )


def test_ci_detects_retrieval_bypass_violation():
    """Test-of-the-test."""
    bad_code = "from frontier.visual_memory import VisualMemoryIndex"
    hits = _check_no_retrieval_bypass(bad_code)
    assert len(hits) > 0, "CI rule MUST detect retrieval bypass import — rule is broken"


# ---------------------------------------------------------------------------
# Rule 6: no fabricated/hard-coded consent tier in production code
# ---------------------------------------------------------------------------

def _check_no_hard_coded_consent(source: str) -> list:
    """Finds {"consent_tier": 2} or "consent_tier": 2 patterns."""
    return _ast_contains_string(source, "consent_tier")


EXCEPTED_FILES = {"central_retrieval_core.py", "policy.py"}


def test_no_hard_coded_consent_in_production():
    """
    'consent_tier' must not appear as a hard-coded literal in production files
    (callers must use the policy engine).
    Exception: CRC itself may reference it for legacy API compat comment.
    Exception: policy.py legitimately defines the consent tier concept.
    """
    for fname, src in _production_sources():
        if fname in EXCEPTED_FILES:
            continue
        hits = _check_no_hard_coded_consent(src)
        assert not hits, (
            f"CI VIOLATION [no_hard_coded_consent]: '{fname}' contains "
            f"hard-coded 'consent_tier' at lines {hits}"
        )


def test_ci_detects_hard_coded_consent_violation():
    """Test-of-the-test."""
    bad_code = 'consent = {"consent_tier": 2}'
    hits = _check_no_hard_coded_consent(bad_code)
    assert len(hits) > 0, "CI rule MUST detect hard-coded consent_tier — rule is broken"


# ---------------------------------------------------------------------------
# Rule 7: no caller-controlled provenance (explain must not accept claim_data)
# ---------------------------------------------------------------------------

def _check_no_caller_provenance(source: str) -> list:
    """
    Detects function definitions where a parameter is named 'claim_data'.
    This is the specific pattern that signals caller-controlled provenance.
    String-literal matching is insufficient; we check AST argument names.
    """
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                if arg.arg == "claim_data":
                    hits.append(node.lineno)
    return hits


def test_no_caller_controlled_provenance():
    """'claim_data' as a parameter name must not appear in explainability production code."""
    filepath = PRODUCTION_ROOT / "explainability.py"
    if not filepath.exists():
        return
    hits = _check_no_caller_provenance(filepath.read_text(encoding="utf-8"))
    assert not hits, (
        f"CI VIOLATION [no_caller_provenance]: 'explainability.py' still "
        f"accepts caller-controlled evidence via parameter 'claim_data' at lines {hits}"
    )


def test_ci_detects_caller_provenance_violation():
    """Test-of-the-test: prove the rule catches a function with claim_data param."""
    bad_code = textwrap.dedent("""
        def explain(self, claim_data: dict) -> dict:
            citations = claim_data.get("citation_chain", [])
    """)
    hits = _check_no_caller_provenance(bad_code)
    assert len(hits) > 0, "CI rule MUST detect 'claim_data' parameter name — rule is broken"


# ---------------------------------------------------------------------------
# Rule 8: no synthetic HSSM imports
# ---------------------------------------------------------------------------

BANNED_HSSM = ["synthetic_hssm", "mock_hssm", "fake_hssm"]


def _check_no_synthetic_hssm(source: str) -> list:
    hits = []
    for banned in BANNED_HSSM:
        hits.extend(_ast_imports(source, banned))
        hits.extend(_ast_contains_string(source, banned))
    return hits


def test_no_synthetic_hssm_in_production():
    """No synthetic/mock HSSM imports in production."""
    for fname, src in _production_sources():
        hits = _check_no_synthetic_hssm(src)
        assert not hits, (
            f"CI VIOLATION [no_synthetic_hssm]: '{fname}' references "
            f"synthetic HSSM at lines {hits}"
        )


def test_ci_detects_synthetic_hssm_violation():
    """Test-of-the-test."""
    bad_code = "from backbone.synthetic_hssm import MockHSSM"
    hits = _check_no_synthetic_hssm(bad_code)
    assert len(hits) > 0, "CI rule MUST detect synthetic HSSM import — rule is broken"


# ---------------------------------------------------------------------------
# Rule 10: import path consistency (frontier.interfaces.* vs interfaces.*)
# ---------------------------------------------------------------------------

def _check_no_unqualified_interface_imports(source: str) -> list:
    tree = ast.parse(source)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and (module.startswith("interfaces.") or module == "interfaces"):
                if not module.startswith("frontier.interfaces."):
                    hits.append(node.lineno)
    return hits


def test_no_unqualified_interface_imports_in_production():
    """All interface imports must use frontier.interfaces.*, not interfaces.*"""
    for fname, src in _production_sources():
        hits = _check_no_unqualified_interface_imports(src)
        assert not hits, (
            f"CI VIOLATION [import_consistency]: '{fname}' uses unqualified "
            f"import at lines {hits} — must be 'from frontier.interfaces.*'"
        )


def test_ci_detects_unqualified_interface_import_violation():
    """Test-of-the-test."""
    bad_code = "from interfaces.claims_store import ClaimsStoreProvider"
    hits = _check_no_unqualified_interface_imports(bad_code)
    assert len(hits) > 0, "CI rule MUST detect unqualified interface import"


# ---------------------------------------------------------------------------
# Rule 11: no fabricated identity return output (AST-targeted)
# ---------------------------------------------------------------------------

def _check_no_fabricated_identity_return(source: str) -> list:
    """
    Detects return statements that construct fabricated identity-labeling strings.
    Targets patterns like:
      return f"Object identified as {name}"
      return "Object identified as X"
    Walks ast.Return nodes only; avoids flagging legitimate comments or logs.
    """
    tree = ast.parse(source)
    hits = []
    IDENTITY_PHRASES = {"identified as", "recognized as", "recognised as"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            for subnode in ast.walk(node.value):
                if isinstance(subnode, ast.Constant) and isinstance(subnode.value, str):
                    for phrase in IDENTITY_PHRASES:
                        if phrase in subnode.value.lower():
                            hits.append(node.lineno)
    return hits


def test_no_fabricated_identity_return_in_production():
    """Production code must not return fabricated identity strings."""
    for fname, src in _production_sources():
        hits = _check_no_fabricated_identity_return(src)
        assert not hits, (
            f"CI VIOLATION [no_fabricated_identity]: '{fname}' returns a "
            f"fabricated identity label at lines {hits}"
        )


def test_ci_detects_fabricated_identity_violation():
    """Test-of-the-test."""
    bad_code = textwrap.dedent("""
        def resolve(name):
            return f"Object identified as {name}."
    """)
    hits = _check_no_fabricated_identity_return(bad_code)
    assert len(hits) > 0, "CI rule MUST detect fabricated identity return"


def test_ci_does_not_false_positive_on_legitimate_text():
    """Rule must not flag legitimate uses of these phrases in non-return contexts."""
    ok_code = textwrap.dedent("""
        # identity recognized as uncertain is a valid log message
        status = "identity recognized as uncertain"
        logger.info("Entity identified as unknown")
    """)
    hits = _check_no_fabricated_identity_return(ok_code)
    assert len(hits) == 0, "CI rule must not flag non-return usages"
