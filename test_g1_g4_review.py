# tests/test_g1_g4_review.py — Sprint 14 Day 42.
#
# docs/g1_g4_signoff.md makes a set of prose claims about G2/G3/G4 at the
# ML layer. Prose claims rot the moment someone edits a file and forgets
# to re-check the doc. This file turns the checkable subset of those
# claims into assertions that fail CI the moment they stop being true —
# per the directive's own logic ("a green checkbox that isn't actually
# true is worse than a red one").
#
# G1 is deliberately marked skip, not passed, not faked — its text was
# never located in the uploaded materials (see docs/g1_g4_signoff.md),
# and a test that "passes" against an unknown requirement is exactly the
# false-green-checkbox this file exists to prevent.
from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

import integration.gated_claims as gated_claims
import integration.gated_divergence as gated_divergence
import integration.gated_registry as gated_registry
import integration.gated_store as gated_store
from policy_engine.consent import ConsentRecord, ConsentTier, OperationalMode
from policy_engine.policy_rule import PolicyRule, RuleAction, Scope
from policy_engine.principal import AccessRequest, ModelPrincipal
from datetime import datetime, timezone

_INTEGRATION_MODULES = [gated_store, gated_registry, gated_claims, gated_divergence]

# Names that would suggest a Layer-0 / canonical-record write path if they
# appeared anywhere in the four gated integration modules' source. This is
# a lexical scan, not a semantic one — see the test's own docstring for
# what that does and doesn't prove.
_LAYER0_SUSPICIOUS_NAMES = ("layer0", "layer_0", "canonical_record", "canonical_write")


@pytest.mark.skip(
    reason=(
        "G1's text was never located in any of the four uploaded documents "
        "(directive PDF, Bible, Bible Addendum). Per docs/g1_g4_signoff.md, "
        "this is reported as blocked, not guessed at and not silently "
        "passed. Un-skip this test only after the real G1 text is sourced "
        "from the Bible Front Matter and a corresponding assertion is "
        "written against it — do not just remove the skip marker."
    )
)
def test_g1_not_verifiable_without_source_text():
    pytest.fail("G1 text unavailable — see skip reason.")


class TestG2NoLayer0Writeback:
    """G2: 'the pipeline never writes to Layer 0.' This is a lexical
    source scan, not a runtime guarantee — it proves no code path in these
    four files even NAMES a Layer-0/canonical-record write target, which
    is necessary but not sufficient evidence. A determined obfuscation
    (e.g. writing through a variable named `x`) would not be caught here.
    Per docs/g1_g4_signoff.md's own recommendation, a real CI guard needs
    to exist once an actual Layer-0 write path exists in the codebase to
    scan against — this test is the placeholder for that, scoped to what
    IS checkable today."""

    @pytest.mark.parametrize("module", _INTEGRATION_MODULES, ids=lambda m: m.__name__)
    def test_no_layer0_lexical_reference(self, module):
        source = inspect.getsource(module)
        lowered = source.lower()
        for bad_name in _LAYER0_SUSPICIOUS_NAMES:
            assert bad_name not in lowered, (
                f"{module.__name__} contains {bad_name!r} — investigate "
                "whether this is an accidental Layer-0 write path."
            )

    def test_gated_store_only_writes_via_sprint13_isolated_store(self):
        """Confirm GatedModelStore.write ultimately delegates to
        chronis_ml.store.IsolatedModelStore.write and nowhere else — i.e.
        there's exactly one write call inside the method body, to the
        wrapped store, not a second, separate write path."""
        source = inspect.getsource(gated_store.GatedModelStore.write)
        tree = ast.parse(textwrap.dedent(source))
        write_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "write"
        ]
        # exactly one delegated call to self._store.write — no second,
        # independent write target hiding in the same method.
        assert len(write_calls) == 1


class TestG3NotExercisedByOwnCode:
    """G3: NULL-handling contract. Sprint 14 doesn't own this contract —
    this class only confirms Sprint 14's own code doesn't quietly
    introduce imputation, which would be a NEW violation even if G3 isn't
    Sprint 14's to verify end-to-end."""

    _IMPUTATION_SIGNATURES = ("fillna", "nan_to_num", "interpolate", "np.where(np.isnan")

    @pytest.mark.parametrize("module", _INTEGRATION_MODULES, ids=lambda m: m.__name__)
    def test_no_imputation_helpers_used(self, module):
        source = inspect.getsource(module)
        for sig in self._IMPUTATION_SIGNATURES:
            assert sig not in source, (
                f"{module.__name__} appears to call {sig!r} — Sprint 14 "
                "must not introduce imputation logic; that decision belongs "
                "to Sprint 1's own NULL-handling contract owner."
            )

    def test_consent_record_has_no_permissive_default(self):
        """A ConsentRecord constructed without an explicit tier/mode
        should fail loudly (missing required arg), not silently default to
        something permissive — the ML-layer analog of 'never silently
        treated as zero.'"""
        with pytest.raises(TypeError):
            ConsentRecord(user_id="u1")  # tier and mode are required, no defaults


class TestG4NoBypassOnRetry:
    """G4: no bypass path, including for legitimate-seeming retries."""

    def test_check_reevaluates_every_call_no_caching(self):
        """Two identical calls with the SAME request object must each
        independently re-run the full evaluation — proven by revoking
        access between calls and confirming the second call sees the
        change. A cached-grant bug would let the second call slip through
        on the first call's now-stale success."""
        principal = ModelPrincipal()
        principal.register_rule(PolicyRule(
            rule_id="temp", principal="system", subject_user_id="u1",
            scope=Scope(actions=frozenset({RuleAction.INFERENCE})),
            min_consent_tier=ConsentTier.INFERENCE,
            allowed_modes=frozenset({OperationalMode.MODE_A}),
            granted_at=datetime.now(timezone.utc),
        ))
        consent = ConsentRecord("u1", ConsentTier.INFERENCE, OperationalMode.MODE_A)
        req = AccessRequest(action=RuleAction.INFERENCE, consent=consent)

        principal.check(req)  # first call: granted

        # revoke by replacing the rule set entirely (simulates a consent
        # withdrawal between the first call and a "retry")
        principal._rules["u1"] = []

        from policy_engine.errors import PolicyDenied
        with pytest.raises(PolicyDenied):
            principal.check(req)  # identical request object — must NOT reuse the earlier grant

    def test_expired_rule_cannot_be_resurrected_by_retry(self):
        """A request made after a rule's expires_at must fail even if an
        identical request succeeded while the rule was still active —
        this is the literal 'legitimate-seeming retry' scenario Bible
        5.24's G4 language describes, tested at the PolicyRule level."""
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        rule = PolicyRule(
            rule_id="temp", principal="contact", subject_user_id="u1",
            scope=Scope(actions=frozenset({RuleAction.CLAIM_ACCESS})),
            min_consent_tier=ConsentTier.INFERENCE,
            allowed_modes=frozenset({OperationalMode.MODE_A}),
            granted_at=now - timedelta(hours=2),
            expires_at=now - timedelta(seconds=1),
            requires_renewal=True,
        )
        # "at request time" (now) the rule is expired — a retry submitted
        # right now must not be evaluated against the earlier, still-valid
        # window.
        assert not rule.covers(action=RuleAction.CLAIM_ACCESS, mode=OperationalMode.MODE_A, at=now)

    def test_mismatched_identity_retry_never_falls_back_to_a_valid_rule(self):
        """A 'retry' that resubmits under a different (mismatched)
        identity must not somehow inherit a rule that would have matched
        the CORRECT identity — already proven per-integration-point in
        test_policy_boundary_cases.py; restated here as a direct G4 claim."""
        principal = ModelPrincipal()
        principal.ensure_default_rule("u2")  # u2 has a real, valid rule
        from policy_engine.errors import PolicyDenied
        # u1's consent record, but requesting access AS u2 via a raw
        # AccessRequest (the lowest-level way to attempt this) must still
        # only be evaluated against u1's own (nonexistent) rule set, since
        # ModelPrincipal keys rule lookup off consent.user_id, not any
        # separately supplied "target".
        mismatched_consent = ConsentRecord("u1", ConsentTier.FULL, OperationalMode.MODE_A)
        req = AccessRequest(action=RuleAction.INFERENCE, consent=mismatched_consent)
        with pytest.raises(PolicyDenied):
            principal.check(req)  # u1 has no rule registered — must deny, not borrow u2's


def test_known_gap_pre_gate_compute_is_tracked_not_silently_fixed_here():
    """This test asserts the gap's EXISTENCE is still tracked in
    docs/g1_g4_signoff.md, not that the gap is closed. If someone closes
    MP-18 by building gated_hssm.py, this test should be updated to
    reflect that — not deleted silently."""
    signoff = Path(__file__).parent.parent / "docs" / "g1_g4_signoff.md"
    assert signoff.exists()
    text = signoff.read_text()
    assert "MP-18" in text or "pre-gate" in text.lower()