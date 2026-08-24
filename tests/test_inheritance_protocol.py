"""
tests/test_inheritance_protocol.py

Written BEFORE inheritance_protocol.py's logic.

This module reuses Mansi's real generate_insight() pipeline (Sprint 9),
which is NOT importable here yet (separate package). Rather than fake
her real logic, we dependency-inject a stub "insight generator" callable
that matches her exact real signature -- generate_insight(claim,
divergence_state, candidate_excerpts, llm_client) -- and prove our
ORCHESTRATION logic (which claim to pick, how to wrap the result, never
faking a signature) is correct. The real Mansi function gets swapped in
at integration time with zero changes needed here.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fixtures.synthetic_user_profile import (
    build_claims, USER_ID, build_session_excerpts,
)
from behavioral_dna import build_behavioral_dna_export
from inheritance_protocol import build_inheritance_letter, NoEligibleClaimError


class FakeGeneratedInsight:
    """Stand-in for Mansi's real GeneratedInsight dataclass -- only the
    two fields inheritance_protocol.py actually reads."""
    def __init__(self, text, citation_chain):
        self.text = text
        self.citation_chain = citation_chain


def fake_insight_generator(claim, divergence_state, candidate_excerpts, llm_client):
    """Records what it was called with, so tests can inspect the call --
    and returns a fake insight shaped like Mansi's real return type."""
    fake_insight_generator.last_call = {
        "claim": claim,
        "divergence_state": divergence_state,
        "candidate_excerpts": candidate_excerpts,
        "llm_client": llm_client,
    }
    return FakeGeneratedInsight(
        text=f"A short letter about {claim.claim_id}.",
        citation_chain=["session-01", "session-02"],
    )


def test_picks_the_most_recently_created_admissible_claim():
    """claim-005 was planted with a LATER created_at than claim-001 --
    the letter should be built from claim-005, not the older one."""
    claims = build_claims()
    export = build_behavioral_dna_export(user_id=USER_ID, claims=claims)
    excerpts = build_session_excerpts()

    letter = build_inheritance_letter(
        export=export,
        divergence_state=None,   # opaque pass-through, see module docstring
        candidate_excerpts=excerpts,
        insight_generator=fake_insight_generator,
        llm_client=None,
    )

    assert letter.source_claim_id == "claim-005"


def test_forwards_the_correct_claim_and_excerpts_to_the_generator():
    claims = build_claims()
    export = build_behavioral_dna_export(user_id=USER_ID, claims=claims)
    excerpts = build_session_excerpts()

    build_inheritance_letter(
        export=export,
        divergence_state="fake-divergence-state-object",
        candidate_excerpts=excerpts,
        insight_generator=fake_insight_generator,
        llm_client="fake-llm-client",
    )

    last_call = fake_insight_generator.last_call
    assert last_call["claim"].claim_id == "claim-005"
    assert last_call["candidate_excerpts"] == excerpts
    assert last_call["divergence_state"] == "fake-divergence-state-object"
    assert last_call["llm_client"] == "fake-llm-client"


def test_letter_text_and_citations_come_from_the_generator_result():
    claims = build_claims()
    export = build_behavioral_dna_export(user_id=USER_ID, claims=claims)
    excerpts = build_session_excerpts()

    letter = build_inheritance_letter(
        export=export,
        divergence_state=None,
        candidate_excerpts=excerpts,
        insight_generator=fake_insight_generator,
        llm_client=None,
    )

    assert letter.letter_text == "A short letter about claim-005."
    assert letter.citation_chain == ["session-01", "session-02"]


def test_letter_is_never_falsely_marked_as_signed():
    """Same rule as Behavioral DNA export: no real signing infra exists
    yet in Sprint 11's scope. Never fake a signature."""
    claims = build_claims()
    export = build_behavioral_dna_export(user_id=USER_ID, claims=claims)
    excerpts = build_session_excerpts()

    letter = build_inheritance_letter(
        export=export,
        divergence_state=None,
        candidate_excerpts=excerpts,
        insight_generator=fake_insight_generator,
        llm_client=None,
    )

    assert letter.is_signed is False
    assert letter.signature is None


def test_raises_a_clear_typed_error_when_no_eligible_claims_exist():
    """An export with zero Level 3 claims (e.g. a Stage 0-3 cold-start
    user, per Mayank's Sprint 10 gating) has nothing to write a letter
    about. This must fail loudly and specifically -- never silently
    return an empty/fake letter."""
    empty_export = build_behavioral_dna_export(user_id=USER_ID, claims=[])
    excerpts = build_session_excerpts()

    try:
        build_inheritance_letter(
            export=empty_export,
            divergence_state=None,
            candidate_excerpts=excerpts,
            insight_generator=fake_insight_generator,
            llm_client=None,
        )
        assert False, "expected NoEligibleClaimError to be raised, nothing was raised"
    except NoEligibleClaimError:
        pass  # correct behavior


def test_propagates_the_generators_own_error_instead_of_swallowing_it():
    """Mansi's REAL generate_insight() internally requires at least one
    is_near_miss=True excerpt and raises ValueError if none exists (see
    her select_excerpts()). inheritance_protocol.py deliberately does NOT
    duplicate that check itself -- it forwards excerpts straight through
    and trusts the real generator to enforce it. This test simulates that
    real behavior and proves the error is NOT silently caught/hidden here."""
    def generator_that_enforces_near_miss(claim, divergence_state, candidate_excerpts, llm_client):
        has_near_miss = False
        for excerpt in candidate_excerpts:
            if excerpt.is_near_miss:
                has_near_miss = True
        if not has_near_miss:
            raise ValueError("No near-miss counter-example session available.")
        return FakeGeneratedInsight(text="ok", citation_chain=["session-01"])

    claims = build_claims()
    export = build_behavioral_dna_export(user_id=USER_ID, claims=claims)
    excerpts_with_no_near_miss = [e for e in build_session_excerpts() if not e.is_near_miss]

    try:
        build_inheritance_letter(
            export=export,
            divergence_state=None,
            candidate_excerpts=excerpts_with_no_near_miss,
            insight_generator=generator_that_enforces_near_miss,
            llm_client=None,
        )
        assert False, "expected the generator's ValueError to propagate, nothing was raised"
    except ValueError as e:
        assert "near-miss" in str(e)
def test_inheritance_can_be_signed_and_verified_with_device_key():
    from signing import DeviceSigner
    signer = DeviceSigner.generate()
    export = build_behavioral_dna_export(USER_ID, build_claims())
    letter = build_inheritance_letter(export, None, build_session_excerpts(), fake_insight_generator, None, signer)
    assert letter.is_signed and letter.signature
    assert letter.verify_signature(signer)

def test_inheritance_rejects_cross_user_evidence():
    import pytest
    from signing import DeviceSigner
    from upstream_interfaces import SessionExcerpt
    export = build_behavioral_dna_export(USER_ID, build_claims())
    excerpts = build_session_excerpts() + [SessionExcerpt("bad", "other", build_session_excerpts()[0].timestamp, "other", 0.1)]
    with pytest.raises(ValueError):
        build_inheritance_letter(export, None, excerpts, fake_insight_generator, None, DeviceSigner.generate())
