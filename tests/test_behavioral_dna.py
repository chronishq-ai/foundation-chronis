"""
tests/test_behavioral_dna.py

Written BEFORE behavioral_dna.py's logic. We know the right answer because
synthetic_user_profile.build_claims() plants 4 claims: one admissible
Level 3 (should be included), one inadmissible Level 3 (must be excluded),
one Level 2 (must be excluded), and one belonging to a different user
entirely (must be excluded).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fixtures.synthetic_user_profile import build_claims, USER_ID
from behavioral_dna import build_behavioral_dna_export


def test_only_admissible_level3_claims_for_this_user_are_included():
    claims = build_claims()

    export = build_behavioral_dna_export(user_id=USER_ID, claims=claims)

    included_ids = sorted(c.claim_id for c in export.level3_claims)
    assert included_ids == ["claim-001", "claim-005"], \
        f"expected both admissible Level 3 claims for this user, got {included_ids}"


def test_inadmissible_level3_claim_never_leaks_in():
    claims = build_claims()

    export = build_behavioral_dna_export(user_id=USER_ID, claims=claims)

    included_ids = [c.claim_id for c in export.level3_claims]
    assert "claim-002" not in included_ids, "inadmissible Level 3 claim leaked into export"


def test_level2_claim_never_leaks_in():
    claims = build_claims()

    export = build_behavioral_dna_export(user_id=USER_ID, claims=claims)

    included_ids = [c.claim_id for c in export.level3_claims]
    assert "claim-003" not in included_ids, "Level 2 claim leaked into a Level-3-only export"


def test_different_users_claim_never_leaks_in():
    """Even though claim-004 is an admissible Level 3 claim, it belongs
    to user_999, not user_001 -- must never appear in user_001's export."""
    claims = build_claims()

    export = build_behavioral_dna_export(user_id=USER_ID, claims=claims)

    included_ids = [c.claim_id for c in export.level3_claims]
    assert "claim-004" not in included_ids, "another user's claim leaked into this export"


def test_empty_claims_list_produces_empty_export_not_a_crash():
    export = build_behavioral_dna_export(user_id=USER_ID, claims=[])

    assert export.level3_claims == []


def test_export_is_never_falsely_marked_as_signed():
    """Hard rule: no real signing infrastructure exists yet in Sprint 11's
    scope. is_signed must be False and signature must be None -- never a
    fake placeholder string pretending to be a real signature."""
    claims = build_claims()

    export = build_behavioral_dna_export(user_id=USER_ID, claims=claims)

    assert export.is_signed is False
    assert export.signature is None


def test_lexicon_profile_and_social_graph_default_to_none():
    claims = build_claims()

    export = build_behavioral_dna_export(user_id=USER_ID, claims=claims)

    assert export.lexicon_profile is None
    assert export.social_graph_summary is None


def test_lexicon_profile_and_social_graph_are_stored_when_provided():
    claims = build_claims()
    fake_lexicon = {"recurring_words": ["overwhelmed", "trying"]}
    fake_social_graph = {"cluster_count": 4}

    export = build_behavioral_dna_export(
        user_id=USER_ID,
        claims=claims,
        lexicon_profile=fake_lexicon,
        social_graph_summary=fake_social_graph,
    )

    assert export.lexicon_profile == fake_lexicon
    assert export.social_graph_summary == fake_social_graph


def test_export_records_the_correct_user_id():
    claims = build_claims()

    export = build_behavioral_dna_export(user_id=USER_ID, claims=claims)

    assert export.user_id == USER_ID
def test_behavioral_dna_can_be_cryptographically_signed_and_verified():
    from signing import DeviceSigner
    signer = DeviceSigner.generate()
    export = build_behavioral_dna_export(USER_ID, build_claims(), {"recurring_words": ["trying"]}, {"cluster_count": 1}, signer)
    assert export.is_signed is True
    assert export.signature
    assert export.verify_signature(signer)

def test_behavioral_dna_signature_detects_tampering():
    from signing import DeviceSigner
    signer = DeviceSigner.generate()
    export = build_behavioral_dna_export(USER_ID, build_claims(), {"recurring_words": ["trying"]}, {"cluster_count": 1}, signer)
    tampered = type(export)(export.user_id, export.level3_claims, {"recurring_words": ["changed"]}, export.social_graph_summary, export.signature, export.is_signed, export.generated_at)
    assert not tampered.verify_signature(signer)

def test_social_graph_summary_rejects_identity_or_raw_fingerprint_fields():
    import pytest
    with pytest.raises(ValueError):
        build_behavioral_dna_export(USER_ID, build_claims(), social_graph_summary={"name": "Alice"})
