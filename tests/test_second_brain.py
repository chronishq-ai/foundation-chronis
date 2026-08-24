"""
tests/test_second_brain.py

Written BEFORE second_brain.py's logic.

What we're testing here is different in spirit from every other module
so far: we are NOT testing "does this correctly filter/gate something."
We are testing the OPPOSITE -- that this module does NOT filter or gate
anything, because the directive is explicit that gating happens only at
the constitutional-policy layer, never inside this model. A bug in this
module would look like someone "helpfully" adding a safety check that
doesn't belong here.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fixtures.synthetic_user_profile import build_claims, USER_ID
from second_brain import build_decision_replication_snapshot


def test_snapshot_includes_every_claim_regardless_of_level():
    """Deliberately the opposite check from Behavioral DNA's tests -- here,
    Level 0/1/2 claims AND inadmissible claims must all still appear.
    Nothing gets filtered at this layer."""
    claims = build_claims()  # includes Level 2, admissible L3, inadmissible L3, and another user's claim

    snapshot = build_decision_replication_snapshot(user_id=USER_ID, claims=claims)

    included_ids = [c.claim_id for c in snapshot.all_claims]
    assert "claim-003" in included_ids, "Level 2 claim was filtered out -- gating does not belong here"
    assert "claim-002" in included_ids, "inadmissible claim was filtered out -- gating does not belong here"


def test_snapshot_still_scopes_to_the_requested_user_only():
    """The ONE thing this module should still do: not silently mix in a
    different user's data by accident. This isn't a 'gate' in the Bible
    sense (it's not a safety/quality judgment) -- it's basic data
    correctness, the same as every other module in this sprint."""
    claims = build_claims()

    snapshot = build_decision_replication_snapshot(user_id=USER_ID, claims=claims)

    for claim in snapshot.all_claims:
        assert claim.user_id == USER_ID


def test_empty_claims_produces_empty_snapshot_not_a_crash():
    snapshot = build_decision_replication_snapshot(user_id=USER_ID, claims=[])
    assert snapshot.all_claims == []


def test_snapshot_carries_no_gating_decision_fields():
    """Guards against future scope-creep: this dataclass should never grow
    fields like 'is_approved_for_display' or 'blocked_reason' -- that
    logic belongs to the constitutional-policy layer, not here."""
    from second_brain import DecisionReplicationSnapshot
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(DecisionReplicationSnapshot)}
    forbidden_field_names = {"is_approved", "blocked_reason", "gate_passed", "filtered"}

    overlap = field_names & forbidden_field_names
    assert overlap == set(), f"gating-style fields found where none should exist: {overlap}"