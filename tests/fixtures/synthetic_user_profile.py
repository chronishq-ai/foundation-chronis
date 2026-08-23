"""
tests/fixtures/synthetic_user_profile.py

ONE shared fake user, reused by every Sprint 11 module test. This is what
Sprint 11's own Definition of Done requires: "every auxiliary module runs
cleanly against the same shared surrogate profile with no new modeling
assumptions introduced."

Design choices, explained:

  - 15 timestamped BehavioralStateRecords, one per day, all for "user_001".
  - Records at index 2 and index 10 are DELIBERATELY given near-identical
    m_t vectors (cosine similarity > 0.8) AND the same regime_label, so
    Echo Detection has a guaranteed, known echo to find. This is the same
    idea as Kuheli's earlier planted-attractor testing style: know the
    right answer before you write the detector.
  - All other records are spread apart so they should NOT trigger an echo
    -- this is what catches a false-positive bug (detector firing on
    everything).
  - One Level 3 Claim is included (admissible) and one Level 3 Claim that
    is NOT admissible (failed a gate) -- this is the case that would leak
    into a Behavioral DNA export if you filter on level alone instead of
    level + admissible.
"""

from __future__ import annotations
from datetime import datetime, timedelta

from upstream_interfaces import (
    BehavioralStateRecord,
    RegimeState,
    Claim,
    ClaimLevel,
    GateEvaluation,
    GateCheck,
    DomainRecord,
)

USER_ID = "user_001"
BASE_DATE = datetime(2026, 8, 1, 12, 0, 0)


def _make_m_t(seed_value: float) -> list:
    """Build a 10-dim m_t vector by nudging a base pattern with seed_value.
    Plain, explicit, no numpy/random needed for a fixture this small."""
    base = [0.12, -0.45, 0.79, 0.01, -0.11, 0.34, 0.56, -0.09, 0.22, -0.31]
    vector = []
    for value in base:
        vector.append(value + seed_value)
    return vector


def build_behavioral_state_records() -> list:
    """15 daily records for user_001. Records 2 and 10 are near-duplicates
    (the planted echo). Record 7 is in a different regime entirely, to make
    sure the detector doesn't accidentally match across regimes."""
    records = []

    for day_index in range(15):
        timestamp = BASE_DATE + timedelta(days=day_index)

        if day_index == 2:
            m_t = _make_m_t(seed_value=0.0)
            regime_label = 1
        elif day_index == 10:
            # near-identical to day_index 2 -- tiny nudge, still > 0.8 cosine sim
            m_t = _make_m_t(seed_value=0.001)
            regime_label = 1
        elif day_index == 7:
            m_t = _make_m_t(seed_value=5.0)   # far away on purpose
            regime_label = 2
        else:
            # spread the rest out so they don't accidentally echo each other
            m_t = _make_m_t(seed_value=float(day_index) * 0.8)
            regime_label = 0

        record = BehavioralStateRecord(
            user_id=USER_ID,
            timestamp=timestamp,
            m_t=m_t,
            p_t=RegimeState(
                regime_label=regime_label,
                regime_posterior=[0.1, 0.8, 0.1] if regime_label == 1 else [0.7, 0.2, 0.1],
            ),
        )
        records.append(record)

    return records


def build_claims() -> list:
    """One admissible Level 3 claim, one inadmissible Level 3 claim, plus
    a Level 2 claim and a different user's claim -- Behavioral DNA export
    must filter out all three of the latter, keeping only the first."""
    passing_gates = GateEvaluation(
        level=ClaimLevel.LEVEL_3,
        admissible=True,
        checks=[GateCheck(name="level2_admissible", passed=True)],
    )
    failing_gates = GateEvaluation(
        level=ClaimLevel.LEVEL_3,
        admissible=False,
        checks=[GateCheck(name="divergence_type_unambiguous", passed=False,
                           detail="two type scores within 0.15 of each other")],
    )
    level2_gates = GateEvaluation(
        level=ClaimLevel.LEVEL_2,
        admissible=True,
        checks=[GateCheck(name="convergent_level1_patterns", passed=True)],
    )

    admissible_claim = Claim(
        claim_id="claim-001",
        user_id=USER_ID,
        domain_id="domain-001",
        level=ClaimLevel.LEVEL_3,
        dominant_divergence_type="aspiration",
        gate_evaluation=passing_gates,
        created_at=BASE_DATE,
    )

    inadmissible_claim = Claim(
        claim_id="claim-002",
        user_id=USER_ID,
        domain_id="domain-001",
        level=ClaimLevel.LEVEL_3,
        dominant_divergence_type=None,
        gate_evaluation=failing_gates,
        created_at=BASE_DATE,
    )

    level2_claim = Claim(
        claim_id="claim-003",
        user_id=USER_ID,
        domain_id="domain-001",
        level=ClaimLevel.LEVEL_2,
        dominant_divergence_type="self_protection",
        gate_evaluation=level2_gates,
        created_at=BASE_DATE,
    )

    different_users_claim = Claim(
        claim_id="claim-004",
        user_id="user_999",   # NOT user_001 -- must never leak into user_001's export
        domain_id="domain-005",
        level=ClaimLevel.LEVEL_3,
        dominant_divergence_type="aspiration",
        gate_evaluation=passing_gates,
        created_at=BASE_DATE,
    )

    return [admissible_claim, inadmissible_claim, level2_claim, different_users_claim]


def build_domains() -> list:
    return [
        DomainRecord(domain_id=1, active=True, status="active", confidence=0.82),
        DomainRecord(domain_id=2, active=True, status="candidate", confidence=0.31),
    ]


def build_anomaly_test_records() -> list:
    """A separate 20-day record set, purpose-built for Anomaly Detection
    testing -- kept separate from build_behavioral_state_records() because
    that one plants an ECHO pattern, and mixing the two plants would make
    it unclear which detector is being tested against which signal.

    Layout (all m_t vectors are 5-dim here, for simplicity):
      - Days 0-4:   stable baseline, all near [0.1, 0.1, 0.1, 0.1, 0.1]
      - Day 5:      ACUTE anomaly -- one single wild spike, days 4 and 6
                    are both back to normal (proves it's a single-moment
                    event, not the start of a trend)
      - Days 6-9:   back to baseline
      - Days 10-13: SUSTAINED anomaly -- four consecutive days all
                    shifted away from baseline together
      - Days 14-15: back to baseline (proves the sustained anomaly ended)
      - Days 16-19: STRUCTURAL anomaly -- regime_label permanently
                    shifts from 0 to 2 and stays there (the underlying
                    pattern changed, not just the raw m_t numbers)
    """
    from datetime import datetime, timedelta

    records = []
    base_date = datetime(2026, 9, 1, 12, 0, 0)
    baseline_vector = [0.1, 0.1, 0.1, 0.1, 0.1]

    for day_index in range(20):
        timestamp = base_date + timedelta(days=day_index)

        if day_index == 5:
            m_t = [5.0, 5.0, 5.0, 5.0, 5.0]   # acute spike
            regime_label = 0
        elif 10 <= day_index <= 13:
            m_t = [2.5, 2.5, 2.5, 2.5, 2.5]   # sustained shift
            regime_label = 0
        elif day_index >= 16:
            m_t = [0.1, 0.1, 0.1, 0.1, 0.1]   # numbers look normal...
            regime_label = 2                    # ...but the regime itself changed
        else:
            m_t = list(baseline_vector)
            regime_label = 0

        records.append(BehavioralStateRecord(
            user_id=USER_ID,
            timestamp=timestamp,
            m_t=m_t,
            p_t=RegimeState(
                regime_label=regime_label,
                regime_posterior=[0.9, 0.05, 0.05] if regime_label == 0 else [0.05, 0.05, 0.9],
            ),
        ))

    return records