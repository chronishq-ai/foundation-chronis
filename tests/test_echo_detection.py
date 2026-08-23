"""
tests/test_echo_detection.py

Written BEFORE echo_detection.py's logic. We already know the right answer
because tests/fixtures/synthetic_user_profile.py plants a known echo at
day_index 2 and day_index 10.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fixtures.synthetic_user_profile import build_behavioral_state_records
from echo_detection import cosine_similarity, find_echoes, ECHO_SIMILARITY_THRESHOLD


def test_cosine_similarity_identical_vectors_is_one():
    vector = [1.0, 2.0, 3.0]
    result = cosine_similarity(vector, vector)
    assert abs(result - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors_is_zero():
    vector_a = [1.0, 0.0]
    vector_b = [0.0, 1.0]
    result = cosine_similarity(vector_a, vector_b)
    assert abs(result - 0.0) < 1e-9


def test_planted_echo_is_found():
    """day_index 2 and day_index 10 were built to be near-identical AND
    same regime_label -- this MUST be detected as an echo."""
    records = build_behavioral_state_records()

    echoes = find_echoes(records)

    assert len(echoes) >= 1, "expected at least one echo, found none"

    found_the_planted_pair = False
    for echo in echoes:
        indices = {echo.record_index_a, echo.record_index_b}
        if indices == {2, 10}:
            found_the_planted_pair = True
            assert echo.similarity_score > ECHO_SIMILARITY_THRESHOLD

    assert found_the_planted_pair, "the specific planted echo (index 2 & 10) was not found"


def test_far_apart_records_are_not_echoes():
    """day_index 0 and day_index 7 were built to be far apart AND in
    different regimes -- this must NOT be flagged as an echo."""
    records = build_behavioral_state_records()

    echoes = find_echoes(records)

    for echo in echoes:
        indices = {echo.record_index_a, echo.record_index_b}
        assert indices != {0, 7}, "false positive: unrelated records flagged as an echo"


def test_no_echo_matches_across_different_regimes():
    """Even if two m_t vectors happened to be similar, a different
    regime_label should block the match -- context must agree, not just math."""
    records = build_behavioral_state_records()

    echoes = find_echoes(records)

    for echo in echoes:
        record_a = records[echo.record_index_a]
        record_b = records[echo.record_index_b]
        assert record_a.p_t.regime_label == record_b.p_t.regime_label


def test_regime_gate_blocks_match_even_with_near_identical_m_t():
    """Isolates the regime-gate specifically: two records with HIGH
    similarity m_t but DIFFERENT regime_label must not be flagged.
    (The earlier test used records that differed in both m_t and regime --
    this test proves the regime check works on its own, not by accident.)"""
    from upstream_interfaces import BehavioralStateRecord, RegimeState
    from datetime import datetime

    similar_m_t = [0.1, 0.2, 0.3, 0.4, 0.5]

    record_a = BehavioralStateRecord(
        user_id="user_001",
        timestamp=datetime(2026, 1, 1),
        m_t=similar_m_t,
        p_t=RegimeState(regime_label=0, regime_posterior=[0.9, 0.1]),
    )
    record_b = BehavioralStateRecord(
        user_id="user_001",
        timestamp=datetime(2026, 1, 2),
        m_t=similar_m_t,  # identical m_t on purpose
        p_t=RegimeState(regime_label=1, regime_posterior=[0.1, 0.9]),  # different regime
    )

    echoes = find_echoes([record_a, record_b])

    assert echoes == [], "regime gate failed to block a match across different regimes"


def test_empty_input_returns_empty_list():
    """Zero records in -> zero echoes out, not a crash."""
    echoes = find_echoes([])
    assert echoes == []


def test_single_record_returns_empty_list():
    """One record can't echo against itself."""
    records = build_behavioral_state_records()
    echoes = find_echoes([records[0]])
    assert echoes == []