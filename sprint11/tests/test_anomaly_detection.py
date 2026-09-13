"""
tests/test_anomaly_detection.py

Written BEFORE anomaly_detection.py's logic. We know the right answers
because synthetic_user_profile.build_anomaly_test_records() plants exactly
one acute, one sustained, and one structural anomaly at known day-indexes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.fixtures.synthetic_user_profile import build_anomaly_test_records
from anomaly_detection import (
    detect_acute_anomalies,
    detect_sustained_anomalies,
    detect_structural_anomalies,
    AnomalyScale,
)
from clinical_terms import contains_clinical_terminology


def test_acute_anomaly_is_found_at_planted_day():
    records = build_anomaly_test_records()

    anomalies = detect_acute_anomalies(records)

    found_day_5 = False
    for anomaly in anomalies:
        if anomaly.record_index == 5:
            found_day_5 = True
    assert found_day_5, "planted acute anomaly at day 5 was not detected"


def test_acute_anomaly_does_not_fire_on_stable_baseline_days():
    records = build_anomaly_test_records()

    anomalies = detect_acute_anomalies(records)

    # days 0-3 are stable baseline -- should never fire here
    for anomaly in anomalies:
        assert anomaly.record_index not in (0, 1, 2, 3), \
            f"false positive: acute anomaly fired on stable day {anomaly.record_index}"


def test_sustained_anomaly_is_found_across_planted_window():
    records = build_anomaly_test_records()

    anomalies = detect_sustained_anomalies(records, min_consecutive_days=3)

    assert len(anomalies) >= 1, "expected at least one sustained anomaly, found none"
    covered_days = set()
    for anomaly in anomalies:
        for day in range(anomaly.start_index, anomaly.end_index + 1):
            covered_days.add(day)

    # days 10-13 were planted as the sustained shift
    for planted_day in (10, 11, 12, 13):
        assert planted_day in covered_days, \
            f"planted sustained anomaly day {planted_day} was not covered"


def test_sustained_anomaly_requires_the_minimum_run_length():
    """A single acute spike (day 5) must NOT count as 'sustained' just
    because it's also a deviation -- sustained requires a genuine run."""
    records = build_anomaly_test_records()

    anomalies = detect_sustained_anomalies(records, min_consecutive_days=3)

    for anomaly in anomalies:
        run_length = anomaly.end_index - anomaly.start_index + 1
        assert run_length >= 3, "a sustained anomaly shorter than the minimum run leaked through"
        # the single-day acute spike at index 5 should never appear alone as a "sustained" run
        assert not (anomaly.start_index == 5 and anomaly.end_index == 5)


def test_sustained_anomaly_does_not_flag_the_entire_history():
    """Regression test for a real bug found during review: a mean-based
    baseline gets DRAGGED toward the anomalies themselves, which can
    corrupt the whole comparison and falsely flag every single day as
    'sustained.' Stable baseline days (0-3, 6-9, 14-15) must NEVER be
    included in any sustained-anomaly run."""
    records = build_anomaly_test_records()

    anomalies = detect_sustained_anomalies(records, min_consecutive_days=3)

    covered_days = set()
    for anomaly in anomalies:
        for day in range(anomaly.start_index, anomaly.end_index + 1):
            covered_days.add(day)

    known_stable_days = {0, 1, 2, 3, 6, 7, 8, 9, 14, 15}
    overlap = covered_days & known_stable_days
    assert overlap == set(), f"stable baseline days were incorrectly flagged as sustained: {overlap}"


def test_structural_anomaly_is_found_at_regime_shift():
    records = build_anomaly_test_records()

    anomalies = detect_structural_anomalies(records, window_size=4)

    assert len(anomalies) >= 1, "expected at least one structural anomaly, found none"
    found_the_shift = False
    for anomaly in anomalies:
        if anomaly.record_index >= 16:
            found_the_shift = True
    assert found_the_shift, "planted structural (regime-shift) anomaly at day 16+ was not found"


def test_output_copy_never_contains_clinical_language():
    """Hard rule from the directive: automated string check, not manual
    eyeballing. Every one of Mansi's banned terms must be caught."""
    banned_examples = [
        "This looks like a sign of depression.",
        "You might be feeling anxious lately.",
        "This could indicate a disorder.",
        "A possible diagnosis worth noting.",
    ]
    for example_text in banned_examples:
        result = contains_clinical_terminology(example_text)
        assert result is not None, f"clinical filter FAILED to catch: '{example_text}'"


def test_clean_copy_passes_the_filter():
    clean_text = "Your evening routine looked different than usual this week."
    result = contains_clinical_terminology(clean_text)
    assert result is None, f"filter incorrectly flagged clean text, matched on: '{result}'"


def test_anomaly_scale_enum_has_all_three_required_values():
    """Guards against someone accidentally deleting a scale later."""
    assert AnomalyScale.ACUTE
    assert AnomalyScale.SUSTAINED
    assert AnomalyScale.STRUCTURAL
def test_anomaly_copy_validator_rejects_diagnostic_language():
    import pytest
    from anomaly_detection import validate_anomaly_copy
    with pytest.raises(ValueError):
        validate_anomaly_copy("This could indicate depression.")

def test_anomaly_engine_rejects_non_finite_behavioral_values():
    import pytest
    from datetime import datetime
    from upstream_interfaces import BehavioralStateRecord, RegimeState
    from anomaly_detection import detect_acute_anomalies
    records=[BehavioralStateRecord("u",datetime(2026,1,1),[float('nan')],RegimeState(0,[1.0])), BehavioralStateRecord("u",datetime(2026,1,2),[1.0],RegimeState(0,[1.0]))]
    with pytest.raises(ValueError): detect_acute_anomalies(records)

def test_clinical_safety_function_is_single_sprint11_source():
    import anomaly_detection
    import clinical_terms
    assert anomaly_detection.validate_anomaly_copy.__module__ == "anomaly_detection"
    assert clinical_terms.contains_clinical_terminology("anxiety") == "anxiety"
