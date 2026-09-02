import pytest
import numpy as np
from backbone.attractors.detector import compute_attractor_stats


def _get_fixture():
    # Sequence: 3 of regime 0, 2 of regime 1, 2 of regime 0 -> A-A-A-B-B-A-A
    regime_labels = np.array([0, 0, 0, 1, 1, 0, 0])
    m_t = np.array([
        [0.10, 0.10],
        [0.10, 0.10],
        [0.10, 0.10],
        [10.0, 10.0],
        [10.1, 10.0],
        [0.10, 0.10],
        [0.10, 0.10],
    ])
    return m_t, regime_labels


def test_s34_2_t1_attractor_run_ordering_detects_separate_runs():
    # T1: Synthetic sequence A-A-A-B-B-A-A. Detect regime-A runs. MUST report TWO separate runs.
    m_t, regime_labels = _get_fixture()
    stats = compute_attractor_stats(m_t, regime_labels, target_regime=0)
    
    # We should have two entries in exit/entry or dwell times (conceptually two separate runs)
    # The mean dwell time is 2.5 which is computed from runs of lengths 3 and 2.
    # Revisit count = 2 proves we detected two separate runs.
    assert stats["revisit_count"] == 2, f"Expected 2 separate runs, got {stats['revisit_count']}"


def test_s34_2_t2_attractor_run_ordering_independent_dwell_stats():
    # T2: Same fixture, compute dwell time -> two independent dwell statistics, not summed into one.
    m_t, regime_labels = _get_fixture()
    stats = compute_attractor_stats(m_t, regime_labels, target_regime=0)
    
    # If summed, mean dwell time would be 5.0. If independent, mean dwell time is (3 + 2) / 2 = 2.5.
    assert stats["mean_dwell_time"] == 2.5, f"Expected independent mean dwell time of 2.5, got {stats['mean_dwell_time']}"


def test_s34_2_t3_attractor_run_ordering_revisit_count():
    # T3: Same fixture, compute revisit count for regime A -> must equal 2, not 1.
    m_t, regime_labels = _get_fixture()
    stats = compute_attractor_stats(m_t, regime_labels, target_regime=0)
    assert stats["revisit_count"] == 2, f"Expected revisit count of 2, got {stats['revisit_count']}"
