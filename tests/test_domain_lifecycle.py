import pytest
from domain_emergence.domain_lifecycle import (
    should_split, should_merge, DomainRegistry,
)


def test_should_split_sustained_true():
    within = [5.0, 6.0, 4.0]
    between = [2.0, 2.0, 5.0]  # 2 of 3 windows have within > between
    assert should_split(within, between, min_sustained_windows=2) is True


def test_should_split_single_spike_not_sustained():
    within = [5.0, 1.0, 1.0]
    between = [2.0, 5.0, 5.0]  # only 1 window qualifies
    assert should_split(within, between, min_sustained_windows=2) is False


def test_should_split_scattered_non_contiguous_not_sustained():
    """S56.8 regression: 2 qualifying windows total but NOT contiguous
    (window 0 and window 3) must NOT trigger 'sustained' split -- this
    is the exact case the pre-fix total-count logic got wrong."""
    within = [5.0, 1.0, 1.0, 5.0]
    between = [2.0, 5.0, 5.0, 2.0]  # windows 0 and 3 qualify, not adjacent
    assert should_split(within, between, min_sustained_windows=2) is False


def test_should_split_genuine_contiguous_run_triggers():
    within = [1.0, 5.0, 6.0, 1.0]
    between = [5.0, 2.0, 2.0, 5.0]  # windows 1,2 qualify, contiguous
    assert should_split(within, between, min_sustained_windows=2) is True


def test_should_split_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        should_split([1.0, 2.0], [1.0])


def test_should_merge_both_conditions_sustained():
    trans = [0.5, 0.6, 0.1]
    comention = [0.4, 0.5, 0.9]
    # windows 0,1 both exceed thresholds (0.3/0.3); window 2 fails transition
    assert should_merge(trans, comention, transition_threshold=0.3,
                         comention_threshold=0.3, min_sustained_windows=2) is True


def test_should_merge_only_one_condition_never_both():
    trans = [0.9, 0.9, 0.9]      # transition always high
    comention = [0.1, 0.1, 0.1]  # co-mention always low
    assert should_merge(trans, comention, transition_threshold=0.3,
                         comention_threshold=0.3, min_sustained_windows=2) is False


def test_should_merge_scattered_non_contiguous_not_sustained():
    """R2-S56.7 regression: windows 1,3,5,7 (of 8) all qualify -- 4 total
    qualifying windows, well over min_sustained_windows=2 -- but none of
    them are adjacent, so this must NOT count as 'sustained'. This is
    exactly the total-count-vs-contiguous-run bug should_split already
    had fixed; should_merge must use the same rule."""
    trans =     [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1]
    comention = [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1]
    assert should_merge(trans, comention, transition_threshold=0.3,
                         comention_threshold=0.3, min_sustained_windows=2) is False


def test_should_merge_contiguous_windows_triggers_once_duration_met():
    """Windows 1,2,3,4 (of the 6) form one unbroken qualifying run of
    length 4 -- should merge once min_sustained_windows<=4."""
    trans =     [0.1, 0.9, 0.9, 0.9, 0.9, 0.1]
    comention = [0.1, 0.9, 0.9, 0.9, 0.9, 0.1]
    assert should_merge(trans, comention, transition_threshold=0.3,
                         comention_threshold=0.3, min_sustained_windows=4) is True
    assert should_merge(trans, comention, transition_threshold=0.3,
                         comention_threshold=0.3, min_sustained_windows=5) is False


def test_should_merge_irregular_timestamps_uses_real_duration_not_count():
    """R2-S56.7: with irregularly-spaced windows, a 2-window contiguous
    qualifying run can span very different real time depending on the
    gaps between those specific windows -- decision must follow the
    actual elapsed time, not just 'it was 2 windows'."""
    trans = [0.9, 0.9, 0.1]
    comention = [0.9, 0.9, 0.1]
    # run covers windows 0-1: tightly spaced (1 day apart) -> short span
    tight_timestamps = [0.0, 1.0, 30.0]
    assert should_merge(
        trans, comention, transition_threshold=0.3, comention_threshold=0.3,
        timestamps=tight_timestamps, min_sustained_duration=10.0,
    ) is False
    # same qualifying pattern, but the run's two windows are 30 days
    # apart -> spans enough real time to count as sustained
    wide_timestamps = [0.0, 30.0, 60.0]
    assert should_merge(
        trans, comention, transition_threshold=0.3, comention_threshold=0.3,
        timestamps=wide_timestamps, min_sustained_duration=10.0,
    ) is True


def test_should_merge_timestamps_without_duration_raises():
    with pytest.raises(ValueError):
        should_merge([0.9], [0.9], timestamps=[0.0])


def test_should_merge_duration_without_timestamps_raises():
    with pytest.raises(ValueError):
        should_merge([0.9], [0.9], min_sustained_duration=5.0)


def test_should_split_irregular_timestamps_uses_real_duration_not_count():
    """Same temporal rule applied to should_split for symmetry (R2-S56.7:
    'apply one explicit temporal rule to both')."""
    within = [5.0, 6.0, 1.0]
    between = [2.0, 2.0, 5.0]
    tight_timestamps = [0.0, 1.0, 30.0]
    assert should_split(
        within, between, timestamps=tight_timestamps, min_sustained_duration=10.0,
    ) is False
    wide_timestamps = [0.0, 30.0, 60.0]
    assert should_split(
        within, between, timestamps=wide_timestamps, min_sustained_duration=10.0,
    ) is True


def test_registry_register_and_get():
    reg = DomainRegistry()
    did = reg.register_domain(history=[{"window": 1}])
    d = reg.get(did)
    assert d.active is True
    assert d.parent_ids == []
    assert d.history == [{"window": 1}]


def test_registry_split_marks_parent_inactive_preserves_history():
    reg = DomainRegistry()
    parent_id = reg.register_domain(history=[{"w": 1}, {"w": 2}])
    child_ids = reg.split_domain(parent_id, n_children=2)

    parent = reg.get(parent_id)
    assert parent.active is False
    assert parent.pre_split_hold is True
    assert parent.child_ids == child_ids

    for cid in child_ids:
        child = reg.get(cid)
        assert child.active is True
        assert child.parent_ids == [parent_id]
        assert child.history == [{"w": 1}, {"w": 2}]  # inherited


def test_registry_split_never_deletes_parent():
    reg = DomainRegistry()
    parent_id = reg.register_domain()
    reg.split_domain(parent_id)
    # parent still retrievable, just inactive -- registry has no delete method at all
    assert reg.get(parent_id) is not None
    assert not hasattr(reg, "delete_domain")
    assert not hasattr(reg, "remove_domain")


def test_registry_merge_marks_both_inactive_combines_history():
    reg = DomainRegistry()
    a = reg.register_domain(history=[{"w": 1}])
    b = reg.register_domain(history=[{"w": 2}])
    merged_id = reg.merge_domains(a, b)

    assert reg.get(a).active is False
    assert reg.get(b).active is False
    merged = reg.get(merged_id)
    assert merged.active is True
    assert merged.parent_ids == [a, b]
    assert merged.history == [{"w": 1}, {"w": 2}]


def test_active_domains_excludes_split_and_merged_parents():
    reg = DomainRegistry()
    a = reg.register_domain()
    b = reg.register_domain()
    c = reg.register_domain()
    reg.merge_domains(a, b)
    reg.split_domain(c)

    active_ids = {d.domain_id for d in reg.active_domains()}
    assert a not in active_ids
    assert b not in active_ids
    assert c not in active_ids