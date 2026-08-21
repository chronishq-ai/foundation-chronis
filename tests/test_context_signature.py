import numpy as np
from domain_emergence.context_signature import extract_episodes, build_context_signatures, Episode


def test_extract_episodes_simple():
    seq = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    episodes = extract_episodes(seq)
    assert len(episodes) == 3
    assert episodes[0].regime_id == 0 and episodes[0].start == 0 and episodes[0].end == 3
    assert episodes[1].regime_id == 1 and episodes[1].start == 3 and episodes[1].end == 5
    assert episodes[2].regime_id == 2 and episodes[2].start == 5 and episodes[2].end == 9


def test_extract_episodes_single_regime():
    seq = np.array([1, 1, 1, 1])
    episodes = extract_episodes(seq)
    assert len(episodes) == 1
    assert episodes[0].duration == 4


def test_extract_episodes_empty():
    assert extract_episodes(np.array([])) == []


def test_extract_episodes_alternating():
    seq = np.array([0, 1, 0, 1])
    episodes = extract_episodes(seq)
    assert len(episodes) == 4
    assert all(ep.duration == 1 for ep in episodes)


def test_signatures_shape():
    seq = np.array([0, 0, 0, 1, 1, 1])
    obs = np.random.default_rng(0).normal(size=(6, 3))
    episodes = extract_episodes(seq)
    sigs, kept = build_context_signatures(obs, episodes)
    assert sigs.shape == (2, 2 * 3 + 1)  # mean(3) + std(3) + log_duration(1)
    assert len(kept) == 2


def test_signatures_exclude_nan_rows():
    seq = np.array([0, 0, 0, 0])
    obs = np.array([[1.0, 1.0], [np.nan, np.nan], [3.0, 3.0], [5.0, 5.0]])
    episodes = extract_episodes(seq)
    sigs, kept = build_context_signatures(obs, episodes)
    # mean of present rows only: (1+3+5)/3 = 3.0
    assert np.isclose(sigs[0, 0], 3.0)
    assert np.isclose(sigs[0, 1], 3.0)


def test_signatures_drop_episode_below_min_present():
    seq = np.array([0, 0, 1, 1])
    obs = np.array([[np.nan, np.nan], [np.nan, np.nan], [1.0, 1.0], [2.0, 2.0]])
    episodes = extract_episodes(seq)
    sigs, kept = build_context_signatures(obs, episodes, min_present_sessions=1)
    # first episode has 0 present rows, should be dropped
    assert len(kept) == 1
    assert kept[0].regime_id == 1


def test_signatures_all_dropped_returns_empty():
    seq = np.array([0, 0])
    obs = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    episodes = extract_episodes(seq)
    sigs, kept = build_context_signatures(obs, episodes)
    assert sigs.shape == (0, 5)  # F=2 -> mean(2)+std(2)+log_dur(1)
    assert kept == []


def test_single_present_row_std_is_zero():
    seq = np.array([0])
    obs = np.array([[2.0, 4.0]])
    episodes = extract_episodes(seq)
    sigs, kept = build_context_signatures(obs, episodes)
    # std over 1 sample -> defined as 0, not NaN
    assert np.allclose(sigs[0, 2:4], 0.0)