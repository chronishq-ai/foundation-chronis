import numpy as np
from domain_emergence.synthetic_transcripts import (
    generate_synthetic_transcripts,
    TOPIC_VOCAB,
    TOPIC_NAMES,
)


def test_output_length_matches_n_episodes():
    regimes = np.array([0, 0, 1, 1, 2])
    texts = generate_synthetic_transcripts(5, regimes, seed=0)
    assert len(texts) == 5


def test_silent_rate_produces_nones():
    regimes = np.zeros(500, dtype=int)
    texts = generate_synthetic_transcripts(500, regimes, silent_rate=0.3, seed=1)
    n_silent = sum(1 for t in texts if t is None)
    # loose bound, stochastic around 30%
    assert 100 <= n_silent <= 220


def test_zero_silent_rate_no_nones():
    regimes = np.zeros(100, dtype=int)
    texts = generate_synthetic_transcripts(100, regimes, silent_rate=0.0, seed=2)
    assert all(t is not None for t in texts)


def test_regime_topic_correlation_holds():
    """With independent_topic_rate=0 and silent_rate=0, every non-silent
    text for a given regime must come from that regime's mapped topic
    vocab -- this is the correlation the alignment step depends on."""
    regimes = np.array([0] * 50 + [1] * 50)
    regime_topic_map = {0: "career", 1: "health"}
    texts = generate_synthetic_transcripts(
        100, regimes, regime_topic_map=regime_topic_map,
        silent_rate=0.0, independent_topic_rate=0.0, seed=3,
    )
    for i, t in enumerate(texts):
        expected_topic = "career" if regimes[i] == 0 else "health"
        assert t in TOPIC_VOCAB[expected_topic]


def test_independent_topic_rate_breaks_some_correlation():
    regimes = np.zeros(300, dtype=int)
    regime_topic_map = {0: "career"}
    texts = generate_synthetic_transcripts(
        300, regimes, regime_topic_map=regime_topic_map,
        silent_rate=0.0, independent_topic_rate=0.3, seed=4,
    )
    non_career = sum(1 for t in texts if t not in TOPIC_VOCAB["career"])
    # roughly 30% should escape the career vocab via independent assignment
    assert 50 <= non_career <= 130


def test_default_regime_topic_map_is_deterministic_roundrobin():
    regimes = np.array([0, 1, 2, 3])
    map1 = None
    texts1 = generate_synthetic_transcripts(
        4, regimes, regime_topic_map=map1, silent_rate=0.0,
        independent_topic_rate=0.0, seed=5,
    )
    # each regime's text must belong to TOPIC_NAMES[regime % len]
    for i, t in enumerate(texts1):
        expected_topic = TOPIC_NAMES[i % len(TOPIC_NAMES)]
        assert t in TOPIC_VOCAB[expected_topic]