"""
Day 17 -- Synthetic transcript segment generator.

Mocks the Audio Transcription Pipeline (Bible Part 4.3) output: short
conversational text segments per episode, standing in for real transcripts
until Team 2/Layer 1 wiring is available. Matches the interface BERTopic
consumes (list of text documents, one per episode).

Topics are seeded with a configurable regime<->topic correlation so the
downstream Fisher's-exact alignment step (domain_alignment.py) has real
signal to find, plus deliberate noise:
  - some episodes get NO narrative text (silent sessions) -> behavioral
    candidate with no narrative partner -> HIGH IGNORANCE PRIOR
  - some narrative topics are injected independent of any regime ->
    narrative candidate with no behavioral partner -> ASPIRATIONAL OR
    HYPOTHETICAL
"""

from __future__ import annotations
import numpy as np

TOPIC_VOCAB = {
    "career": [
        "meeting with my manager about the project deadline",
        "finished the quarterly report today",
        "stressed about the performance review",
        "got positive feedback from the client",
        "working late on the presentation again",
    ],
    "relationships": [
        "had dinner with my partner tonight",
        "argument with my sibling about the holidays",
        "video call with my parents this weekend",
        "missing my friends from college",
        "we talked about moving in together",
    ],
    "health": [
        "went for a run this morning",
        "trouble sleeping again last night",
        "doctor said my numbers look better",
        "skipped the gym today, feeling tired",
        "started a new meditation routine",
    ],
    "hobby": [
        "practiced guitar for an hour",
        "finally finished the painting",
        "reading a new book before bed",
        "played chess online with strangers",
        "started learning a new recipe",
    ],
}

TOPIC_NAMES = list(TOPIC_VOCAB.keys())


def generate_synthetic_transcripts(
    n_episodes: int,
    episode_regime_ids: np.ndarray,
    regime_topic_map: dict[int, str] | None = None,
    silent_rate: float = 0.15,
    independent_topic_rate: float = 0.1,
    seed: int | None = None,
) -> list:
    """Generate one text segment per episode (or None for a silent
    session). regime_topic_map ties regime_id -> topic name, so behavioral
    and narrative clusters correlate (real signal for Fisher's-exact to
    find). Falls back to a stable round-robin assignment if not given.

    independent_topic_rate: fraction of NON-silent episodes that get a
    topic chosen independent of their regime (narrative activity with no
    behavioral counterpart) -> feeds ASPIRATIONAL OR HYPOTHETICAL flag
    downstream.
    """
    rng = np.random.default_rng(seed)

    if regime_topic_map is None:
        unique_regimes = sorted(set(episode_regime_ids.tolist()))
        regime_topic_map = {
            r: TOPIC_NAMES[i % len(TOPIC_NAMES)] for i, r in enumerate(unique_regimes)
        }

    texts = []
    for i in range(n_episodes):
        if rng.random() < silent_rate:
            texts.append(None)
            continue

        regime_id = int(episode_regime_ids[i])
        if rng.random() < independent_topic_rate:
            topic = rng.choice(TOPIC_NAMES)
        else:
            topic = regime_topic_map.get(regime_id, rng.choice(TOPIC_NAMES))

        sentence = rng.choice(TOPIC_VOCAB[topic])
        texts.append(sentence)

    return texts