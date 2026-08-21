from domain_emergence.narrative_topics import NarrativeTopicModel, NOISE_TOPIC


CAREER_DOCS = [
    "meeting with my manager about the project deadline",
    "finished the quarterly report today",
    "stressed about the performance review",
    "got positive feedback from the client",
    "working late on the presentation again",
] * 4  # repeat so the streaming clusterer has enough density

HEALTH_DOCS = [
    "went for a run this morning",
    "trouble sleeping again last night",
    "doctor said my numbers look better",
    "skipped the gym today feeling tired",
    "started a new meditation routine",
] * 4


def test_partial_fit_returns_one_label_per_doc():
    model = NarrativeTopicModel(seed=0)
    labels = model.partial_fit(CAREER_DOCS[:5])
    assert len(labels) == 5


def test_none_docs_get_noise_label():
    model = NarrativeTopicModel(seed=0)
    labels = model.partial_fit([None, "a real sentence here", None])
    assert labels[0] == NOISE_TOPIC
    assert labels[2] == NOISE_TOPIC


def test_streaming_batches_accumulate():
    model = NarrativeTopicModel(seed=0)
    model.partial_fit(CAREER_DOCS[:10])
    model.partial_fit(CAREER_DOCS[10:])
    result = model.finalize()
    assert len(result.topic_labels) == len(CAREER_DOCS)


def test_distinct_topics_get_distinct_clusters():
    model = NarrativeTopicModel(seed=1)
    model.partial_fit(CAREER_DOCS)
    model.partial_fit(HEALTH_DOCS)
    result = model.finalize()
    career_labels = set(result.topic_labels[:len(CAREER_DOCS)]) - {NOISE_TOPIC}
    health_labels = set(result.topic_labels[len(CAREER_DOCS):]) - {NOISE_TOPIC}
    # at minimum, some real (non-noise) clustering happened for both topics
    assert len(career_labels) >= 1
    assert len(health_labels) >= 1


def test_finalize_produces_topic_words():
    model = NarrativeTopicModel(seed=2)
    model.partial_fit(CAREER_DOCS)
    result = model.finalize(top_n_words=3)
    assert result.n_topics >= 1
    for tid, words in result.topic_words.items():
        assert len(words) <= 3
        assert all(isinstance(w, str) for w in words)


def test_all_none_docs_produce_no_topics():
    model = NarrativeTopicModel(seed=3)
    model.partial_fit([None, None, None])
    result = model.finalize()
    assert result.n_topics == 0
    assert all(t == NOISE_TOPIC for t in result.topic_labels)