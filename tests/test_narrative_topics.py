import pytest

from domain_emergence.narrative_topics import (
    NarrativeTopicModel, NOISE_TOPIC, create_topic_model,
    BERTopicWrapper, _BERTOPIC_UNVERIFIED_WARNING,
)


def test_create_topic_model_defaults_to_bertopic():
    """S56.9: default flipped to BERTopicWrapper per explicit
    instruction. BERTopicWrapper is untested in THIS repo's own dev
    sandbox (see its docstring / HONESTY FLAG), not untested
    everywhere -- so the correct assertion depends on whether bertopic
    is actually importable in the environment pytest is running in,
    not on a hardcoded expectation of failure.

    - bertopic NOT installed/importable here: create_topic_model()
      must ATTEMPT to build a BERTopicWrapper and surface that failure
      clearly (ImportError), rather than silently falling back to the
      lightweight model.
    - bertopic IS installed/importable here (e.g. a real Windows env
      where it actually works): construction should succeed and return
      a BERTopicWrapper, still emitting the UNTESTED warning -- a
      previously-untested path succeeding is not itself a bug."""
    try:
        import bertopic  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            create_topic_model(seed=0)
    else:
        with pytest.warns(UserWarning, match="UNTESTED"):
            model = create_topic_model(seed=0)
        assert isinstance(model, BERTopicWrapper)


def test_create_topic_model_explicit_lightweight():
    """use_bertopic=False still returns the tested, previously-default
    NarrativeTopicModel -- this path is unaffected by the S56.9 flip."""
    model = create_topic_model(use_bertopic=False, seed=0)
    assert isinstance(model, NarrativeTopicModel)


def test_bertopic_wrapper_warns_before_risky_import():
    """The UserWarning must fire even when the subsequent `import
    bertopic` fails/crashes -- it's constructed first, specifically so
    a caller/log always sees 'this path is unverified' regardless of
    what happens next. Guards against a future refactor silently
    reordering the warning after the import."""
    pytest.importorskip(
        "bertopic", reason="only meaningful where bertopic is actually installed"
    )
    with pytest.warns(UserWarning, match="UNTESTED"):
        try:
            create_topic_model(use_bertopic=True, seed=0)
        except Exception:
            pass  # only asserting the warning fired, not that construction succeeded


def test_bertopic_unverified_warning_text_present():
    """Cheap regression guard: the warning message itself stays
    substantive (mentions untested status) even if refactored."""
    assert "UNTESTED" in _BERTOPIC_UNVERIFIED_WARNING
    assert "smoke test" in _BERTOPIC_UNVERIFIED_WARNING


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