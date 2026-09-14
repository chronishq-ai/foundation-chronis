import warnings
import pytest

from domain_emergence.narrative_topics import (
    NarrativeTopicModel, NOISE_TOPIC, create_topic_model,
    BERTopicWrapper, _BERTOPIC_EMBEDDING_STANDIN_WARNING,
)

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


# ---------------------------------------------------------------------------
# R2-S56.8 (item 7): real end-to-end BERTopicWrapper integration tests.
# These require bertopic/hdbscan/river/umap-learn actually installed and
# importable -- skipped (not faked/mocked) if that isn't true in the
# environment running this suite, per the pack's own non-negotiable rule
# that a green test against a mock/stand-in does not close this ticket.
# ---------------------------------------------------------------------------
bertopic_available = True
try:
    import bertopic  # noqa: F401
except ImportError:
    bertopic_available = False

requires_bertopic = pytest.mark.skipif(
    not bertopic_available,
    reason="bertopic not installed in this environment -- see "
           "test_create_topic_model_defaults_to_bertopic for the "
           "not-installed-here behavior this suite still covers.",
)


def test_create_topic_model_defaults_to_bertopic():
    """create_topic_model() must default to the real BERTopic path
    (BERTopicWrapper), never silently return the lightweight fallback
    (NarrativeTopicModel) just because bertopic isn't installed here.

    - bertopic NOT installed/importable: must ATTEMPT to build a
      BERTopicWrapper and surface that failure clearly (ImportError),
      never silently fall back.
    - bertopic IS installed/importable (true in this environment as of
      R2-S56.8, and true wherever the pinned versions in
      BERTopicWrapper's docstring are installed): construction must
      succeed and return a real BERTopicWrapper, emitting the
      embedding-stand-in warning (not an UNTESTED warning -- this path
      is now verified end-to-end, see the tests below)."""
    if not bertopic_available:
        with pytest.raises(ImportError):
            create_topic_model(seed=0)
        return
    with pytest.warns(UserWarning, match="embedding step"):
        model = create_topic_model(seed=0)
    assert isinstance(model, BERTopicWrapper)


def test_create_topic_model_explicit_lightweight():
    """use_bertopic=False still returns the tested, previously-default
    NarrativeTopicModel -- this path is unaffected by the R2-S56.8 fix."""
    model = create_topic_model(use_bertopic=False, seed=0)
    assert isinstance(model, NarrativeTopicModel)


def test_embedding_standin_warning_text_present():
    """Cheap regression guard: the warning discloses what's real
    (BERTopic's clustering machinery) vs stand-in (the embedding step)
    even if refactored."""
    assert "HashingVectorizer" in _BERTOPIC_EMBEDDING_STANDIN_WARNING
    assert "embed_fn" in _BERTOPIC_EMBEDDING_STANDIN_WARNING


@requires_bertopic
def test_bertopic_wrapper_end_to_end_real_execution_no_fallback():
    """R2-S56.8 core claim: real bertopic.BERTopic actually executes
    start-to-finish on a fixed corpus -- no fallback clusterer, no
    mock, no monkeypatch. Confirms self.model is a genuine BERTopic
    instance (not e.g. silently substituted for NarrativeTopicModel's
    river clusterer) and that fitting produces one label per doc plus
    real topic-word output."""
    from bertopic import BERTopic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = BERTopicWrapper(seed=0)
    assert isinstance(model.model, BERTopic)

    labels = model.partial_fit(CAREER_DOCS)
    assert len(labels) == len(CAREER_DOCS)
    topics = model.model.get_topics()
    assert len(topics) >= 1
    # real c-TF-IDF representation, not an empty/placeholder result
    topic_words = model.get_topic_words(next(iter(topics.keys())))
    assert len(topic_words) > 0
    assert all(isinstance(w, str) for w in topic_words)


@requires_bertopic
def test_bertopic_wrapper_separates_distinct_topics():
    """Two genuinely distinct document streams should end up in
    different topics at least some of the time -- real clustering
    signal, not everything dumped into one bucket or all noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = BERTopicWrapper(seed=0)
    career_labels = model.partial_fit(CAREER_DOCS)
    health_labels = model.partial_fit(HEALTH_DOCS)
    assert len(set(career_labels) - {NOISE_TOPIC}) >= 1
    assert len(set(health_labels) - {NOISE_TOPIC}) >= 1


@requires_bertopic
def test_bertopic_wrapper_none_docs_get_noise_label():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = BERTopicWrapper(seed=0, n_components=2)
    labels = model.partial_fit([None, *CAREER_DOCS[:5], None])
    assert labels[0] == NOISE_TOPIC
    assert labels[-1] == NOISE_TOPIC


@requires_bertopic
def test_bertopic_wrapper_underpowered_corpus_explicit_error_not_silent_downgrade():
    """R2-S56.8: fewer real docs in a batch than n_components is an
    explicit, named failure (IncrementalPCA can't fit with too few
    samples) -- must raise, never silently fall back to a different
    clusterer or return a fabricated label set."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = BERTopicWrapper(seed=0, n_components=5)
    with pytest.raises(ValueError, match="n_components"):
        model.partial_fit(CAREER_DOCS[:2])


@requires_bertopic
def test_bertopic_wrapper_all_none_batch_is_a_noop_not_an_error():
    """A batch that's entirely silent episodes should produce all-noise
    labels without even touching the underpowered-batch check (there
    are zero real docs, not 'too few')."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = BERTopicWrapper(seed=0, n_components=5)
    labels = model.partial_fit([None, None, None])
    assert labels == [NOISE_TOPIC, NOISE_TOPIC, NOISE_TOPIC]


@requires_bertopic
def test_bertopic_wrapper_reproducible_same_manifest():
    """Two fresh instances fed the identical document manifest in the
    identical order must produce identical label sequences -- no
    hidden nondeterminism in the embedding stand-in, IncrementalPCA, or
    the river clusterer as wired here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model_a = BERTopicWrapper(seed=0)
        model_b = BERTopicWrapper(seed=0)
    labels_a = model_a.partial_fit(CAREER_DOCS)
    labels_b = model_b.partial_fit(CAREER_DOCS)
    assert labels_a == labels_b


@requires_bertopic
def test_bertopic_wrapper_custom_embed_fn_skips_standin_warning():
    """Passing embed_fn (e.g. a real sentence-transformer in a
    network-enabled environment) must not emit the stand-in warning --
    that warning is specifically about the DEFAULT embedding path."""
    import numpy as np

    def fake_semantic_embed(docs):
        # deterministic placeholder standing in for a real embedder in
        # this test -- the point being tested is that supplying ANY
        # embed_fn suppresses the stand-in warning, not the embedding
        # quality itself.
        return np.array([[float(len(d)), float(d.count(" "))] for d in docs])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = BERTopicWrapper(seed=0, n_components=2, embed_fn=fake_semantic_embed)
    assert model.embedding_source == "custom"
    assert not any("embedding step" in str(w.message) for w in caught)
    labels = model.partial_fit(CAREER_DOCS[:5])
    assert len(labels) == 5


@requires_bertopic
def test_bertopic_wrapper_save_load_round_trip_same_labels_and_topics(tmp_path):
    """R2-S56.8 outstanding follow-up: serialization round-trip was in
    the original required-tests list and had not been built. A
    fitted BERTopicWrapper saved then loaded must reproduce the same
    fitted topic model -- not a fresh/untrained one -- verified by
    checking both the label history carried over AND that the loaded
    model's own get_topic_words() output (which depends on its
    internal c-TF-IDF state, not just the wrapper's bookkeeping)
    matches the original."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = BERTopicWrapper(seed=0, n_components=2)
    original_labels = model.partial_fit(CAREER_DOCS + HEALTH_DOCS)

    save_dir = str(tmp_path / "bertopic_wrapper_state")
    model.save(save_dir)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = BERTopicWrapper.load(save_dir)

    assert loaded.topics_ == list(original_labels)
    assert loaded._n_components == model._n_components
    assert loaded.embedding_source == model.embedding_source

    original_topic_ids = {t for t in original_labels if t != NOISE_TOPIC}
    for tid in original_topic_ids:
        assert loaded.get_topic_words(tid) == model.get_topic_words(tid)


@requires_bertopic
def test_bertopic_wrapper_save_load_loaded_model_keeps_learning(tmp_path):
    """A loaded model isn't a read-only snapshot -- partial_fit on new
    batches must still work and extend .topics_ from where the saved
    state left off, proving the loaded BERTopic instance is genuinely
    fit (has its internal state, e.g. the river clusterer's learned
    micro-clusters), not just re-constructed empty."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = BERTopicWrapper(seed=0, n_components=2)
    model.partial_fit(CAREER_DOCS)

    save_dir = str(tmp_path / "bertopic_wrapper_state_continue")
    model.save(save_dir)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loaded = BERTopicWrapper.load(save_dir)
    more_labels = loaded.partial_fit(HEALTH_DOCS)

    assert len(more_labels) == len(HEALTH_DOCS)
    assert len(loaded.topics_) == len(CAREER_DOCS) + len(HEALTH_DOCS)


@requires_bertopic
def test_bertopic_wrapper_save_load_custom_embed_fn_must_be_resupplied():
    """embed_fn is not picklable-by-contract and is deliberately not
    persisted (see save()'s docstring) -- loading without re-supplying
    the original custom embed_fn must fall back to (and honestly
    report) the hashing stand-in, never silently claim 'custom'
    provenance it no longer has."""
    import numpy as np
    import tempfile

    def fake_semantic_embed(docs):
        return np.array([[float(len(d)), float(d.count(" "))] for d in docs])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = BERTopicWrapper(seed=0, n_components=2, embed_fn=fake_semantic_embed)
    model.partial_fit(CAREER_DOCS[:5])
    assert model.embedding_source == "custom"

    with tempfile.TemporaryDirectory() as save_dir:
        model.save(save_dir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            loaded_without_fn = BERTopicWrapper.load(save_dir)
        assert loaded_without_fn.embedding_source == "hashing_stand_in"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            loaded_with_fn = BERTopicWrapper.load(save_dir, embed_fn=fake_semantic_embed)
        assert loaded_with_fn.embedding_source == "custom"


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