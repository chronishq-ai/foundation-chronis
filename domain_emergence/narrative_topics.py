"""
Day 17 -- Online/streaming narrative topic modeling.

Bible Part 5.8 specifies BERTopic in online/streaming mode for the narrative
side of domain emergence.

R2-S56.8 UPDATE (item 7): `bertopic`/`hdbscan`/`river` now import and run
successfully in this environment (bertopic==0.17.4, hdbscan==0.8.44,
umap-learn==0.5.12, river==0.26.1, scikit-learn==1.8.0, numpy==2.4.4,
Python 3.12.3) -- the SIGBUS-on-import failure noted in an earlier pass
was specific to that other sandbox's numba/binary setup, not reproduced
here. BERTopicWrapper (bottom of this file) has now been run end-to-end
against real installs, not just structurally reviewed -- see
tests/test_narrative_topics.py for the integration tests that prove
this (real fit on a fixed corpus, underpowered-corpus behavior,
run-to-run reproducibility). One real blocker remains and is disclosed,
not hidden: this network policy still cannot reach huggingface.co, so
BERTopic's default sentence-transformer embedding step cannot run here
either. BERTopicWrapper works around this by precomputing embeddings
itself (default: a deterministic HashingVectorizer stand-in, with an
`embed_fn` hook for a real semantic embedder in a network-enabled
environment) -- see `_BERTOPIC_EMBEDDING_STANDIN_WARNING` and the class
docstring for exactly what that does and does not verify.

`create_topic_model()`'s default remains `use_bertopic=True`
(BERTopicWrapper). Pass `use_bertopic=False` to get NarrativeTopicModel
-- the lightweight, fully-tested implementation below, kept as the
network-free fallback for environments where even the stand-in
embedding step isn't wanted:
  - embed short texts (hashing-trick TF vectors -- deterministic, no network,
    no pretrained model download; stands in for the semantic embedding step)
  - cluster ONLINE/STREAMING via river.cluster.DBSTREAM (density-based,
    incremental -- same spirit as BERTopic's default HDBSCAN-on-UMAP, but
    genuinely online rather than requiring the full corpus up front)
  - represent each topic by its top words via class-based TF-IDF (c-TF-IDF),
    matching BERTopic's own topic-representation method

None entries in `docs` (silent episodes, see synthetic_transcripts.py) are
skipped entirely -- they contribute no narrative signal, by design.
"""

from __future__ import annotations
import warnings
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from river import cluster
from sklearn.feature_extraction.text import HashingVectorizer


NOISE_TOPIC = -1


def create_topic_model(use_bertopic: bool = True, **kwargs):
    """S56.9 wiring point: the module previously had no way to select
    BERTopicWrapper -- callers always got NarrativeTopicModel regardless
    of what Sprint 6's spec requires. This factory makes the choice
    explicit and callable in one place.

    use_bertopic=True (DEFAULT) returns BERTopicWrapper -- the real
    BERTopic path Bible Part 5.8 specifies, now verified end-to-end in
    this environment (R2-S56.8, see BERTopicWrapper's docstring).
    Raises a UserWarning on instantiation unless `embed_fn` is passed,
    since the default embedding step is a network-free stand-in, not
    the doctrinal semantic embedding (see
    _BERTOPIC_EMBEDDING_STANDIN_WARNING).

    use_bertopic=False returns NarrativeTopicModel -- the lightweight
    hashing-trick + river streaming clusterer documented above. This
    was the previous default and remains fully tested; pass this
    explicitly if you need the previously-default, known-working
    behavior."""
    if use_bertopic:
        return BERTopicWrapper(**kwargs)
    return NarrativeTopicModel(**kwargs)


@dataclass
class TopicModelResult:
    topic_labels: list          # one per INPUT doc (including None for silent), -1 = noise/no-doc
    topic_words: dict           # {topic_id: [top words]}
    n_topics: int


class NarrativeTopicModel:
    """Online/streaming topic model. Call .partial_fit(docs) per batch of
    episodes as they arrive (matches doctrine's streaming requirement --
    never needs the full corpus at once). .finalize() computes c-TF-IDF
    topic-word representations from everything seen so far."""

    def __init__(self, n_features: int = 128, seed: int | None = None):
        self.vectorizer = HashingVectorizer(
            n_features=n_features, alternate_sign=False, norm=None
        )
        self.clusterer = cluster.DBSTREAM()
        self._word_counts_per_topic: dict = defaultdict(Counter)
        self._doc_topic_labels: list = []
        self._raw_texts: list = []

    def _embed(self, text: str) -> dict:
        """HashingVectorizer -> sparse row -> river-friendly dict of
        {feature_index: weight}, since river's streaming API takes dicts."""
        vec = self.vectorizer.transform([text])
        coo = vec.tocoo()
        return {int(j): float(v) for j, v in zip(coo.col, coo.data)}

    def partial_fit(self, docs: list) -> list:
        """Feed one batch of docs (episode texts; None = silent episode,
        skipped). Returns the topic label assigned to each doc in THIS
        batch (-1 for None entries)."""
        batch_labels = []
        for text in docs:
            if text is None:
                batch_labels.append(NOISE_TOPIC)
                self._doc_topic_labels.append(NOISE_TOPIC)
                self._raw_texts.append(None)
                continue

            x = self._embed(text)
            self.clusterer.learn_one(x)
            label = self.clusterer.predict_one(x)
            label = int(label) if label is not None else NOISE_TOPIC

            batch_labels.append(label)
            self._doc_topic_labels.append(label)
            self._raw_texts.append(text)

            if label != NOISE_TOPIC:
                for word in text.lower().split():
                    self._word_counts_per_topic[label][word] += 1

        return batch_labels

    def finalize(self, top_n_words: int = 5) -> TopicModelResult:
        """Compute c-TF-IDF top words per topic from everything seen via
        partial_fit so far, and return the full result."""
        topic_ids = [t for t in self._word_counts_per_topic.keys() if t != NOISE_TOPIC]

        # class-based TF-IDF: term freq within topic, weighted down by how
        # many topics a term appears across (matches BERTopic's own c-TF-IDF)
        doc_freq: Counter = Counter()
        for tid in topic_ids:
            for word in self._word_counts_per_topic[tid]:
                doc_freq[word] += 1

        n_topics = max(len(topic_ids), 1)
        topic_words = {}
        for tid in topic_ids:
            counts = self._word_counts_per_topic[tid]
            total = sum(counts.values()) or 1
            scored = []
            for word, c in counts.items():
                tf = c / total
                idf = np.log(1 + n_topics / doc_freq[word])
                scored.append((word, tf * idf))
            scored.sort(key=lambda x: -x[1])
            topic_words[tid] = [w for w, _ in scored[:top_n_words]]

        return TopicModelResult(
            topic_labels=list(self._doc_topic_labels),
            topic_words=topic_words,
            n_topics=len(topic_ids),
        )


_BERTOPIC_EMBEDDING_STANDIN_WARNING = (
    "R2-S56.8: BERTopicWrapper's embedding step defaults to a "
    "deterministic HashingVectorizer stand-in, NOT the doctrinal "
    "sentence-transformer semantic embedding Bible Part 5.8 specifies "
    "-- huggingface.co (needed to download the default "
    "all-MiniLM-L6-v2 model) is unreachable under this sandbox's "
    "network policy (confirmed: raises OSError/LocalEntryNotFoundError "
    "here). The real BERTopic online-mode machinery -- river-backed "
    "incremental clustering via _RiverClusterWrapper, IncrementalPCA "
    "dimensionality reduction, c-TF-IDF topic representation -- IS now "
    "verified executing end-to-end against real bertopic/hdbscan/river "
    "installs (see tests/test_narrative_topics.py). Only the embedding "
    "INPUT is a stand-in: topic separation here reflects hashing-trick "
    "lexical overlap, not semantic similarity. Pass "
    "embed_fn=<callable taking a list[str], returning an (n_docs, dim) "
    "array> -- e.g. a real sentence-transformer's .encode -- in an "
    "environment where huggingface.co is reachable, to get the "
    "doctrinal semantic embeddings this class was designed around."
)


class BERTopicWrapper:
    """Real BERTopic in online/streaming mode, per Bible 5.8's literal spec.

    R2-S56.8 FIX (item 7): now verified end-to-end (see
    tests/test_narrative_topics.py) against real installs of
    bertopic==0.17.4, hdbscan==0.8.44, umap-learn==0.5.12,
    river==0.26.1, scikit-learn==1.8.0, numpy==2.4.4, on Python 3.12.3
    -- this is a real execution record, not a structural-review-only
    claim like the previous version of this docstring made. Two real
    blockers were found and fixed:

    1. CONFIRMED BUG (previously): `from bertopic.cluster import River`
       -- that name has never existed in bertopic.cluster (confirmed
       against bertopic's own source: bertopic.cluster only ships
       BaseCluster). Fixed by writing `_RiverClusterWrapper` --
       BERTopic's own documented ~10-line adapter pattern for plugging
       any river.cluster model into the `hdbscan_model` slot. This fix
       is now proven by real execution, not just source inspection.
    2. BERTopic's DEFAULT embedding backend
       (sentence-transformers/all-MiniLM-L6-v2) downloads weights from
       huggingface.co on first use -- confirmed unreachable here
       (OSError/LocalEntryNotFoundError). Worked around by precomputing
       document embeddings ourselves and passing them via
       `BERTopic.partial_fit(docs, embeddings=...)`, which skips
       BERTopic's `select_backend()` call entirely (confirmed by
       reading `BERTopic.partial_fit`'s source: the sentence-transformer
       backend is only instantiated when `embeddings is None`). BY
       DEFAULT this uses a deterministic HashingVectorizer stand-in,
       NOT real semantic embeddings -- see
       `_BERTOPIC_EMBEDDING_STANDIN_WARNING`, which fires on every
       instantiation unless `embed_fn` is supplied. Pass a real
       sentence-transformer (or other embedding model) via `embed_fn`
       in a network-enabled environment to get doctrinal semantic
       embeddings.

    Seed policy: `seed` is accepted for interface parity with
    NarrativeTopicModel, but neither IncrementalPCA, river's DBSTREAM,
    nor the default HashingVectorizer embedding step have a meaningful
    stochastic seeding surface here -- output is otherwise deterministic
    given the same input order/batching (verified by
    test_bertopic_wrapper_reproducible_same_manifest).

    Underpowered-corpus policy: with very few/short documents, the
    online river clusterer may legitimately assign every document to
    the noise topic (-1) rather than forming any topic -- that is a
    real, explicit "no topic found yet" result, not a silent fallback
    or crash (see test_bertopic_wrapper_underpowered_corpus_all_noise).

    Same interface contract as NarrativeTopicModel: .partial_fit(docs)
    takes a batch (None = silent episode, skipped), returns per-doc
    labels for that batch; .topics_ holds the full label history after
    fitting. Requires each batch to contain at least `n_components`
    real (non-None) docs -- IncrementalPCA's partial_fit needs at least
    that many samples; raises ValueError otherwise rather than a cryptic
    sklearn error.
    """

    def __init__(self, n_components: int = 5, n_features: int = 128,
                 seed: int | None = None, embed_fn=None):
        from bertopic import BERTopic
        from bertopic.vectorizers import OnlineCountVectorizer
        from river import cluster as river_cluster
        from river import stream as river_stream
        from sklearn.decomposition import IncrementalPCA

        if embed_fn is not None:
            self._embed_fn = embed_fn
            self.embedding_source = "custom"
        else:
            hashing_vectorizer = HashingVectorizer(
                n_features=n_features, alternate_sign=False, norm=None
            )
            self._embed_fn = lambda docs: hashing_vectorizer.transform(docs).toarray()
            self.embedding_source = "hashing_stand_in"
            warnings.warn(_BERTOPIC_EMBEDDING_STANDIN_WARNING, UserWarning, stacklevel=2)

        self._n_components = n_components
        umap_model = IncrementalPCA(n_components=n_components)
        cluster_model = _RiverClusterWrapper(river_cluster.DBSTREAM(), river_stream)
        vectorizer_model = OnlineCountVectorizer(stop_words="english")

        self.model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model,
        )
        self.topics_: list = []

    def partial_fit(self, docs: list) -> list:
        real_docs = [d for d in docs if d is not None]
        slot_is_real = [d is not None for d in docs]

        if real_docs and len(real_docs) < self._n_components:
            raise ValueError(
                f"BERTopicWrapper.partial_fit received {len(real_docs)} "
                f"real doc(s) but n_components={self._n_components} -- "
                "IncrementalPCA (the umap_model slot here) requires at "
                "least n_components samples per partial_fit call. Batch "
                "more episodes together, or construct with a smaller "
                "n_components."
            )

        batch_labels_real = []
        if real_docs:
            embeddings = self._embed_fn(real_docs)
            self.model.partial_fit(real_docs, embeddings=embeddings)
            batch_labels_real = list(self.model.topics_[-len(real_docs):])

        batch_labels = []
        it = iter(batch_labels_real)
        for is_real in slot_is_real:
            label = next(it) if is_real else NOISE_TOPIC
            batch_labels.append(label)
            self.topics_.append(label)

        return batch_labels

    def get_topic_words(self, topic_id: int, top_n_words: int = 5) -> list:
        words = self.model.get_topic(topic_id)
        if not words:
            return []
        return [w for w, _ in words[:top_n_words]]

    def save(self, path: str) -> None:
        """R2-S56.8 follow-up (item 7's disclosed remaining gap): save/load
        round-trip, previously in the required-tests list but not built.

        Writes two things under `path` (created if missing):
          - `bertopic_model/` -- the real fitted BERTopic instance, via
            BERTopic's own `.save(..., serialization="pickle")`. Pickle
            (not safetensors) is required here because our `umap_model`
            (IncrementalPCA) and `hdbscan_model` (`_RiverClusterWrapper`
            wrapping a river DBSTREAM instance) are not the standard
            sklearn/HDBSCAN* objects BERTopic's safetensors path assumes
            -- pickle preserves them as-is.
          - `wrapper_state.json` -- everything BERTopic's own save doesn't
            know about: `n_components`, `embedding_source`, and the full
            `topics_` label history (BERTopic's on-disk state does not
            include this wrapper's per-call bookkeeping).

        NOT saved: `embed_fn`. An arbitrary caller-supplied callable
        (e.g. a real sentence-transformer's `.encode`) is not reliably
        picklable and is not this wrapper's to serialize -- pass the
        same `embed_fn` again to `load()` if one was used originally.
        Loading without it when the original used a custom embed_fn
        silently falls back to the hashing stand-in for any *future*
        `partial_fit` calls (already-fitted state is unaffected); this
        is flagged via `embedding_source` on the loaded instance so a
        caller can check `wrapper.embedding_source` post-load rather
        than discovering the mismatch implicitly.
        """
        import json
        import os

        os.makedirs(path, exist_ok=True)
        self.model.save(
            os.path.join(path, "bertopic_model"),
            serialization="pickle",
            save_embedding_model=False,
        )
        with open(os.path.join(path, "wrapper_state.json"), "w") as f:
            json.dump(
                {
                    "n_components": self._n_components,
                    "embedding_source": self.embedding_source,
                    "topics_": list(self.topics_),
                },
                f,
            )

    @classmethod
    def load(cls, path: str, embed_fn=None, n_features: int = 128,
              seed: int | None = None) -> "BERTopicWrapper":
        """Inverse of `save()`. Reconstructs a `BERTopicWrapper` whose
        `.model` is the real fitted BERTopic instance from disk (not a
        fresh/untrained one) and whose `.topics_` history matches what
        was saved.

        Pass the SAME `embed_fn` used before `save()` if a custom one
        was supplied -- see `save()`'s docstring for why it isn't
        persisted. Instantiating without it re-triggers
        `_BERTOPIC_EMBEDDING_STANDIN_WARNING` (same as any other
        default-path construction), since the loaded instance would
        otherwise silently claim doctrinal-embedding provenance it
        does not have for any subsequent `partial_fit` call.
        """
        import json
        import os
        from bertopic import BERTopic

        with open(os.path.join(path, "wrapper_state.json")) as f:
            state = json.load(f)

        wrapper = cls(
            n_components=state["n_components"],
            n_features=n_features,
            seed=seed,
            embed_fn=embed_fn,
        )
        wrapper.model = BERTopic.load(os.path.join(path, "bertopic_model"))
        wrapper.topics_ = list(state["topics_"])
        # NOTE: embedding_source is deliberately left as whatever cls()'s
        # own __init__ just set (based on whether embed_fn is None here,
        # not on the saved state) -- see save()'s docstring: embed_fn
        # itself isn't persisted, so state["embedding_source"] reflects
        # what was true when SAVED, not what's true now. Copying it in
        # unconditionally is a real bug found while testing this: it
        # would report "custom" on a load that has no embed_fn at all.
        return wrapper


class _RiverClusterWrapper:
    """Adapts any river.cluster incremental model to the fit/transform
    interface BERTopic's `hdbscan_model` slot expects, so it can be used
    for genuinely online clustering. Copied from BERTopic's own
    documented "Online Topic Modeling" example (bertopic.readthedocs.io) --
    NOT importable from the bertopic package itself (see class docstring
    above for why the previous `from bertopic.cluster import River` was
    a real, confirmed bug).

    R2-S56.6-follow-up bug fix (found while building save()/load()):
    the original version stored the raw `river.stream` MODULE object
    (`self._stream = river_stream_module`) so it could call
    `self._stream.iter_array(...)`. Python's pickle cannot serialize a
    module object (`TypeError: cannot pickle 'module' object`) -- this
    only surfaced once `BERTopic.save(..., serialization="pickle")`
    actually tried to pickle a fitted model end-to-end (this class had
    only been exercised via live objects before, never round-tripped
    through disk). Fixed by storing the specific `iter_array` FUNCTION
    reference instead of the module -- module-level functions pickle
    fine by reference (module path + qualname), whole modules don't.
    """

    def __init__(self, model, river_stream_module):
        self.model = model
        self._iter_array = river_stream_module.iter_array

    def partial_fit(self, umap_embeddings):
        for umap_embedding, _ in self._iter_array(umap_embeddings):
            self.model.learn_one(umap_embedding)

        labels = []
        for umap_embedding, _ in self._iter_array(umap_embeddings):
            labels.append(self.model.predict_one(umap_embedding))
        self.labels_ = labels
        return self