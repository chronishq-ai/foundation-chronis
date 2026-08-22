"""
Day 17 -- Online/streaming narrative topic modeling.

Bible Part 5.8 specifies BERTopic in online/streaming mode for the narrative
side of domain emergence. The real `bertopic` package crashes on import in
this dev sandbox (SIGBUS, binary/numba incompatibility) and its default
embedding backend requires downloading a model from huggingface.co, which
this sandbox's network policy does not allow. Neither issue is doctrinal.

S56.9 UPDATE: `create_topic_model()`'s default is now `use_bertopic=True`
(BERTopicWrapper, at the bottom of this file), flipped per explicit
instruction. This is NOT a claim that BERTopicWrapper has been verified --
it has never been run successfully in any environment available here; it is
only structurally reviewed against BERTopic's documented online-mode API.
BERTopicWrapper raises a loud UserWarning on every instantiation for exactly
this reason. Run its own smoke test (`pip install bertopic
sentence-transformers river` somewhere bertopic actually imports, e.g. off
this sandbox) before trusting its output. Pass `use_bertopic=False` to get
NarrativeTopicModel -- the lightweight, fully-tested implementation below,
and the module's previous default:
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

# HONESTY FLAG (S56.9): flipped per explicit instruction, NOT because
# BERTopicWrapper has been verified. It has not been smoke-tested even
# once -- bertopic SIGBUS-crashes on import in this dev sandbox and its
# default embedder needs huggingface.co network access this sandbox
# also blocks, so there was no way to actually run it here before
# flipping this default. See BERTopicWrapper's docstring for what WAS
# checked (structural review against BERTopic's documented online-mode
# API) versus what was NOT (any real execution). This class raises a
# loud UserWarning on every instantiation for exactly this reason --
# do not silence/filter that warning without first running
# BERTopicWrapper's own smoke test in an environment where bertopic
# actually imports.
_BERTOPIC_UNVERIFIED_WARNING = (
    "S56.9: create_topic_model() now defaults to BERTopicWrapper "
    "(use_bertopic=True). BERTopicWrapper is UNTESTED -- it has never "
    "been run successfully; bertopic SIGBUS-crashes on import in the "
    "sandbox this was built in, and its default embedder needs "
    "huggingface.co network access that sandbox also blocked. This "
    "default was flipped on explicit instruction, not because the "
    "class was verified. Run BERTopicWrapper's own smoke test before "
    "trusting output from this path. Pass use_bertopic=False to get "
    "the lightweight (tested, previously-default) NarrativeTopicModel "
    "instead."
)


def create_topic_model(use_bertopic: bool = True, **kwargs):
    """S56.9 wiring point: the module previously had no way to select
    BERTopicWrapper -- callers always got NarrativeTopicModel regardless
    of what Sprint 6's spec requires. This factory makes the choice
    explicit and callable in one place.

    use_bertopic=True (DEFAULT, flipped per S56.9 instruction) returns
    BERTopicWrapper -- the real BERTopic path Bible Part 5.8 specifies.
    UNVERIFIED: raises a loud UserWarning on instantiation (see
    _BERTOPIC_UNVERIFIED_WARNING) because this class has never actually
    been run -- see its docstring. Flipping this default was done on
    explicit request, ahead of the real smoke test / senior
    confirmation the pack's Ownership Model would otherwise require
    first; treat any output from this path as unverified until that
    smoke test happens in an environment where bertopic can import.

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


class BERTopicWrapper:
    """Real BERTopic in online/streaming mode, per Bible 5.8's literal spec.

    UNTESTED end-to-end -- bertopic cannot import in this sandbox (SIGBUS
    on import) and its default embedder needs huggingface.co (blocked
    here), so this class has only been reviewed against BERTopic's
    documented online-mode API, never actually run start-to-finish.

    CONFIRMED REAL BUG, NOW FIXED (not hypothetical): the previous
    version of this class did `from bertopic.cluster import River` --
    that name has never existed in bertopic.cluster (confirmed against
    BERTopic's own docs/source: bertopic.cluster only ships BaseCluster).
    On a real Windows install with bertopic actually present, this
    failed immediately with ImportError, proving it. BERTopic's own
    "Online Topic Modeling" docs show `River` is not an importable
    class at all -- it's a ~10-line wrapper YOU write, adapting any
    river.cluster model to the fit/predict interface BERTopic's
    hdbscan_model slot expects. `_RiverClusterWrapper` below is that
    wrapper, copied from BERTopic's own documented example. This is a
    genuine fix to a real bug, not just a version-pin issue -- but the
    class remains UNVERIFIED end-to-end (no environment available here
    has successfully run bertopic to confirm this now actually works).

    Same interface contract as NarrativeTopicModel: .partial_fit(docs)
    takes a batch (None = silent episode, skipped), returns per-doc labels
    for that batch; .topics_ holds the full label history after fitting.
    """

    def __init__(self, n_components: int = 5, seed: int | None = None):
        # Loud warning BEFORE the risky imports below, so it fires even
        # if bertopic then crashes on import -- caller/log should never
        # be left wondering whether this path was ever verified.
        warnings.warn(_BERTOPIC_UNVERIFIED_WARNING, UserWarning, stacklevel=2)

        from bertopic import BERTopic
        from bertopic.vectorizers import OnlineCountVectorizer
        from river import cluster as river_cluster
        from river import stream as river_stream
        from sklearn.decomposition import IncrementalPCA

        umap_model = IncrementalPCA(n_components=n_components)
        cluster_model = _RiverClusterWrapper(river_cluster.DBSTREAM(), river_stream)
        vectorizer_model = OnlineCountVectorizer(stop_words="english")

        self.model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model,
        )
        self.topics_: list = []
        self._doc_index_map: list = []  # tracks None-slots vs real-doc slots per batch

    def partial_fit(self, docs: list) -> list:
        real_docs = [d for d in docs if d is not None]
        slot_is_real = [d is not None for d in docs]

        batch_labels_real = []
        if real_docs:
            self.model.partial_fit(real_docs)
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


class _RiverClusterWrapper:
    """Adapts any river.cluster incremental model to the fit/transform
    interface BERTopic's `hdbscan_model` slot expects, so it can be used
    for genuinely online clustering. Copied from BERTopic's own
    documented "Online Topic Modeling" example (bertopic.readthedocs.io) --
    NOT importable from the bertopic package itself (see class docstring
    above for why the previous `from bertopic.cluster import River` was
    a real, confirmed bug)."""

    def __init__(self, model, river_stream_module):
        self.model = model
        self._stream = river_stream_module

    def partial_fit(self, umap_embeddings):
        for umap_embedding, _ in self._stream.iter_array(umap_embeddings):
            self.model.learn_one(umap_embedding)

        labels = []
        for umap_embedding, _ in self._stream.iter_array(umap_embeddings):
            labels.append(self.model.predict_one(umap_embedding))
        self.labels_ = labels
        return self
