"""
Day 17 -- Online/streaming narrative topic modeling.

Bible Part 5.8 specifies BERTopic in online/streaming mode for the narrative
side of domain emergence. The real `bertopic` package crashes on import in
this dev sandbox (SIGBUS, binary/numba incompatibility) and its default
embedding backend requires downloading a model from huggingface.co, which
this sandbox's network policy does not allow. Neither issue is doctrinal --
try `pip install bertopic sentence-transformers river` in your own Windows
venv; if it imports cleanly there, swap NarrativeTopicModel below for the
BERTopicWrapper class at the bottom of this file (same interface:
.partial_fit(docs) / .topics_). BERTopicWrapper is UNTESTED -- only
structurally reviewed against BERTopic's documented online-mode API, never
actually run, since bertopic can't import in this sandbox at all. Run its
own smoke test before trusting it.

Until then, this module implements the SAME algorithmic shape the doctrine
calls for, without the bertopic dependency:
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
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from river import cluster
from sklearn.feature_extraction.text import HashingVectorizer


NOISE_TOPIC = -1


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

    UNTESTED -- bertopic cannot import in this sandbox (SIGBUS on import)
    and its default embedder needs huggingface.co (blocked here), so this
    class has only been reviewed against BERTopic's documented online-mode
    API (umap_model=IncrementalPCA, hdbscan_model=river cluster via the
    River wrapper, vectorizer_model=OnlineCountVectorizer), never actually
    run. Try it in your Windows venv where network access to huggingface.co
    works; if `pip install bertopic sentence-transformers river` succeeds
    and this smoke-tests clean, swap it in for NarrativeTopicModel above.

    Same interface contract as NarrativeTopicModel: .partial_fit(docs)
    takes a batch (None = silent episode, skipped), returns per-doc labels
    for that batch; .topics_ holds the full label history after fitting.
    """

    def __init__(self, n_components: int = 5, seed: int | None = None):
        from bertopic import BERTopic
        from bertopic.vectorizers import OnlineCountVectorizer
        from bertopic.cluster import River
        from river import cluster as river_cluster
        from sklearn.decomposition import IncrementalPCA

        umap_model = IncrementalPCA(n_components=n_components)
        cluster_model = River(river_cluster.DBSTREAM())
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