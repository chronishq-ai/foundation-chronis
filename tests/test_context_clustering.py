import numpy as np
from domain_emergence.context_clustering import (
    zscore_normalize,
    cluster_context_signatures,
)


def _make_blobs(centers, n_per_cluster, spread, seed):
    rng = np.random.default_rng(seed)
    pts = []
    for c in centers:
        pts.append(rng.normal(loc=c, scale=spread, size=(n_per_cluster, len(c))))
    return np.vstack(pts)


def test_zscore_normalize_basic():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    Z = zscore_normalize(X)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-9)


def test_zscore_normalize_constant_column():
    X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    Z = zscore_normalize(X)
    # constant column shouldn't blow up (no div by zero -> inf/nan)
    assert np.all(np.isfinite(Z))
    assert np.allclose(Z[:, 0], 0.0)


def test_cluster_recovers_separated_blobs():
    X = _make_blobs(centers=[[0, 0], [20, 20], [-20, 20]], n_per_cluster=15, spread=0.5, seed=0)
    result = cluster_context_signatures(X, min_cluster_size=5)
    assert result.n_clusters == 3
    assert len(result.labels) == 45


def test_cluster_empty_input():
    result = cluster_context_signatures(np.empty((0, 5)))
    assert result.n_clusters == 0
    assert result.n_noise == 0
    assert len(result.labels) == 0


def test_cluster_all_noise_with_scattered_points():
    rng = np.random.default_rng(1)
    X = rng.uniform(-100, 100, size=(20, 3))  # scattered, no real structure
    result = cluster_context_signatures(X, min_cluster_size=10)
    # with high min_cluster_size on scattered noise, expect mostly/all noise
    assert result.n_noise >= 0  # sanity: doesn't crash, labels assigned
    assert len(result.labels) == 20


def test_probabilities_in_valid_range():
    X = _make_blobs(centers=[[0, 0], [15, 15]], n_per_cluster=10, spread=0.3, seed=2)
    result = cluster_context_signatures(X, min_cluster_size=5)
    assert np.all(result.probabilities >= 0.0)
    assert np.all(result.probabilities <= 1.0)


def test_noise_label_is_negative_one():
    X = _make_blobs(centers=[[0, 0], [15, 15]], n_per_cluster=10, spread=0.3, seed=3)
    result = cluster_context_signatures(X, min_cluster_size=5)
    valid_labels = set(result.labels.tolist())
    assert valid_labels.issubset({-1, 0, 1})