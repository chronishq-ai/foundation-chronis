"""
Day 16 -- HDBSCAN clustering of context signatures into candidate domains.

Takes the (n_episodes, feature_dim) signature matrix from
context_signature.build_context_signatures() and clusters it with HDBSCAN.
Output cluster labels are RAW candidate domains -- Fisher's-exact alignment,
domain-count bounds (2-8), split/merge logic, and confidence scoring are
later Day 17/18 steps, not done here. This module's job stops at: signatures
in, candidate cluster labels out, plus basic diagnostics.

Noise points (HDBSCAN label -1) are real and expected -- episodes that don't
fit any recurring domain pattern. Never forced into a cluster.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

try:
    import hdbscan
except ImportError as e:
    raise ImportError(
        "hdbscan not installed. Run: pip install hdbscan --break-system-packages "
        "(or without the flag on non-managed envs)."
    ) from e


@dataclass
class ClusteringResult:
    labels: np.ndarray          # (n_episodes,) int, -1 = noise
    n_clusters: int              # count of non-noise clusters
    n_noise: int                  # count of noise-labeled episodes
    probabilities: np.ndarray    # (n_episodes,) HDBSCAN membership strength


def zscore_normalize(X: np.ndarray) -> np.ndarray:
    """Standardize each feature column to mean 0, std 1. Guards against
    zero-variance columns (constant feature) by leaving them at 0 rather
    than dividing by zero."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    Z = (X - mean) / std_safe
    # zero-variance columns: result is already ~0 after centering, keep as is
    return Z


def cluster_context_signatures(
    signatures: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
) -> ClusteringResult:
    """Z-score normalize signatures, then run HDBSCAN. min_cluster_size is
    the key knob: smaller = more, smaller candidate domains; larger = fewer,
    coarser domains. Tune against Sprint 6 DoD (2-8 domains per user on
    surrogate data) once wired to real per-user signature sets."""
    if signatures.shape[0] == 0:
        return ClusteringResult(
            labels=np.array([], dtype=int),
            n_clusters=0,
            n_noise=0,
            probabilities=np.array([]),
        )

    Z = zscore_normalize(signatures)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    labels = clusterer.fit_predict(Z)
    probabilities = clusterer.probabilities_

    unique_non_noise = set(labels[labels != -1].tolist())
    n_clusters = len(unique_non_noise)
    n_noise = int((labels == -1).sum())

    return ClusteringResult(
        labels=labels,
        n_clusters=n_clusters,
        n_noise=n_noise,
        probabilities=probabilities,
    )