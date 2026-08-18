"""Canonical regime ordering for Kim-style HSSM fits.

This module resolves the label-switching ambiguity that commonly appears in
Markov-switching models by imposing a deterministic ordering over latent
regimes. The canonical rule is:

- Sort regimes by mean behavioral activity level, ascending.
- Reorder transition probabilities, emission means, covariance matrices,
  duration parameters, and any stored regime metadata to match the new order.

This ordering is intentionally a post-hoc fix applied after fitting, and it is
meant to be stable once chosen.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from backbone.hssm.model import KimHSSMModel


def canonicalize_regime_order(model: KimHSSMModel) -> KimHSSMModel:
    """Sort the model's regimes by mean behavioral activity level ascending.

    The activity level for a regime is defined as the mean absolute value of its
    emission mean vector. This yields a deterministic ordering independent of the
    initialization-specific label names.
    """
    if not isinstance(model, KimHSSMModel):
        raise TypeError("Input must be a KimHSSMModel instance.")

    activity_scores = np.mean(np.abs(model.emission_means), axis=1)
    sorted_indices = np.argsort(activity_scores)

    reordered_means = model.emission_means[sorted_indices].copy()
    reordered_covariances = [np.asarray(model.emission_covariances[idx], dtype=float) for idx in sorted_indices]
    reordered_transition = model.transition_matrix[np.ix_(sorted_indices, sorted_indices)].copy()
    reordered_duration_mu = model.duration_mu[sorted_indices].copy()
    reordered_duration_sigma = model.duration_sigma[sorted_indices].copy()

    fixed_model = KimHSSMModel(
        n_regimes=model.n_regimes,
        n_features=model.n_features,
        transition_matrix=reordered_transition,
        emission_means=reordered_means,
        emission_covariances=reordered_covariances,
        duration_mu=reordered_duration_mu,
        duration_sigma=reordered_duration_sigma,
        duration_prior=model.duration_prior,
        metadata=dict(model.metadata),
    )
    fixed_model.metadata["regime_ordering"] = "ascending_activity"
    return fixed_model
