"""Tests for the VIF + PCA feature reduction pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backbone.shared.feature_reduction import reduce_features


def test_vif_removes_multicollinear_pair() -> None:
    """Highly correlated features should be pruned before PCA."""
    rng = np.random.default_rng(seed=7)

    base_signal = rng.normal(size=200)
    pair_a = base_signal + rng.normal(scale=0.02, size=200)
    pair_b = base_signal + rng.normal(scale=0.02, size=200)
    other_features = rng.normal(size=(200, 3))

    frame = pd.DataFrame(
        {
            "feature_1": pair_a,
            "feature_2": pair_b,
            "feature_3": other_features[:, 0],
            "feature_4": other_features[:, 1],
            "feature_5": other_features[:, 2],
        }
    )

    result = reduce_features(frame, target_dim=2, vif_threshold=10.0)

    assert result.reduced_matrix.shape[1] <= 2
    assert result.reduced_matrix.shape[0] == len(frame)
    assert len(set(result.retained_features) & {"feature_1", "feature_2"}) < 2
    assert {"feature_1", "feature_2"} & set(result.dropped_features)
    assert list(result.loadings.columns) == [f"PC{idx + 1}" for idx in range(result.loadings.shape[1])]
    assert result.report


def test_missing_values_are_rejected() -> None:
    """Missing values should not be silently filled or imputed."""
    frame = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0],
        "x2": [2.0, np.nan, 4.0],
        "x3": [3.0, 4.0, 5.0],
    })

    try:
        reduce_features(frame, target_dim=2)
    except ValueError:
        return

    raise AssertionError("Missing values should raise a ValueError instead of being imputed.")
