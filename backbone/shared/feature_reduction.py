"""Feature reduction for BACKBONE HSSM inputs.

This module implements a two-stage pipeline suitable for preparing a per-user
feature matrix for the fast latent state m_t in the Kim (1994) HSSM:

1. Iterative VIF filtering removes multicollinear features.
2. PCA reduces the remaining feature set to a configurable target dimension.

The reducer intentionally raises on missing values instead of imputing them, so
that missingness is treated as a modelling boundary condition rather than being
silently filled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA



@dataclass(frozen=True)
class FeatureReductionResult:
    """Container for the reduced feature matrix and its PCA loading diagnostics."""

    reduced_matrix: pd.DataFrame
    loadings: pd.DataFrame
    retained_features: List[str]
    dropped_features: List[str]
    report: Dict[str, List[Tuple[str, float]]]

    @property
    def n_components(self) -> int:
        """Number of retained latent dimensions."""
        return int(self.reduced_matrix.shape[1])


def _validate_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the input matrix and return a strictly numeric copy."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("Input feature matrix is empty.")
    if frame.shape[1] == 0:
        raise ValueError("Input feature matrix must contain at least one feature column.")

    if frame.isnull().values.any():
        raise ValueError(
            "Input feature matrix contains NaN or pandas-missing values; missing values "
            "must be handled upstream and are not imputed here."
        )

    numeric_frame = frame.select_dtypes(include=[np.number, bool]).copy()
    if numeric_frame.shape[1] != frame.shape[1]:
        offending = [
            column
            for column in frame.columns
            if not pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column])
        ]
        raise TypeError(
            "All feature columns must be numeric or boolean; non-numeric columns found: "
            f"{offending}."
        )

    return numeric_frame.astype(float, copy=False)


def _compute_vif(column: pd.Series, other_columns: pd.DataFrame) -> float:
    """Compute the variance inflation factor for a single feature."""
    y = column.to_numpy(dtype=float)
    if other_columns.empty:
        variance = float(np.var(y, ddof=0))
        return np.inf if np.isclose(variance, 0.0) else 0.0

    x_design = np.column_stack([np.ones(len(y), dtype=float), other_columns.to_numpy(dtype=float)])
    coefficients, _, _, _ = np.linalg.lstsq(x_design, y, rcond=None)
    fitted = x_design @ coefficients

    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    ss_residual = float(np.sum((y - fitted) ** 2))
    if np.isclose(ss_total, 0.0):
        return np.inf

    r_squared = 1.0 - (ss_residual / ss_total)
    r_squared = float(np.clip(r_squared, 0.0, 0.999999999))
    return 1.0 / (1.0 - r_squared)


def _iterative_vif_filter(frame: pd.DataFrame, vif_threshold: float = 10.0) -> Tuple[List[str], List[str]]:
    """Drop the highest-VIF feature until no feature exceeds the threshold."""
    if vif_threshold <= 0:
        raise ValueError("The VIF threshold must be positive.")

    remaining = list(frame.columns)
    dropped: List[str] = []

    while len(remaining) > 1:
        vif_values = {
            feature: _compute_vif(frame[feature], frame.drop(columns=[feature]))
            for feature in remaining
        }
        worst_feature = max(remaining, key=lambda feature: vif_values[feature])
        worst_vif = vif_values[worst_feature]

        if worst_vif > vif_threshold:
            dropped.append(worst_feature)
            remaining.remove(worst_feature)
            continue
        break

    return remaining, dropped


def _pca_loadings(raw_matrix: np.ndarray, target_dim: int) -> Tuple[np.ndarray, pd.DataFrame]:
    """Return PCA scores and a loading matrix for the retained features."""
    centered = raw_matrix - raw_matrix.mean(axis=0, keepdims=True)
    std = raw_matrix.std(axis=0, ddof=0)
    std[std == 0.0] = 1.0
    standardized = centered / std

    u, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    n_components = min(target_dim, standardized.shape[1], singular_values.shape[0])
    if n_components <= 0:
        raise ValueError("No principal components could be extracted from the input matrix.")

    component_matrix = vt[:n_components, :]
    scores = standardized @ component_matrix.T
    loadings = pd.DataFrame(
        component_matrix.T,
        index=[f"feature_{idx}" for idx in range(component_matrix.shape[1])],
        columns=[f"PC{idx + 1}" for idx in range(component_matrix.shape[0])],
    )

    return scores[:, :n_components], loadings


def _build_loading_report(loadings: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
    """Summarize which raw features load most strongly on each retained component."""
    report: Dict[str, List[Tuple[str, float]]] = {}
    for component_name in loadings.columns:
        ranked = loadings[component_name].abs().sort_values(ascending=False)
        report[component_name] = [
            (str(feature_name), float(abs_loading))
            for feature_name, abs_loading in ranked.head(5).items()
        ]
    return report


def reduce_features(
    frame: pd.DataFrame,
    target_dim: int = 10,
    vif_threshold: float = 10.0,
) -> FeatureReductionResult:
    """Reduce a raw feature matrix via iterative VIF filtering and PCA.

    Args:
        frame: T x F feature matrix. Each column is a raw feature and each row a
            timestamped observation/session vector.
        target_dim: Target number of retained latent components. Values are
            clipped to the available rank when the feature count is smaller than
            the requested dimension.
        vif_threshold: VIF cutoff for dropping multicollinear features.

    Returns:
        A FeatureReductionResult containing the reduced m_t-ready matrix and
        the component loading diagnostics used for documentation.

    Raises:
        TypeError: If the input is not a DataFrame or contains non-numeric data.
        ValueError: If the input is empty, contains missing values, or has an
            invalid target dimensionality.
    """
    numeric_frame = _validate_numeric_frame(frame)

    if target_dim < 1:
        raise ValueError("target_dim must be at least 1.")

    retained_features, dropped_features = _iterative_vif_filter(numeric_frame, vif_threshold=vif_threshold)

    if not retained_features:
        raise ValueError("All features were removed during VIF filtering; no matrix remains for PCA.")

    reduced_input = numeric_frame[retained_features].copy()
    effective_dim = min(target_dim, reduced_input.shape[1], reduced_input.shape[0])
    if effective_dim < 1:
        raise ValueError("The reduced feature matrix is empty or degenerate after VIF filtering.")

    scores, loadings = _pca_loadings(reduced_input.to_numpy(dtype=float), effective_dim)
    loadings.index = retained_features
    loadings.columns = [f"PC{idx + 1}" for idx in range(loadings.shape[1])]

    reduced_matrix = pd.DataFrame(
        scores,
        index=numeric_frame.index,
        columns=[f"m_t_{idx + 1}" for idx in range(scores.shape[1])],
    )

    report = _build_loading_report(loadings)
    return FeatureReductionResult(
        reduced_matrix=reduced_matrix,
        loadings=loadings,
        retained_features=retained_features,
        dropped_features=dropped_features,
        report=report,
    )


# ---------- Teammate's Functions supporting Missing Sessions ----------

def per_person_zscore(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each feature column using THIS person's own mean/std only.
    NaN rows (missing sessions) are ignored for mean/std and remain NaN in output
    (marginalized later by the HSSM's gating/emission logic, never imputed here)."""
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std == 0] = 1.0
    Z = (X - mean) / std
    return Z, mean, std


def _compute_vif_numpy(X: np.ndarray) -> np.ndarray:
    n_feat = X.shape[1]
    vifs = np.zeros(n_feat)
    corr = np.corrcoef(X, rowvar=False)
    try:
        inv_corr = np.linalg.inv(corr)
        vifs = np.diag(inv_corr)
    except np.linalg.LinAlgError:
        for i in range(n_feat):
            others = [j for j in range(n_feat) if j != i]
            r2 = max(corr[i, j] ** 2 for j in others) if others else 0.0
            vifs[i] = 1.0 / max(1e-6, (1 - r2))
    return vifs


def remove_high_vif(X: np.ndarray, feature_names: list[str], vif_threshold: float = 10.0):
    """Iteratively drop the feature with highest VIF until all remaining are below
    threshold, or we hit the floor of 8 features (target minimum dims)."""
    df = pd.DataFrame(X, columns=feature_names).dropna()
    if len(df) < len(feature_names) + 2:
        raise ValueError(
            f"Not enough complete-case rows ({len(df)}) to compute VIF for "
            f"{len(feature_names)} features."
        )

    remaining = list(feature_names)
    dropped = []
    while len(remaining) > 8:
        vifs = _compute_vif_numpy(df[remaining].values)
        max_vif_idx = int(np.argmax(vifs))
        max_vif = vifs[max_vif_idx]
        if max_vif < vif_threshold:
            break
        dropped_feat = remaining.pop(max_vif_idx)
        dropped.append((dropped_feat, max_vif))
    return remaining, dropped


def reduce_dimensionality(
    Z: np.ndarray,
    feature_names: list[str],
    target_dims: int = 12,
    min_dims: int = 8,
    max_dims: int = 15,
):
    """PCA reduction to target_dims (clamped to [min_dims, max_dims]).
    Fit on complete-case rows; rows with any raw NaN remain NaN in reduced space
    (reduced-space missingness is what the HSSM gating marginalizes, not raw-space)."""
    target_dims = int(np.clip(target_dims, min_dims, min(max_dims, Z.shape[1])))

    complete_mask = ~np.isnan(Z).any(axis=1)
    if complete_mask.sum() < target_dims + 2:
        raise ValueError("Not enough complete-case rows to fit PCA reliably.")

    pca = PCA(n_components=target_dims, random_state=0)
    pca.fit(Z[complete_mask])

    Z_reduced = np.full((Z.shape[0], target_dims), np.nan)
    V = pca.components_.T  # Shape (F, target_dims)

    for i in range(Z.shape[0]):
        obs_idx = ~np.isnan(Z[i])
        if not np.any(obs_idx):
            continue  # Entirely missing row remains NaN
        
        z_obs = Z[i, obs_idx]
        V_obs = V[obs_idx, :]  # Loadings restricted to observed dimensions (|O_i|, target_dims)
        
        if len(z_obs) == Z.shape[1]:
            # All features present: standard exact PCA transform
            Z_reduced[i] = z_obs @ V
        else:
            # Partial missingness: exact least-squares projection using observed loadings V_obs
            # Solves min || z_obs - score @ V_obs.T ||^2 without ANY zero/mean imputation
            score, _, _, _ = np.linalg.lstsq(V_obs, z_obs, rcond=None)
            Z_reduced[i] = score

    loadings = pd.DataFrame(
        pca.components_.T, index=feature_names,
        columns=[f"PC{i+1}" for i in range(target_dims)],
    )
    top_loadings = {
        col: loadings[col].abs().sort_values(ascending=False).index[:5].tolist()
        for col in loadings.columns
    }

    report = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": float(np.sum(pca.explained_variance_ratio_)),
        "top5_loading_features_per_component": top_loadings,
        "n_components": target_dims,
        "caveat": (
            "Components are PCA directions, not interpretable psychological axes. "
            "Loadings show which raw features contribute most to each direction."
        ),
    }
    return Z_reduced, pca, report

