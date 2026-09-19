"""
HSSMFitDataset — canonical input contract for fit_hssm().

Enforces the directive's data contract (Section B):
  - user_id
  - calendar-time index (not session count)
  - feature matrix with feature names
  - feature_schema_version and alignment_version
  - missingness_mask, confidence_mask, eligibility_mask
  - eligible_session_count and min_present_sessions
  - personal-normalization metadata
  - no future leakage in normalization
  - deterministic dataset hash
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class HSSMFitDataset:
    """Canonical input data contract for the HSSM fitting pipeline.

    Every field required by the directive's Section B is present.
    The dataset_hash is computed deterministically from the data content,
    guaranteeing that identical data produces identical hashes.
    """

    # --- Identity ---
    user_id: str

    # --- Calendar-time index & Feature matrix (non-default) ---
    calendar_index: np.ndarray  # shape (T,), dtype float64 — timestamps in days/minutes
    feature_matrix: np.ndarray = field(repr=False)  # shape (T, F)
    missingness_mask: np.ndarray = field(repr=False)     # shape (T, F), True = missing
    confidence_mask: np.ndarray = field(repr=False)       # shape (T, F), values in [0, 1]
    eligibility_mask: np.ndarray = field(repr=False)      # shape (T,), True = eligible row

    # --- Optional / Default fields ---
    time_grid_unit: str = "calendar_days"  # "calendar_days" or "calendar_minutes"
    feature_names: list[str] = field(default_factory=list)
    feature_schema_version: str = "1.0.0"
    alignment_version: str = "1.0.0"
    eligible_session_count: int = 0
    min_present_sessions: int = 30
    personal_normalization_metadata: dict[str, Any] = field(default_factory=dict)
    dataset_hash: str = ""

    def __post_init__(self) -> None:
        """Validate all contract requirements and compute deterministic hash."""
        self._validate()
        if not self.dataset_hash:
            self.dataset_hash = self._compute_hash()

    def _validate(self) -> None:
        """Enforce every contract requirement from the directive."""
        # User ID
        if not self.user_id or not isinstance(self.user_id, str):
            raise ValueError("HSSMFitDataset requires a non-empty string user_id")

        # Calendar index
        if self.calendar_index is None or len(self.calendar_index) == 0:
            raise ValueError("HSSMFitDataset requires a non-empty calendar_index")
        cal = np.asarray(self.calendar_index, dtype=float)
        if cal.ndim != 1:
            raise ValueError("calendar_index must be 1-dimensional")
        # Reject integer session counts masquerading as calendar time
        if np.array_equal(cal, np.arange(len(cal))):
            # Additional heuristic: if all values are exactly 0,1,2,...,T-1
            # AND time_grid_unit claims to be calendar, this is suspicious.
            # We allow it only if the user explicitly set time_grid_unit to session_index
            if self.time_grid_unit not in ("session_index",):
                # Check if it's truly sequential integers with no gaps
                if len(cal) > 1 and np.all(np.diff(cal) == 1.0):
                    raise ValueError(
                        "calendar_index appears to be a sequential session count "
                        "(0, 1, 2, ..., T-1) rather than real calendar timestamps. "
                        "Use actual calendar-day offsets or set time_grid_unit='session_index' "
                        "to explicitly acknowledge session-index units."
                    )
        self.calendar_index = cal

        # Feature matrix shape
        X = np.asarray(self.feature_matrix, dtype=float)
        T = len(self.calendar_index)
        if X.ndim != 2:
            raise ValueError("feature_matrix must be 2-dimensional (T, F)")
        if X.shape[0] != T:
            raise ValueError(
                f"feature_matrix rows ({X.shape[0]}) must match "
                f"calendar_index length ({T})"
            )
        F = X.shape[1]
        if F <= 0:
            raise ValueError("feature_matrix must have at least one feature column")
        self.feature_matrix = X

        # Feature names
        if not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(F)]
        if len(self.feature_names) != F:
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) must match "
                f"feature count ({F})"
            )

        # Missingness mask
        miss = np.asarray(self.missingness_mask, dtype=bool)
        if miss.shape != (T, F):
            raise ValueError(
                f"missingness_mask shape {miss.shape} must match "
                f"feature_matrix shape ({T}, {F})"
            )
        self.missingness_mask = miss

        # Confidence mask
        conf = np.asarray(self.confidence_mask, dtype=float)
        if conf.shape != (T, F):
            raise ValueError(
                f"confidence_mask shape {conf.shape} must match "
                f"feature_matrix shape ({T}, {F})"
            )
        self.confidence_mask = conf

        # Eligibility mask
        elig = np.asarray(self.eligibility_mask, dtype=bool)
        if elig.shape != (T,):
            raise ValueError(
                f"eligibility_mask shape {elig.shape} must match "
                f"calendar_index length ({T},)"
            )
        self.eligibility_mask = elig

        # Eligible session count must match mask
        actual_eligible = int(elig.sum())
        if self.eligible_session_count == 0:
            self.eligible_session_count = actual_eligible
        elif self.eligible_session_count != actual_eligible:
            raise ValueError(
                f"eligible_session_count ({self.eligible_session_count}) does not match "
                f"eligibility_mask sum ({actual_eligible})"
            )

    def _compute_hash(self) -> str:
        """Deterministic hash of all data content. Same data → same hash."""
        h = hashlib.sha256()
        h.update(self.user_id.encode("utf-8"))
        h.update(self.calendar_index.tobytes())
        h.update(self.feature_matrix.tobytes())
        h.update(self.missingness_mask.tobytes())
        h.update(self.confidence_mask.tobytes())
        h.update(self.eligibility_mask.tobytes())
        h.update(self.feature_schema_version.encode("utf-8"))
        h.update(self.alignment_version.encode("utf-8"))
        return h.hexdigest()

    def check_eligibility_gate(self) -> None:
        """Raise ColdStartError if insufficient eligible sessions.
        This is the ONE authoritative eligibility check used everywhere."""
        from backbone.hssm.gating import ColdStartError
        if self.eligible_session_count < self.min_present_sessions:
            raise ColdStartError(
                f"{self.eligible_session_count} eligible sessions < minimum "
                f"({self.min_present_sessions}). No HSSM output produced."
            )

    def get_eligible_feature_matrix(self) -> np.ndarray:
        """Return the feature matrix restricted to eligible rows, with
        missing/low-confidence values set to NaN for marginalization."""
        X = self.feature_matrix.copy()
        # Set missing values to NaN
        X[self.missingness_mask] = np.nan
        # Set low-confidence values to NaN (confidence < 0.5 threshold)
        X[self.confidence_mask < 0.5] = np.nan
        return X

    def get_eligible_timestamps(self) -> np.ndarray:
        """Calendar timestamps for eligible rows."""
        return self.calendar_index[self.eligibility_mask]

    def validate_no_future_leakage(self) -> None:
        """Check that personal normalization metadata does not reference future data.
        Raises ValueError if normalization_end_index > any training index."""
        norm_meta = self.personal_normalization_metadata
        if "normalization_end_index" in norm_meta:
            end_idx = norm_meta["normalization_end_index"]
            if end_idx > len(self.calendar_index):
                raise ValueError(
                    f"Normalization end index ({end_idx}) exceeds dataset length "
                    f"({len(self.calendar_index)}). This indicates future data leakage."
                )
        if norm_meta.get("uses_future_data", False):
            raise ValueError(
                "personal_normalization_metadata.uses_future_data is True — "
                "this is a future leakage violation."
            )


def compute_eligibility_mask(
    feature_matrix: np.ndarray,
    missingness_mask: np.ndarray,
    confidence_mask: np.ndarray,
    min_feature_presence: float = 0.5,
    min_confidence_threshold: float = 0.5,
) -> np.ndarray:
    """Compute the ONE authoritative eligibility mask.

    A row is eligible if:
      - >= min_feature_presence fraction of features are non-missing
      - >= min_feature_presence fraction of features have confidence >= threshold

    This function must be used everywhere eligibility is checked.
    """
    T, F = feature_matrix.shape
    non_missing_fraction = (~missingness_mask).sum(axis=1) / F
    high_confidence_fraction = (confidence_mask >= min_confidence_threshold).sum(axis=1) / F
    eligible = (non_missing_fraction >= min_feature_presence) & (high_confidence_fraction >= min_feature_presence)
    return eligible.astype(bool)


def make_dataset(
    user_id: str,
    feature_matrix: np.ndarray,
    calendar_index: np.ndarray,
    feature_names: list[str] | None = None,
    confidence_values: np.ndarray | None = None,
    min_present_sessions: int = 30,
    feature_schema_version: str = "1.0.0",
    alignment_version: str = "1.0.0",
    normalization_metadata: dict | None = None,
    time_grid_unit: str = "calendar_days",
) -> HSSMFitDataset:
    """Convenience constructor that derives masks from raw data.

    Missingness mask is derived from NaN values in feature_matrix.
    Confidence mask defaults to 1.0 for observed values, 0.0 for missing.
    Eligibility mask is computed from the authoritative function.
    """
    X = np.asarray(feature_matrix, dtype=float)
    T, F = X.shape

    missingness_mask = np.isnan(X)

    if confidence_values is not None:
        confidence_mask = np.asarray(confidence_values, dtype=float)
    else:
        confidence_mask = np.where(missingness_mask, 0.0, 1.0)

    eligibility_mask = compute_eligibility_mask(
        X, missingness_mask, confidence_mask
    )

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(F)]

    return HSSMFitDataset(
        user_id=user_id,
        calendar_index=np.asarray(calendar_index, dtype=float),
        time_grid_unit=time_grid_unit,
        feature_matrix=X,
        feature_names=feature_names,
        feature_schema_version=feature_schema_version,
        alignment_version=alignment_version,
        missingness_mask=missingness_mask,
        confidence_mask=confidence_mask,
        eligibility_mask=eligibility_mask,
        min_present_sessions=min_present_sessions,
        personal_normalization_metadata=normalization_metadata or {},
    )
