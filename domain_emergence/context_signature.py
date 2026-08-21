"""
Day 16 -- Context-signature extraction.

Converts a session-level HSSM regime_sequence + observations into
episode-level "context signatures": one feature vector per contiguous
dwell episode (a run of the same regime). These signatures are what
HDBSCAN clusters into emergent life domains (Day 16 continued / Day 17).

An episode is a maximal contiguous run of one regime in regime_sequence.
Missing sessions (NaN observation rows) inside an episode are excluded
from the mean/std computation, never imputed -- consistent with how
missingness is handled everywhere else in the HSSM pipeline (emission
loglik contributes 0, not an imputed value).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class Episode:
    regime_id: int
    start: int          # inclusive, session index
    end: int             # exclusive, session index
    duration: int         # end - start, session-index units

    @property
    def n_present(self) -> int:
        return self.duration


def extract_episodes(regime_sequence: np.ndarray) -> list[Episode]:
    """Split a regime_sequence into maximal contiguous same-regime runs."""
    if len(regime_sequence) == 0:
        return []

    episodes = []
    start = 0
    current = regime_sequence[0]
    for t in range(1, len(regime_sequence)):
        if regime_sequence[t] != current:
            episodes.append(Episode(regime_id=int(current), start=start, end=t, duration=t - start))
            start = t
            current = regime_sequence[t]
    episodes.append(Episode(regime_id=int(current), start=start, end=len(regime_sequence),
                             duration=len(regime_sequence) - start))
    return episodes


def build_context_signatures(
    observations: np.ndarray,
    episodes: list[Episode],
    min_present_sessions: int = 1,
) -> tuple[np.ndarray, list[Episode]]:
    """Build one feature vector per episode: [mean_obs (F,), std_obs (F,),
    log(duration)]. Episodes with fewer than min_present_sessions non-NaN
    rows are dropped (mean/std undefined otherwise) -- returns the filtered
    episode list alongside the matching feature matrix so callers can trace
    signatures back to their source episode.

    NaN rows within an episode are excluded from mean/std, never imputed.
    """
    F = observations.shape[1]
    signatures = []
    kept_episodes = []

    for ep in episodes:
        window = observations[ep.start:ep.end]
        present_mask = ~np.isnan(window).any(axis=1)
        present = window[present_mask]

        if present.shape[0] < min_present_sessions:
            continue

        mean_vec = present.mean(axis=0)
        std_vec = present.std(axis=0) if present.shape[0] > 1 else np.zeros(F)
        log_duration = np.log(ep.duration)

        sig = np.concatenate([mean_vec, std_vec, [log_duration]])
        signatures.append(sig)
        kept_episodes.append(ep)

    if not signatures:
        return np.empty((0, 2 * F + 1)), []

    return np.vstack(signatures), kept_episodes