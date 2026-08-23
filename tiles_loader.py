# e2e/tiles_loader.py — Sprint 14 Day 42.
#
# HONEST SCOPE NOTE, read before trusting this file's name:
# The real TILES-2018 dataset (Sprint 1 Day 2's actual surrogate corpus)
# was never uploaded to this workspace — only Sprint 3-13's ALGORITHM code
# was. This loader does NOT read TILES-2018. It calls Sprint 3-4's own
# `generate_surrogate_user` (backbone/shared/synthetic_data.py), which is
# real code already used by that team's own test suite to validate the
# HSSM fitter against a semi-Markov, log-normal-dwell synthetic process —
# legitimate surrogate data per the Bible's own doctrine (Part 13.1), just
# not the specific TILES-2018 corpus the directive names.
#
# Whoever owns Sprint 1 should point this loader at a real TILES-2018
# ingest path before this pipeline's timing numbers are used to justify
# the directive's "<20 minutes per day of TILES-2018 data" claim — a
# timing result against synthetic data is a timing result, not evidence
# about the real dataset.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backbone.shared.synthetic_data import generate_surrogate_user


@dataclass(frozen=True)
class SurrogateDayOfData:
    """One user's one day-of-data equivalent, shaped the way the e2e
    runner's downstream stages expect. `n_sessions` here stands in for
    "sessions in one day" — Sprint 1's real ingest would produce this from
    actual session boundaries; we don't have that logic, so we just treat
    the whole generated trajectory as one timing unit."""

    user_id: str
    X: np.ndarray            # (T, F) — NaN-free, "aligned feature matrix"
    true_regimes: np.ndarray  # ground truth, NOT fed into fitting — eval only
    n_sessions: int


def load_surrogate_day(user_id: str, *, n_sessions: int = 40, n_features: int = 6,
                        n_regimes: int = 2, seed: int | None = None) -> SurrogateDayOfData:
    """
    Stand-in for Sprint 1/2's real ingest -> decrypt-in-RAM -> transcribe
    -> align pipeline. Those stages' code was not present in the uploaded
    zips (only Sprint 3 onward was) — this function starts the e2e demo
    one stage later than the directive's own pipeline diagram, at the
    point where an aligned feature matrix already exists.
    """
    data = generate_surrogate_user(
        n_sessions=n_sessions, n_features=n_features, n_regimes=n_regimes, seed=seed,
    )
    return SurrogateDayOfData(
        user_id=user_id,
        X=data["X_complete"],  # NaN-free variant; real NaN-handling (Sprint 1 Day 2's
                                # "never impute, never zero" rule) is explicitly out of
                                # scope here — we start post-imputation-policy.
        true_regimes=data["true_regimes"],
        n_sessions=n_sessions,
    )