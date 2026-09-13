"""
upstream_interfaces.py

CONTRACT FILE — defines the exact data shapes Sprint 8/9/10 code expects
from upstream sprints. No real fitting logic lives here.

When wiring into the real chronis-ml repo:
  - Delete this file.
  - Import the real HSSM/Attractor/Domain types directly.
  - Everything downstream only depends on the shapes below — the swap is
    a pure import change.

Sprint 3 (backbone) structure confirmed from:
  backbone/hssm/model.py         — GaussianHSMM / KimHSSMModel
  backbone/hssm/fitting.py       — fit_with_random_restarts / fit_hssm_model
  backbone/hssm/label_switching.py — canonicalize_labels (regime 0 = slowest)
  backbone/hssm/config.py        — HSSMFitConfig, ColdStartConfig
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence
import numpy as np


# ---------------------------------------------------------------------------
# Sprint 3 — HSSM fitted model contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HSSMFit:
    """
    The fitted HSSM model for one user (behavioral system), as produced by
    backbone.hssm.fitting.fit_with_random_restarts() after label canonicalization.

    After canonicalize_labels() (backbone/hssm/label_switching.py), regimes are
    ALWAYS sorted by ascending activity level:
        regime 0   =  lowest activity  =  SLOW regime  =  slow_regime_id = 0
        regime K-1 =  highest activity

    duration_parameters holds the log-normal dwell-time parameters the EM
    algorithm fitted for each regime.  The mean dwell time (D*) for any
    regime k is:
        E[dwell_k] = exp(dur_mu_k + dur_sigma_k^2 / 2)    [log-normal mean]

    These are the REAL fitted parameters from GaussianHSMM.dur_mu / .dur_sigma
    after EM convergence — not proxies derived from posteriors.
    """
    user_id: str
    fit_id: str                   # unique fit identifier (e.g. ISO timestamp + seed)
    slow_regime_id: int           # always 0 after canonicalize_labels — kept for explicitness
    n_regimes: int                # K in {2, 3, 4}  (BIC-selected)
    duration_parameters: dict     # {regime_id (int): {"dur_mu": float, "dur_sigma": float}}
                                  # dur_mu / dur_sigma are the log-space parameters of the
                                  # log-normal dwell-time distribution the HSSM learned.
    log_likelihood: float         # log-likelihood of the converged EM run
    converged: bool               # only converged runs are ever selected


def hssm_fit_from_backbone(model, user_id: str, fit_id: str) -> "HSSMFit":
    """
    Bridge: convert a fitted backbone GaussianHSMM (or KimHSSMModel) to HSSMFit.

    Call this AFTER fit_with_random_restarts() has applied canonicalize_labels(),
    so slow_regime_id is guaranteed to be 0.

    Example:
        from backbone.hssm.fitting import fit_with_random_restarts
        model, _ = fit_with_random_restarts(X, n_regimes=2, n_features=8,
                                             n_init=10, base_seed=0)
        fit = hssm_fit_from_backbone(model, user_id="u_001",
                                     fit_id="2026-08-23T00:00:00Z_s0")
    """
    assert model.dur_mu is not None and model.dur_sigma is not None, \
        "Model must be fitted before converting to HSSMFit."
    duration_parameters = {
        k: {
            "dur_mu":    float(model.dur_mu[k]),
            "dur_sigma": float(model.dur_sigma[k]),
        }
        for k in range(model.K)
    }
    return HSSMFit(
        user_id=user_id,
        fit_id=fit_id,
        slow_regime_id=0,          # canonical: always 0 after canonicalize_labels
        n_regimes=int(model.K),
        duration_parameters=duration_parameters,
        log_likelihood=float(model.log_likelihood_),
        converged=bool(model.converged_),
    )


# ---------------------------------------------------------------------------
# Sprint 3 — Per-timestep observation contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeObservation:
    """
    A single-timestep observation record from the HSSM forward-backward pass.

    m_t and the regime assignment come from the same fitted model — they must
    not be mixed across different fits or users.

    NOTE: D* is NOT stored here. D* is a property of the fitted model
    (HSSMFit.duration_parameters), not a per-observation quantity.
    """
    user_id: str
    timestamp: datetime
    m_t: np.ndarray                # shape (F,)  — fast latent state (PCA-reduced)
    regime_label: int              # MAP regime assignment (0-indexed, ascending activity)
    regime_posterior: np.ndarray   # shape (K,) — forward-backward marginal P(state_t=k | all data)


# ---------------------------------------------------------------------------
# Sprint 3 — RegimeSeries  (kept for Sprint 7 NSSM compatibility)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeSeries:
    """
    A fitted slow-regime process for one user, one system (behavioral or narrative).

    Mirrors Sprint 3 Day 7-9 (HSSM: p_t) and Sprint 7 Day 20 (NSSM: q_t) output shape.

    For Cold Start (Sprint 10), use HSSMFit + RegimeObservation instead of this
    type — they carry the actual fitted duration parameters needed for D*.
    """
    user_id: str
    system: str                        # "behavioral" | "narrative"
    session_ids: Sequence[str]         # session index -> session id, length T
    timestamps: Sequence[datetime]     # length T, NTP-aligned
    regime_labels: np.ndarray          # shape (T,), dtype int, canonical-sorted regime id per session
    n_regimes: int                     # K (behavioral) or J (narrative)
    fast_state: np.ndarray             # shape (T, d) — m_t (behavioral) or n_t (narrative)
    in_fit_set: np.ndarray             # shape (T,), dtype bool
    gated: bool                        # True if user never cleared cold-start gate (< 30 sessions)


# ---------------------------------------------------------------------------
# Sprint 4 — Attractor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttractorRecord:
    """Mirrors Sprint 4 output: a declared attractor for one user/regime/context."""
    user_id: str
    regime_id: int
    context_key: str
    revisit_count: int
    mean_dwell_time: float
    transition_stability: float
    declared: bool  # True only if all 3 stats cleared person-calibrated N/T (hard AND)


# ---------------------------------------------------------------------------
# Sprint 6 — Domain
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Domain:
    """Mirrors Sprint 6 output: a behaviorally+narratively aligned domain."""
    domain_id: str
    user_id: str
    label: str
    behavioral_regime_ids: Sequence[int]
    narrative_regime_ids: Sequence[int]
    confidence: float
    active: bool
    high_ignorance_prior: bool
    aspirational_or_hypothetical: bool


# ---------------------------------------------------------------------------
# Sprint 9 — Evidence types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionExcerpt:
    """A retrievable, citable unit of evidence — a transcript/session slice."""
    session_id: str
    user_id: str
    timestamp: datetime
    text: str
    contribution_score: float
    is_near_miss: bool = False


@dataclass(frozen=True)
class SelfReflectionSessionFlag:
    """Mirrors Sprint-9-adjacent tagging for self-reflection-mode sessions."""
    session_id: str
    user_id: str
    timestamp: datetime
    is_self_reflection_mode: bool
