"""
Sprint 3, Day 8 — EM fitting harness + K selection via BIC.

Spec:
  - hard minimum of 10 random initializations per fit
  - select the CONVERGED run with the highest log-likelihood, never "best-looking"
  - K selection by BIC ONLY (K in {2,3,4})

Label-switching canonicalization lives in label_switching.py (separated per
Palash review) and is applied here as the final step before returning.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Sequence, Tuple, cast

from backbone.hssm.model import GaussianHSMM, KimHSSMModel, NotFittedError, FittingConvergenceError, InternalStateError
from backbone.hssm.label_switching import canonicalize_labels
from backbone.hssm.config import DEFAULT_FIT_CONFIG, HSSMFitConfig


def fit_with_random_restarts(
    X: np.ndarray,
    n_regimes: int,
    n_features: int,
    n_init: int = DEFAULT_FIT_CONFIG.n_init,
    n_iter: int = DEFAULT_FIT_CONFIG.n_iter,
    max_duration: int = DEFAULT_FIT_CONFIG.max_duration,
    base_seed: int = 0,
    verbose: bool = False,
    bypass_init_gate: bool = False,
    timestamps: np.ndarray | None = None,
) -> tuple[GaussianHSMM, list[dict]]:
    """Fit with >= n_init random initializations, return the converged run
    with highest log-likelihood, canonicalized via the MP-02 label-switching
    fix. run_log records every attempt for auditability."""
    if not bypass_init_gate:
        if n_init < 10:
            raise ValueError("Directive Day 8 requires a HARD MINIMUM of 10 random initializations.")
    else:
        if n_init < 1:
            raise ValueError("Must run at least 1 random initialization.")

    run_log = []
    best_model: GaussianHSMM | None = None
    best_ll = -np.inf

    for i in range(n_init):
        model = GaussianHSMM(n_regimes=n_regimes, n_features=n_features,
                              max_duration=max_duration, seed=base_seed + i)
        if timestamps is not None:
            model.fit(X, n_iter=n_iter, verbose=False, timestamps=timestamps)
        else:
            model.fit(X, n_iter=n_iter, verbose=False)

        run_log.append({
            "init": i,
            "seed": base_seed + i,
            "converged": bool(model.converged_),
            "n_iter": model.n_iter_,
            "log_likelihood": float(model.log_likelihood_) if model.log_likelihood_ is not None else None,
            "monotonic_ll": model.is_log_likelihood_monotonic(),
        })
        if verbose:
            ll_val = model.log_likelihood_ if model.log_likelihood_ is not None else 0.0
            print(f"  init {i}: converged={model.converged_}, ll={ll_val:.3f}")

        if model.converged_:
            ll = model.log_likelihood_
            if ll is None:
                raise InternalStateError("Log likelihood is None for converged model")
            if ll > best_ll:
                best_ll = ll
                best_model = model

    if best_model is None:
        raise FittingConvergenceError(
            f"None of {n_init} random-init EM runs converged. Real failure, "
            f"not something to paper over — check data quality, n_iter, or "
            f"max_duration before proceeding."
        )

    convergence_rate = sum(1.0 if r["converged"] else 0.0 for r in run_log) / len(run_log)
    if best_model is None:
        raise FittingConvergenceError("best_model is None prior to label canonicalization")
    best_model = cast(GaussianHSMM, canonicalize_labels(best_model))
    if best_model is None:
        raise FittingConvergenceError("Canonicalization returned None for best_model")
    best_model.convergence_rate_ = convergence_rate

    return best_model, run_log


def select_k_by_bic(
    X: np.ndarray,
    n_features: int,
    k_candidates: tuple[int, ...] = DEFAULT_FIT_CONFIG.k_candidates,
    n_init: int = DEFAULT_FIT_CONFIG.n_init,
    n_iter: int = DEFAULT_FIT_CONFIG.n_iter,
    max_duration: int = DEFAULT_FIT_CONFIG.max_duration,
    base_seed: int = 0,
    verbose: bool = False,
    bypass_init_gate: bool = False,
    timestamps: np.ndarray | None = None,
) -> tuple[GaussianHSMM, dict]:
    """Fit each candidate K with the full random-restart harness, select by
    BIC only — never because a K 'gives more interesting regimes'."""
    results = {}
    best_k = None
    best_bic = np.inf
    best_model: GaussianHSMM | None = None

    for k in k_candidates:
        if verbose:
            print(f"Fitting K={k}...")
        model, run_log = fit_with_random_restarts(
            X, n_regimes=k, n_features=n_features, n_init=n_init, n_iter=n_iter,
            max_duration=max_duration, base_seed=base_seed + k * 1000, verbose=False,
            bypass_init_gate=bypass_init_gate, timestamps=timestamps,
        )
        if model.log_likelihood_ is None:
            raise NotFittedError("Model log_likelihood_ is None")
        bic = model.bic(n_observations=(~np.isnan(X).any(axis=1)).sum())
        results[k] = {
            "bic": float(bic),
            "log_likelihood": float(model.log_likelihood_),
            "convergence_rate": model.convergence_rate_,
            "run_log": run_log,
        }
        if verbose:
            print(f"  K={k}: BIC={bic:.2f}, conv_rate={model.convergence_rate_:.2f}")

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_model = model

    if best_model is None:
        raise FittingConvergenceError("No models were fitted because k_candidates was empty or none converged.")
    selection_report = {
        "selected_k": best_k,
        "bic_by_k": {k: results[k]["bic"] for k in results},
        "loglik_by_k": {k: results[k]["log_likelihood"] for k in results},
        "convergence_rate_by_k": {k: results[k]["convergence_rate"] for k in results},
        "selection_rule": "min BIC across K candidates — never chosen for 'interesting regimes'",
    }
    return best_model, selection_report


def fit_hssm_model(
    observations: np.ndarray | pd.DataFrame,
    candidate_ks: Sequence[int] = (2, 3, 4),
    n_initializations: int = 10,
    max_iter: int = 200,
    tol: float = 1e-4,
    random_seed: Optional[int] = None,
    allow_fast_test_fit: bool = False,
) -> Tuple[KimHSSMModel, Dict[str, Any]]:
    """Legacy wrapper for backward compatibility with existing test suites.

    If allow_fast_test_fit is False (default), n_initializations < 10 will raise.
    Set to True only for fast unit testing.
    """
    if isinstance(observations, pd.DataFrame):
        X = observations.to_numpy(dtype=float)
    else:
        X = np.asarray(observations, dtype=float)

    n_features = X.shape[1]

    # Explicitly check for NaNs as expected by some legacy tests
    if np.isnan(X).any():
        raise ValueError("Observation matrix contains NaN values; missingness must be handled upstream.")

    model, report = select_k_by_bic(
        X,
        n_features=n_features,
        k_candidates=tuple(candidate_ks),
        n_init=n_initializations,
        n_iter=max_iter,
        base_seed=random_seed or 0,
        bypass_init_gate=allow_fast_test_fit,
    )

    compat_report = {
        "k_selected": report["selected_k"],
        "bic_by_k": report["bic_by_k"],
        "loglik_by_k": report["loglik_by_k"],
        "convergence_rate_by_k": report["convergence_rate_by_k"],
        "selected_loglik": model.log_likelihood_,
        "candidate_ks": tuple(candidate_ks),
        "n_initializations": n_initializations,
        "max_iter": max_iter,
        "tol": tol,
    }

    # Convert GaussianHSMM to KimHSSMModel for compatibility (using genuine EM-integrated durations)
    if model.var is None or model.A is None or model.mu is None or model.dur_mu is None or model.dur_sigma is None:
        raise NotFittedError("Fitted model has uninitialized parameters")

    covariances = [np.diag(v) for v in model.var]
    fitted_model = KimHSSMModel(
        n_regimes=model.K,
        n_features=model.F,
        transition_matrix=model.A,
        emission_means=model.mu,
        emission_covariances=covariances,
        duration_mu=model.dur_mu,
        duration_sigma=model.dur_sigma,
        duration_prior="lognormal",
        max_duration=model.Dmax,
        seed=random_seed,
    )
    fitted_model.log_likelihood_ = model.log_likelihood_
    fitted_model.log_likelihood_history_ = model.log_likelihood_history_
    fitted_model.converged_ = model.converged_
    fitted_model.n_iter_ = model.n_iter_

    return fitted_model, compat_report


def compute_dmax_tail_diagnostic(model: GaussianHSMM) -> dict[int, float]:
    """Ticket S34.4: Compute posterior mass at truncation boundary Dmax for each regime."""
    model._require_fitted()
    dur_logpmf = model._duration_logpmf()
    tail_mass = {}
    for k in range(model.K):
        tail_mass[k] = float(np.exp(dur_logpmf[k, -1]))
    return tail_mass


def sweep_dmax_sensitivity(
    X: np.ndarray,
    n_regimes: int,
    n_features: int,
    dmax_candidates: tuple[int, ...] = (30, 45, 60),
    n_init: int = 10,
    base_seed: int = 0,
) -> dict[int, dict[str, Any]]:
    """Ticket S34.4: Sweep across candidate Dmax values and report log-likelihood,
    BIC, and tail mass at Dmax for diagnostic inspection."""
    report = {}
    for dmax in dmax_candidates:
        model, _ = fit_with_random_restarts(
            X,
            n_regimes=n_regimes,
            n_features=n_features,
            n_init=n_init,
            max_duration=dmax,
            base_seed=base_seed,
        )
        tail_mass = compute_dmax_tail_diagnostic(model)
        report[dmax] = {
            "log_likelihood": float(model.log_likelihood_) if model.log_likelihood_ is not None else 0.0,
            "bic": float(model.bic(n_observations=(~np.isnan(X).any(axis=1)).sum())),
            "max_tail_mass": max(tail_mass.values()),
            "tail_mass_by_regime": tail_mass,
        }
    return report


@dataclass
class HSSMResult:
    """Canonical, backend-neutral result contract for fit_hssm per Ticket S34.7.
    
    Fields supported by current baseline (GaussianHSMM):
      - model: fitted estimator instance
      - p_t: MAP/posterior discrete regime trajectory
      - regime_posterior: T x K posterior probability matrix
      - duration_info: dict containing duration distribution parameters
      - duration_unit: "sessions" or "calendar_days"
      - selected_k: integer count of selected regimes (K)
      - model_class: class name string of the underlying backend
      - convergence_metadata: training/convergence audit log
      
    Fields for future behavioral backbones (e.g. RED-SDS) not supported by current baseline:
      - m_t: None (explicitly None per S34.3 gap — NOT fabricated)
      - m_t_uncertainty: None
    """
    model: Any
    p_t: np.ndarray | None
    regime_posterior: np.ndarray | None
    duration_info: dict[str, Any]
    duration_unit: str
    selected_k: int
    model_class: str
    convergence_metadata: dict[str, Any]
    m_t: np.ndarray | None = None
    m_t_uncertainty: Any = None

    # Backward-compatibility property aliases for legacy callers
    @property
    def k_selected(self) -> int:
        return self.selected_k

    @property
    def bic_by_k(self) -> dict[int, float]:
        return self.convergence_metadata.get("bic_by_k", {})

    @property
    def loglik_by_k(self) -> dict[int, float]:
        return self.convergence_metadata.get("loglik_by_k", {})

    @property
    def convergence_rate_by_k(self) -> dict[int, float]:
        return self.convergence_metadata.get("convergence_rate_by_k", {})

    @property
    def selected_loglik(self) -> float | None:
        return self.convergence_metadata.get("selected_loglik")


def fit_hssm(
    matrix: np.ndarray | pd.DataFrame,
    candidate_ks: Sequence[int] = (2, 3, 4),
    n_initializations: int = 10,
    max_iter: int = 200,
    random_seed: Optional[int] = None,
    allow_fast_test_fit: bool = False,
    timestamps: Optional[np.ndarray] = None,
) -> HSSMResult:
    """Canonical fitting entrypoint for backbone.hssm per Ticket S34.7 & R2-S56.1."""
    if isinstance(matrix, pd.DataFrame):
        X = matrix.to_numpy(dtype=float)
    else:
        X = np.asarray(matrix, dtype=float)

    n_features = X.shape[1]
    if n_features <= 0:
        raise ValueError("Feature count must be strictly greater than zero (F > 0)")

    model, report = select_k_by_bic(
        X,
        n_features=n_features,
        k_candidates=tuple(candidate_ks),
        n_init=n_initializations,
        n_iter=max_iter,
        base_seed=random_seed or 0,
        bypass_init_gate=allow_fast_test_fit,
        timestamps=timestamps,
    )

    if timestamps is not None:
        model.duration_unit = "calendar_days"

    # Compute posterior quantities for the selected model
    regime_post, _, _ = model._forward_backward(X)
    p_t = np.argmax(regime_post, axis=1)

    duration_info = {
        "duration_mu": model.duration_mu,
        "duration_sigma": model.duration_sigma,
        "max_duration": model.Dmax,
    }

    convergence_metadata = {
        "bic_by_k": report["bic_by_k"],
        "loglik_by_k": report["loglik_by_k"],
        "convergence_rate_by_k": report["convergence_rate_by_k"],
        "selected_loglik": model.log_likelihood_,
        "converged": model.converged_,
        "n_iter": model.n_iter_,
    }

    return HSSMResult(
        model=model,
        p_t=p_t,
        regime_posterior=regime_post,
        duration_info=duration_info,
        duration_unit=model.duration_unit,
        selected_k=report["selected_k"],
        model_class=model.__class__.__name__,
        convergence_metadata=convergence_metadata,
        m_t=None,  # Explicitly None for baseline GaussianHSMM (S34.3 known gap)
        m_t_uncertainty=None,
    )
