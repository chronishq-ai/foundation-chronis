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
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Sequence, Tuple, cast

from backbone.hssm.model import GaussianHSMM, KimHSSMModel
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
) -> tuple[GaussianHSMM, list[dict]]:
    """Fit with >= n_init random initializations, return the converged run
    with highest log-likelihood, canonicalized via the MP-02 label-switching
    fix. run_log records every attempt for auditability."""
    if not bypass_init_gate:
        assert n_init >= 10, "Directive Day 8 requires a HARD MINIMUM of 10 random initializations."
    else:
        assert n_init >= 1, "Must run at least 1 random initialization."

    run_log = []
    best_model: GaussianHSMM | None = None
    best_ll = -np.inf

    for i in range(n_init):
        model = GaussianHSMM(n_regimes=n_regimes, n_features=n_features,
                              max_duration=max_duration, seed=base_seed + i)
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
            assert ll is not None
            if ll > best_ll:
                best_ll = ll
                best_model = model

    if best_model is None:
        raise RuntimeError(
            f"None of {n_init} random-init EM runs converged. Real failure, "
            f"not something to paper over — check data quality, n_iter, or "
            f"max_duration before proceeding."
        )

    convergence_rate = sum(1.0 if r["converged"] else 0.0 for r in run_log) / len(run_log)
    assert best_model is not None
    best_model = cast(GaussianHSMM, canonicalize_labels(best_model))
    assert best_model is not None
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
            bypass_init_gate=bypass_init_gate,
        )
        assert model.log_likelihood_ is not None
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
        raise RuntimeError("No models were fitted because k_candidates was empty.")

    assert best_model is not None
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
    assert model.var is not None
    assert model.A is not None
    assert model.mu is not None
    assert model.dur_mu is not None
    assert model.dur_sigma is not None

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
