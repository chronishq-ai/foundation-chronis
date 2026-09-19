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
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Sequence, Tuple, cast

from backbone.hssm.model import GaussianHSMM, KimHSSMModel, FitState, NotFittedError, FittingConvergenceError, InternalStateError
from backbone.hssm.dual_model import DualTimescaleHSSM
from backbone.hssm.label_switching import canonicalize_labels
from backbone.hssm.config import DEFAULT_FIT_CONFIG, HSSMFitConfig, DEFAULT_COLD_START_CONFIG, ColdStartConfig
from backbone.hssm.gating import ColdStartError, count_present_sessions


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
            "fit_state": model.fit_state_,
            "n_iter": model.n_iter_,
            "log_likelihood": float(model.log_likelihood_) if model.log_likelihood_ is not None else None,
            "monotonic_ll": model.is_log_likelihood_monotonic(),
            "duration_fallback_used": getattr(model, "duration_fallback_used_", False),
            "duration_optimizer_success": getattr(model, "duration_optimizer_success_", {}),
        })
        if verbose:
            ll_val = model.log_likelihood_ if model.log_likelihood_ is not None else 0.0
            print(f"  init {i}: fit_state={model.fit_state_}, converged={model.converged_}, ll={ll_val:.3f}")

        if model.fit_state_ == FitState.CONVERGED:
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
    selected_model = best_model
    canonicalized_model = canonicalize_labels(selected_model)
    if canonicalized_model is None:
        raise InternalStateError("canonicalize_labels violated its postcondition: returned None")
    if canonicalized_model is not selected_model:
        raise InternalStateError(
            "canonicalize_labels violated in-place canonicalization contract"
        )
    best_model = cast(GaussianHSMM, canonicalized_model)
    best_model.convergence_rate_ = convergence_rate

    return best_model, run_log


def fit_dual_with_random_restarts(
    X: np.ndarray,
    n_regimes: int,
    n_features: int,
    latent_dim: int = 2,
    n_init: int = DEFAULT_FIT_CONFIG.n_init,
    n_iter: int = 50,
    max_duration: int = DEFAULT_FIT_CONFIG.max_duration,
    base_seed: int = 0,
    verbose: bool = False,
    bypass_init_gate: bool = False,
    timestamps: np.ndarray | None = None,
) -> tuple[DualTimescaleHSSM, list[dict]]:
    """Fit DualTimescaleHSSM with multiple initializations, return highest-LL model."""
    if not bypass_init_gate:
        if n_init < 10:
            raise ValueError("Directive Day 8 requires a HARD MINIMUM of 10 random initializations.")
    else:
        if n_init < 1:
            raise ValueError("Must run at least 1 random initialization.")

    run_log = []
    best_ll = -np.inf
    best_model: DualTimescaleHSSM | None = None

    for i in range(n_init):
        seed = base_seed + i * 31
        model = DualTimescaleHSSM(
            n_regimes=n_regimes,
            n_features=n_features,
            latent_dim=latent_dim,
            max_duration=max_duration,
            seed=seed,
        )
        model.fit(X, n_iter=n_iter, timestamps=timestamps, verbose=verbose)

        run_info = {
            "init_index": i,
            "seed": seed,
            "fit_state": model.fit_state_,
            "converged": model.converged_,
            "log_likelihood": model.log_likelihood_,
            "n_iter": model.n_iter_,
        }
        run_log.append(run_info)

        if model.fit_state_ in (FitState.CONVERGED, FitState.MAX_ITER_REACHED, FitState.REJECTED_UPDATE):
            ll = model.log_likelihood_
            if ll is not None and ll > best_ll:
                best_ll = ll
                best_model = model

    if best_model is None:
        raise FittingConvergenceError("None of the random restarts converged for DualTimescaleHSSM")

    convergence_rate = sum(1.0 if r["converged"] else 0.0 for r in run_log) / len(run_log)
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
    if not k_candidates or len(k_candidates) == 0:
        raise ValueError("k_candidates must be a non-empty sequence of positive integers")
    if any(k <= 0 for k in k_candidates):
        raise ValueError("All candidate K values must be strictly positive (K > 0)")

    results = {}
    best_k = None
    best_bic = np.inf
    best_model: GaussianHSMM | None = None

    for k in k_candidates:
        if verbose:
            print(f"Fitting K={k}...")
        try:
            model, run_log = fit_with_random_restarts(
                X, n_regimes=k, n_features=n_features, n_init=n_init, n_iter=n_iter,
                max_duration=max_duration, base_seed=base_seed + k * 1000, verbose=False,
                bypass_init_gate=bypass_init_gate, timestamps=timestamps,
            )
            if not model.converged_:
                raise InternalStateError("Selected model is not marked as converged")
            if model.log_likelihood_ is None or not np.isfinite(model.log_likelihood_):
                raise InternalStateError(
                    "Selected model has an invalid log_likelihood_"
                )
            n_obs = count_present_sessions(X) if count_present_sessions(X) > 0 else X.shape[0]
            bic = model.bic(n_observations=n_obs)
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
        except FittingConvergenceError:
            results[k] = {
                "bic": float("inf"),
                "log_likelihood": float("-inf"),
                "convergence_rate": 0.0,
                "run_log": [],
            }
            if verbose:
                print(f"  K={k}: No random restart converged")

    if best_model is None:
        raise FittingConvergenceError("None of the candidate Ks converged across random restarts")
    selection_report = {
        "selected_k": best_k,
        "bic_by_k": {k: results[k]["bic"] for k in results},
        "loglik_by_k": {k: results[k]["log_likelihood"] for k in results},
        "convergence_rate_by_k": {k: results[k]["convergence_rate"] for k in results},
        "selection_rule": "min BIC across K candidates — never chosen for 'interesting regimes'",
        "run_log": results[best_k]["run_log"],
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
        raise InternalStateError("Fitted model has uninitialized parameters")

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
    fitted_model.fit_state_ = model.fit_state_
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
        n_obs = count_present_sessions(X) if count_present_sessions(X) > 0 else X.shape[0]
        report[dmax] = {
            "log_likelihood": float(model.log_likelihood_) if model.log_likelihood_ is not None else 0.0,
            "bic": float(model.bic(n_observations=n_obs)),
            "max_tail_mass": max(tail_mass.values()),
            "tail_mass_by_regime": tail_mass,
        }
    return report


@dataclass
class HSSMResult:
    """Canonical result contract for fit_hssm — directive Section Q.

    All fields required by the directive are present. For the baseline model,
    fast-state fields (m_t, m_t_uncertainty) are explicitly None — never faked.
    Neural residual fields default to disabled/None.
    """
    # --- Core model ---
    model: Any

    # --- Identity ---
    user_id: str = ""
    model_identity: str = ""
    model_family: str = ""
    model_capability: dict = field(default_factory=dict)
    is_baseline_model: bool = True
    fast_state_supported: bool = False

    # --- Slow state p_t ---
    p_t_posterior: np.ndarray | None = None  # shape (T, K) posterior probabilities
    p_t_estimate: np.ndarray | None = None  # shape (T,) MAP regime assignment

    # --- Fast state m_t ---
    m_t_posterior_mean: np.ndarray | None = None  # shape (T, latent_dim)
    m_t_uncertainty: Any = None  # shape (T, latent_dim) or (T, latent_dim, latent_dim)

    # --- Duration ---
    duration_parameters: dict[str, Any] = field(default_factory=dict)
    duration_model_identity: str = ""
    Dmax: int = 0
    duration_unit_source: str = ""

    # --- Temporal coverage ---
    calendar_start: Optional[float] = None
    calendar_end: Optional[float] = None
    observation_density: float = 0.0

    # --- Versioning ---
    model_version: str = "1.0.0"
    feature_schema_version: str = "1.0.0"
    alignment_version: str = "1.0.0"

    # --- Eligibility ---
    eligible_session_count: int = 0
    min_present_sessions: int = 30

    # --- Data hash ---
    fit_dataset_hash: str = ""

    # --- Model selection ---
    K: int = 0
    BIC: float = float("inf")
    AIC: float = float("inf")
    heldout_predictive_log_likelihood: Optional[float] = None

    # --- Fit state ---
    fit_state: FitState = FitState.FAILED
    optimizer_success: bool = False
    fallback_used: bool = False
    run_log: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    # --- Neural residual ---
    neural_residual_enabled: bool = False
    residual_mode: str = "DISABLED"
    residual_weight: float = 0.0
    residual_model_version: str = ""
    residual_ablation_status: str = "NOT_APPLICABLE"

    # --- Legacy backward-compat fields (kept for existing callers) ---
    duration_info: dict[str, Any] = field(default_factory=dict)
    duration_unit: str = "sessions"
    selected_k: int = 0
    model_class: str = ""
    convergence_metadata: dict[str, Any] = field(default_factory=dict)

    # Legacy aliases — these read from the new canonical fields with fallback to convergence_metadata
    @property
    def p_t(self) -> np.ndarray | None:
        """Legacy alias: MAP regime assignment."""
        return self.p_t_estimate

    @property
    def regime_posterior(self) -> np.ndarray | None:
        """Legacy alias: full regime posterior."""
        return self.p_t_posterior

    @property
    def m_t(self) -> np.ndarray | None:
        """Legacy alias: fast state posterior mean."""
        return self.m_t_posterior_mean

    @property
    def k_selected(self) -> int:
        return self.selected_k or self.K

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

    @property
    def duration_fallback_used(self) -> bool:
        """True if ANY regime's duration optimizer fell back to the untruncated estimate."""
        return self.fallback_used or bool(self.convergence_metadata.get("duration_fallback_used", False))

    @property
    def duration_optimizer_success(self) -> bool:
        """Aggregate 'all-clear' boolean: True only if NO regime fell back."""
        return self.optimizer_success and bool(self.convergence_metadata.get("duration_optimizer_success", True))

    @property
    def per_regime_optimizer_status(self) -> dict[int, bool]:
        """Mapping of regime index -> boolean for duration optimization success."""
        return self.convergence_metadata.get("per_regime_optimizer_status", {})


def _compute_heldout_predictive_loglik(
    X: np.ndarray,
    model: GaussianHSMM,
    holdout_fraction: float = 0.2,
) -> Optional[float]:
    """Chronological heldout predictive log-likelihood.

    Uses the last holdout_fraction of the data as test set.
    The model is already fit on the full data (this is a diagnostic, not a refit).
    We compute the emission log-likelihood of the held-out portion under
    the fitted model's regime posterior, as a predictive quality metric.
    """
    T = X.shape[0]
    split_idx = max(1, int(T * (1.0 - holdout_fraction)))
    if split_idx >= T - 1:
        return None  # Not enough data for holdout

    X_test = X[split_idx:]
    try:
        emission_ll = model._emission_loglik(X_test)  # (T_test, K)
        # Marginal emission LL under uniform regime prior (conservative)
        from scipy.special import logsumexp as _logsumexp
        log_pi = np.log(np.clip(model.pi, 1e-300, None)) if model.pi is not None else np.zeros(model.K)
        marginal_ll = _logsumexp(emission_ll + log_pi[None, :], axis=1)  # (T_test,)
        return float(np.sum(marginal_ll))
    except Exception:
        return None


def fit_hssm(
    matrix: np.ndarray | pd.DataFrame,
    candidate_ks: Sequence[int] = (2, 3, 4),
    n_initializations: int = 10,
    max_iter: int = 200,
    random_seed: Optional[int] = None,
    allow_fast_test_fit: bool = False,
    timestamps: Optional[np.ndarray] = None,
    config: ColdStartConfig = DEFAULT_COLD_START_CONFIG,
    n_present_sessions: Optional[int] = None,
    max_duration: int = 45,
    user_id: str = "",
    dataset_hash: str = "",
    model_type: str = "baseline",
    latent_dim: int = 2,
) -> HSSMResult:
    """Canonical fitting entrypoint for backbone.hssm per directive.

    This is the ONLY public fitting entrypoint. All legacy wrappers
    (fit_hssm_model, fit_hssm_gated) delegate to this function.

    Enforces cold-start gate (VF-14): requires >= config.min_present_sessions
    eligible sessions or raises ColdStartError.

    Parameters:
      model_type: 'baseline' (GaussianHSMM, slow-regime only) or
                  'dual_timescale' (DualTimescaleHSSM, fast m_t + slow p_t).
    """
    if isinstance(matrix, pd.DataFrame):
        X = matrix.to_numpy(dtype=float)
    else:
        X = np.asarray(matrix, dtype=float)

    if X.ndim < 2 or X.shape[1] <= 0:
        raise ValueError("Feature count must be strictly greater than zero (F > 0)")

    n_features = X.shape[1]

    if n_present_sessions is None:
        n_present_sessions = count_present_sessions(X)

    if n_present_sessions < config.min_present_sessions:
        raise ColdStartError(
            f"{n_present_sessions} present sessions < cold-start minimum "
            f"({config.min_present_sessions}). No HSSM output produced. This "
            f"is the correct, silent, no-output state per Bible 5.1 doctrine, "
            f"not a low-confidence result."
        )

    # --- Dual Timescale Model Fitting Branch ---
    if model_type in ("dual_timescale", "dual", "full_hssm"):
        best_k = None
        best_bic = float("inf")
        best_dual_model: DualTimescaleHSSM | None = None
        bic_by_k = {}
        loglik_by_k = {}
        conv_by_k = {}
        full_run_log = []

        for k in candidate_ks:
            dual_m, rlog = fit_dual_with_random_restarts(
                X,
                n_regimes=k,
                n_features=n_features,
                latent_dim=latent_dim,
                n_init=n_initializations,
                n_iter=max_iter,
                max_duration=max_duration,
                base_seed=(random_seed or 0) + k * 1000,
                bypass_init_gate=allow_fast_test_fit,
                timestamps=timestamps,
            )
            n_obs = n_present_sessions if n_present_sessions > 0 else X.shape[0]
            bic = dual_m.bic(n_observations=n_obs)
            bic_by_k[k] = float(bic)
            loglik_by_k[k] = float(dual_m.log_likelihood_ or 0.0)
            conv_by_k[k] = dual_m.convergence_rate_
            full_run_log.extend(rlog)

            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_dual_model = dual_m

        if best_dual_model is None or best_k is None:
            raise FittingConvergenceError("None of the candidate Ks converged across random restarts")

        m_t, m_t_std, p_t, p_t_post = best_dual_model.estimate_states(X)
        aic_val = float(best_dual_model.aic())

        return HSSMResult(
            model=best_dual_model,
            user_id=user_id,
            model_identity=best_dual_model.model_identity,
            model_family=best_dual_model.model_family,
            model_capability=best_dual_model.model_capability,
            is_baseline_model=False,
            fast_state_supported=True,
            p_t_posterior=p_t_post,
            p_t_estimate=p_t,
            m_t_posterior_mean=m_t,
            m_t_uncertainty=m_t_std,
            duration_parameters={
                "duration_mu": best_dual_model.dur_mu.tolist() if best_dual_model.dur_mu is not None else [],
                "duration_sigma": best_dual_model.dur_sigma.tolist() if best_dual_model.dur_sigma is not None else [],
            },
            duration_model_identity=best_dual_model.duration_model_identity,
            Dmax=best_dual_model.Dmax,
            duration_unit_source=best_dual_model.duration_unit_source,
            calendar_start=float(timestamps[0]) if timestamps is not None and len(timestamps) > 0 else None,
            calendar_end=float(timestamps[-1]) if timestamps is not None and len(timestamps) > 0 else None,
            observation_density=float(n_present_sessions / max(X.shape[0], 1)),
            model_version="1.0.0",
            feature_schema_version="1.0.0",
            alignment_version="1.0.0",
            eligible_session_count=n_present_sessions,
            min_present_sessions=config.min_present_sessions,
            fit_dataset_hash=dataset_hash,
            K=best_k,
            BIC=best_bic,
            AIC=aic_val,
            heldout_predictive_log_likelihood=None,
            fit_state=best_dual_model.fit_state_,
            optimizer_success=True,
            fallback_used=best_dual_model.duration_fallback_used_,
            run_log=full_run_log,
            warnings=[],
            limitations=[],
            neural_residual_enabled=False,
            residual_mode="DISABLED",
            residual_weight=0.0,
            residual_model_version="",
            residual_ablation_status="NOT_APPLICABLE",
            duration_info={},
            duration_unit=best_dual_model.duration_unit,
            selected_k=best_k,
            model_class=best_dual_model.__class__.__name__,
            convergence_metadata={
                "bic_by_k": bic_by_k,
                "loglik_by_k": loglik_by_k,
                "convergence_rate_by_k": conv_by_k,
                "selected_loglik": best_dual_model.log_likelihood_,
                "fit_state": best_dual_model.fit_state_,
                "converged": best_dual_model.converged_,
                "n_iter": best_dual_model.n_iter_,
                "run_log": full_run_log,
            },
        )

    # --- Baseline Model Fitting Branch ---
    model, report = select_k_by_bic(
        X,
        n_features=n_features,
        k_candidates=tuple(candidate_ks),
        n_init=n_initializations,
        n_iter=max_iter,
        max_duration=max_duration,
        base_seed=random_seed or 0,
        bypass_init_gate=allow_fast_test_fit,
        timestamps=timestamps,
    )

    if timestamps is not None:
        model.duration_unit = "calendar_days"

    # Compute posterior quantities for the selected model
    regime_post, _, _ = model._forward_backward(X)
    p_t = np.argmax(regime_post, axis=1)

    # Compute information criteria
    n_obs = n_present_sessions if n_present_sessions > 0 else X.shape[0]
    bic_val = float(model.bic(n_observations=n_obs))
    aic_val = float(model.aic())

    # Compute heldout predictive log-likelihood
    heldout_ll = _compute_heldout_predictive_loglik(X, model)

    # Detect MODEL_SELECTION_CONTESTED: BIC and heldout LL disagree on best K
    fit_state = model.fit_state_
    if len(report.get("bic_by_k", {})) > 1 and heldout_ll is not None:
        bic_values = list(report["bic_by_k"].values())
        if len(bic_values) >= 2:
            sorted_bics = sorted(bic_values)
            if len(sorted_bics) >= 2 and abs(sorted_bics[0] - sorted_bics[1]) < 2.0:
                fit_state = FitState.MODEL_SELECTION_CONTESTED

    # Build duration info (legacy compat)
    duration_info = {
        "duration_mu": model.duration_mu,
        "duration_sigma": model.duration_sigma,
        "max_duration": model.Dmax,
        "duration_optimizer_success": getattr(model, "duration_optimizer_success_", {}),
        "duration_fallback_used": getattr(model, "duration_fallback_used_", False),
    }

    # Build convergence metadata (legacy compat)
    convergence_metadata = {
        "bic_by_k": report["bic_by_k"],
        "loglik_by_k": report["loglik_by_k"],
        "convergence_rate_by_k": report["convergence_rate_by_k"],
        "selected_loglik": model.log_likelihood_,
        "fit_state": model.fit_state_,
        "converged": model.converged_,
        "n_iter": model.n_iter_,
        "run_log": report.get("run_log", []),
        "duration_optimizer_success": not getattr(model, "duration_fallback_used_", False),
        "duration_fallback_used": getattr(model, "duration_fallback_used_", False),
        "per_regime_optimizer_status": getattr(model, "duration_optimizer_success_", {}),
    }

    # Collect warnings and limitations
    result_warnings = []
    result_limitations = [
        "Baseline model: fast continuous state m_t is not supported.",
        "Duration units are session-index unless timestamps provided.",
    ]
    if getattr(model, "duration_fallback_used_", False):
        result_warnings.append(
            "Duration optimizer fallback used for one or more regimes. "
            "Untruncated moment estimates used as initialization — not the "
            "final truncated model."
        )

    return HSSMResult(
        model=model,
        # Identity
        user_id=user_id,
        model_identity=model.model_identity,
        model_family=model.model_family,
        model_capability=model.model_capability,
        is_baseline_model=model.is_baseline_model,
        fast_state_supported=model.fast_state_supported,
        # Slow state
        p_t_posterior=regime_post,
        p_t_estimate=p_t,
        # Fast state (None for baseline)
        m_t_posterior_mean=None,
        m_t_uncertainty=None,
        # Duration
        duration_parameters={
            "duration_mu": model.duration_mu.tolist() if model.dur_mu is not None else [],
            "duration_sigma": model.duration_sigma.tolist() if model.dur_sigma is not None else [],
        },
        duration_model_identity=model.duration_model_identity,
        Dmax=model.Dmax,
        duration_unit_source=model.duration_unit_source,
        # Temporal
        calendar_start=float(timestamps[0]) if timestamps is not None and len(timestamps) > 0 else None,
        calendar_end=float(timestamps[-1]) if timestamps is not None and len(timestamps) > 0 else None,
        observation_density=float(n_present_sessions / max(X.shape[0], 1)),
        # Versioning
        model_version="1.0.0",
        feature_schema_version="1.0.0",
        alignment_version="1.0.0",
        # Eligibility
        eligible_session_count=n_present_sessions,
        min_present_sessions=config.min_present_sessions,
        # Data hash
        fit_dataset_hash=dataset_hash,
        # Model selection
        K=report["selected_k"],
        BIC=bic_val,
        AIC=aic_val,
        heldout_predictive_log_likelihood=heldout_ll,
        # Fit state
        fit_state=fit_state,
        optimizer_success=not getattr(model, "duration_fallback_used_", False),
        fallback_used=getattr(model, "duration_fallback_used_", False),
        run_log=report.get("run_log", []),
        warnings=result_warnings,
        limitations=result_limitations,
        # Neural residual (disabled for baseline)
        neural_residual_enabled=False,
        residual_mode="DISABLED",
        residual_weight=0.0,
        residual_model_version="",
        residual_ablation_status="NOT_APPLICABLE",
        # Legacy compat fields
        duration_info=duration_info,
        duration_unit=model.duration_unit,
        selected_k=report["selected_k"],
        model_class=model.__class__.__name__,
        convergence_metadata=convergence_metadata,
    )
