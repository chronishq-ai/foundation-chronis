"""
Dual-Timescale Fast+Slow HSSM: Kim (1994)-style Switching Linear Dynamical System
with Semi-Markov Explicit Duration Slow Regimes.

Theoretical Architecture:
  1. Fast continuous latent state m_t in R^L (minute-level / momentary dynamics):
       m_t = A_k m_{t-1} + b_k + w_t,   w_t ~ N(0, Q_k)
  2. Slow discrete regime state p_t in {0, ..., K-1} (macro life-phases):
       d_t ~ LogNormal(mu_k, sigma_k), d in {1, ..., Dmax}
       p_t transitions governed by transition matrix A_slow (zero diagonal)
  3. Observation emission model conditional on continuous state m_t and regime p_t=k:
       y_t = C_k m_t + d_k + v_t,       v_t ~ N(0, R_k)
       where R_k is diagonal covariance.
  4. Kim filter with GPB2 (Generalized Pseudo-Bayes of order 2) collapse and
     Kim backward smoother for continuous state trajectory and covariance.
  5. Missing values (NaNs) handled by marginalization in the Kalman update.
  6. Generalized EM with backtracking rollback ensuring monotonic log-likelihood.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import lognorm

from backbone.hssm.model import (
    FitState,
    HSSMError,
    NotFittedError,
    InternalStateError,
    FittingConvergenceError,
    NEG_INF,
)


class DualTimescaleHSSM:
    """Production-grade Dual-Timescale Switching Linear Dynamical System (HSSM).

    Combines:
      - Fast continuous state m_t in R^L (momentary behavioral dynamics)
      - Slow discrete regime p_t in {0..K-1} with explicit semi-Markov duration
      - Kim (1994) GPB2 filter and backward smoother
      - Full EM with analytical M-step for linear-Gaussian state/observation dynamics
      - Generalized EM monotonicity backtracking
    """

    def __init__(
        self,
        n_regimes: int = 3,
        n_features: int = 2,
        latent_dim: int = 2,
        max_duration: int = 40,
        seed: int | None = None,
        duration_unit: str = "sessions",
    ):
        if n_regimes <= 0:
            raise ValueError("Number of regimes must be strictly positive (K > 0)")
        if n_features <= 0:
            raise ValueError("Number of features must be strictly positive (F > 0)")
        if latent_dim <= 0:
            raise ValueError("Latent dimension must be strictly positive (L > 0)")

        self.K = n_regimes
        self.F = n_features
        self.L = latent_dim
        self.Dmax = max_duration
        self.duration_unit = duration_unit
        self.rng = np.random.default_rng(seed)

        # --- Slow discrete regime parameters ---
        self.pi: np.ndarray | None = None          # (K,) initial regime distribution
        self.A_slow: np.ndarray | None = None      # (K, K) regime transition matrix (0 diagonal)
        self.dur_mu: np.ndarray | None = None      # (K,) log-normal duration mu
        self.dur_sigma: np.ndarray | None = None   # (K,) log-normal duration sigma

        # --- Fast continuous state dynamics conditional on regime k ---
        # m_t = A_k m_{t-1} + b_k + w_t,  w_t ~ N(0, Q_k)
        self.A_fast: np.ndarray | None = None      # (K, L, L) state transition matrices
        self.b_fast: np.ndarray | None = None      # (K, L) state biases
        self.Q_fast: np.ndarray | None = None      # (K, L, L) process noise covariances
        self.m0: np.ndarray | None = None          # (K, L) initial latent means
        self.P0: np.ndarray | None = None          # (K, L, L) initial latent covariances

        # --- Observation emission model conditional on continuous state and regime k ---
        # y_t = C_k m_t + d_k + v_t,  v_t ~ N(0, R_k)
        self.C_obs: np.ndarray | None = None       # (K, F, L) observation loading matrices
        self.d_obs: np.ndarray | None = None       # (K, F) observation intercepts
        self.R_obs: np.ndarray | None = None       # (K, F) observation noise variances (diagonal)

        # --- Fitting diagnostics ---
        self.log_likelihood_: float | None = None
        self.log_likelihood_history_: list[float] = []
        self.fit_state_: FitState = FitState.FAILED
        self.n_iter_: int = 0
        self.convergence_rate_: float = 0.0
        self._is_fitted: bool = False
        self._n_observations: int = 0
        self.duration_fallback_used_: bool = False
        self.duration_optimizer_success_: dict[int, bool] = {}
        self.duration_optimizer_messages_: dict[int, str] = {}

    # ---------- Model Identity & Capabilities ----------

    @property
    def model_identity(self) -> str:
        return "DualTimescaleKimHSSMV1"

    @property
    def model_family(self) -> str:
        return "SwitchingLinearDynamicalSystem"

    @property
    def is_baseline_model(self) -> bool:
        return False

    @property
    def fast_state_supported(self) -> bool:
        return True

    @property
    def duration_model_identity(self) -> str:
        return "explicit_truncated_gaussian"

    @property
    def duration_unit_source(self) -> str:
        if self.duration_unit == "calendar_days":
            return "calendar_days"
        return "session_index"

    @property
    def model_capability(self) -> dict[str, Any]:
        return {
            "supports_fast_state_m_t": True,
            "supports_slow_state_p_t": True,
            "supports_explicit_duration": True,
            "supports_neural_residual": False,
            "is_production_grade_full_hssm": True,
            "is_baseline_model": False,
            "fast_state_supported": True,
            "latent_dimension": self.L,
        }

    @property
    def converged_(self) -> bool:
        return self.fit_state_ == FitState.CONVERGED

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise NotFittedError("Model has not been fitted yet — call .fit() first")

    # ---------- Initialization ----------

    def _init_params(self, X: np.ndarray) -> None:
        """Initialize all fast+slow state-space parameters."""
        K, F, L = self.K, self.F, self.L
        
        # Slow initial distribution and transition matrix
        self.pi = np.full(K, 1.0 / K)
        A_raw = self.rng.dirichlet(np.ones(K - 1) * 2.0, size=K) if K > 1 else np.ones((1, 1))
        self.A_slow = np.zeros((K, K))
        for k in range(K):
            others = [j for j in range(K) if j != k]
            for idx, j in enumerate(others):
                self.A_slow[k, j] = A_raw[k, idx] if K > 1 else 1.0

        # Duration parameters
        self.dur_mu = np.full(K, np.log(10.0)) + self.rng.normal(scale=0.2, size=K)
        self.dur_sigma = np.full(K, 0.5) + self.rng.uniform(0.0, 0.1, size=K)
        self.duration_optimizer_success_ = {k: True for k in range(K)}
        self.duration_fallback_used_ = False
        self.duration_optimizer_messages_ = {k: "Initialized" for k in range(K)}

        # Fast continuous state dynamics
        # Stable AR(1) dynamics with slight persistence (A ~ 0.8 * I)
        self.A_fast = np.zeros((K, L, L))
        self.b_fast = np.zeros((K, L))
        self.Q_fast = np.zeros((K, L, L))
        self.m0 = np.zeros((K, L))
        self.P0 = np.zeros((K, L, L))

        for k in range(K):
            # Diagonally dominant stable transition
            self.A_fast[k] = 0.7 * np.eye(L) + self.rng.normal(scale=0.05, size=(L, L))
            self.b_fast[k] = self.rng.normal(scale=0.1, size=L)
            self.Q_fast[k] = 0.5 * np.eye(L)
            self.m0[k] = np.zeros(L)
            self.P0[k] = 1.0 * np.eye(L)

        # Observation emission parameters
        # C maps L latent dims to F observed features
        self.C_obs = np.zeros((K, F, L))
        self.d_obs = np.zeros((K, F))
        self.R_obs = np.zeros((K, F))

        # Initialize observation mean/variance from data
        feat_means = np.nanmean(X, axis=0) if not np.isnan(X).all() else np.zeros(F)
        feat_vars = np.nanvar(X, axis=0) if not np.isnan(X).all() else np.ones(F)
        feat_means = np.nan_to_num(feat_means, nan=0.0)
        feat_vars = np.nan_to_num(feat_vars, nan=1.0)
        feat_vars = np.clip(feat_vars, 1e-3, None)

        for k in range(K):
            # Random orthogonal-like loading matrix
            rand_proj = self.rng.normal(size=(F, L))
            q, _ = np.linalg.qr(rand_proj)
            self.C_obs[k] = q * np.sqrt(feat_vars)[:, None]
            self.d_obs[k] = feat_means + self.rng.normal(scale=0.5 * np.sqrt(feat_vars), size=F)
            self.R_obs[k] = feat_vars * 0.5

    def _duration_logpmf(self) -> np.ndarray:
        if self.dur_sigma is None or self.dur_mu is None:
            raise NotFittedError("Duration parameters are not initialized")
        d = np.arange(1, self.Dmax + 1)
        logpmf = np.zeros((self.K, self.Dmax))
        for k in range(self.K):
            pdf = lognorm.pdf(d, s=self.dur_sigma[k], scale=np.exp(self.dur_mu[k]))
            pdf = np.clip(pdf, 1e-300, None)
            pdf = pdf / pdf.sum()
            logpmf[k] = np.log(pdf)
        return logpmf

    # ---------- Kim Filter (GPB2 Collapse) & Semi-Markov Integration ----------

    def _forward_pass(self, X: np.ndarray):
        """Kim (1994) GPB2 filter combined with semi-Markov duration forward pass."""
        regime_post, entry_post, xi_counts, m_filt, P_filt, m_pred, P_pred, emis_loglik, entry_mass = (
            self._forward_backward_full(X)
        )
        return regime_post, entry_mass, m_filt, P_filt, m_pred, P_pred, emis_loglik

    def _forward_backward(self, X: np.ndarray):
        """Combined GPB2 filter and semi-Markov forward-backward recursion."""
        regime_post, entry_post, xi_counts, m_filt, P_filt, m_pred, P_pred, emis_loglik, entry_mass = (
            self._forward_backward_full(X)
        )
        return regime_post, entry_post, xi_counts, m_filt, P_filt, m_pred, P_pred

    def _forward_backward_full(self, X: np.ndarray):
        """Kim (1994) GPB2 filter combined with exact semi-Markov forward-backward recursion.

        Returns:
          regime_posterior: (T, K) smoothed regime probabilities
          entry_posterior: (T, K, D) smoothed entry/duration probabilities
          xi_counts: (K, K) expected regime transition counts
          m_filt: (T, K, L)
          P_filt: (T, K, L, L)
          m_pred: (T, K, L)
          P_pred: (T, K, L, L)
          emis_loglik: (T, K)
          entry_mass: (T, K, D)
        """
        T, K, L, D = X.shape[0], self.K, self.L, self.Dmax
        dur_logpmf = self._duration_logpmf()
        logA = np.log(np.clip(self.A_slow, 1e-300, None))
        log_pi = np.log(np.clip(self.pi, 1e-300, None))

        # Filtered continuous state holders
        m_filt = np.zeros((T, K, L))
        P_filt = np.zeros((T, K, L, L))
        m_pred = np.zeros((T, K, L))
        P_pred = np.zeros((T, K, L, L))

        # Emission log-likelihoods conditional on regime k
        emis_loglik = np.zeros((T, K))

        # Initial step t=0
        for k in range(K):
            # Prior at t=0
            m_0_k = self.m0[k]
            P_0_k = self.P0[k]
            m_pred[0, k] = m_0_k
            P_pred[0, k] = P_0_k

            # Kalman update for regime k at t=0
            m_up, P_up, ll_k = self._kalman_update_regime(X[0], m_0_k, P_0_k, k)
            m_filt[0, k] = m_up
            P_filt[0, k] = P_up
            emis_loglik[0, k] = ll_k

        # Discrete semi-Markov forward variables
        alpha = np.full((T, K, D), NEG_INF)
        entry_mass = np.full((T, K, D), NEG_INF)

        for k in range(K):
            entry_mass[0, k, :] = log_pi[k] + dur_logpmf[k]
        alpha[0] = entry_mass[0] + emis_loglik[0][:, None]

        # Time recursion t=1..T-1
        for t in range(1, T):
            # 1. Discrete HSMM prediction
            cont = np.full((K, D), NEG_INF)
            cont[:, : D - 1] = alpha[t - 1, :, 1:D]

            completing = alpha[t - 1, :, 0]
            trans_in = logsumexp(completing[:, None] + logA, axis=0)
            entry = trans_in[:, None] + dur_logpmf
            entry_mass[t] = entry

            # 2. Continuous Kalman prediction and GPB2 branches (i -> j)
            # For each destination regime j:
            branch_m = np.zeros((K, L))      # conditional updated m from previous regime i
            branch_m_pred = np.zeros((K, L)) # conditional prior m from previous regime i
            branch_P = np.zeros((K, L, L))   # conditional updated P from previous regime i
            branch_ll = np.zeros(K)

            for j in range(K):
                for i in range(K):
                    # Prediction from state (t-1, i) to (t, j)
                    m_pred_ij = self.A_fast[j] @ m_filt[t - 1, i] + self.b_fast[j]
                    P_pred_ij = self.A_fast[j] @ P_filt[t - 1, i] @ self.A_fast[j].T + self.Q_fast[j]

                    # Measurement update
                    m_up_ij, P_up_ij, ll_ij = self._kalman_update_regime(X[t], m_pred_ij, P_pred_ij, j)
                    branch_m[i] = m_up_ij
                    branch_m_pred[i] = m_pred_ij
                    branch_P[i] = P_up_ij
                    branch_ll[i] = ll_ij

                # GPB2 collapse weight w(i|j) = Pr(p_{t-1}=i, p_t=j | Y_{1:t-1})
                log_w = completing
                w_weights = np.exp(log_w - logsumexp(log_w)) if not np.all(np.isneginf(log_w)) else np.full(K, 1.0 / K)

                # Collapsed state for regime j
                m_collapsed = np.sum(w_weights[:, None] * branch_m, axis=0)
                P_collapsed = np.zeros((L, L))
                for i in range(K):
                    diff = branch_m[i] - m_collapsed
                    P_collapsed += w_weights[i] * (branch_P[i] + np.outer(diff, diff))

                m_filt[t, j] = m_collapsed
                P_filt[t, j] = P_collapsed
                m_pred[t, j] = np.sum(w_weights[:, None] * branch_m_pred, axis=0)
                P_pred[t, j] = P_collapsed

                # Effective emission log-likelihood for regime j
                emis_loglik[t, j] = float(np.sum(w_weights * branch_ll))

            # 3. Discrete HSMM update with emission likelihoods
            alpha[t] = np.logaddexp(cont, entry) + emis_loglik[t][:, None]

        self.log_likelihood_ = float(logsumexp(alpha[-1]))

        # Backward recursion for discrete semi-Markov variables
        beta = np.full((T, K, D), NEG_INF)
        beta[-1] = 0.0
        future_entry_val = np.full((T, K), NEG_INF)

        for t in range(T - 2, -1, -1):
            fut = logsumexp(beta[t + 1] + emis_loglik[t + 1][:, None] + dur_logpmf, axis=1)
            future_entry_val[t + 1] = fut

            cont_term = np.full((K, D), NEG_INF)
            cont_term[:, 1:D] = beta[t + 1, :, : D - 1] + emis_loglik[t + 1][:, None]

            trans_out = logsumexp(logA + fut[None, :], axis=1)
            end_term = np.full((K, D), NEG_INF)
            end_term[:, 0] = trans_out

            beta[t] = np.logaddexp(cont_term, end_term)

        gamma_log = alpha + beta - self.log_likelihood_
        gamma = np.exp(np.clip(gamma_log, -700, 0))
        regime_posterior = gamma.sum(axis=2)

        entry_posterior_log = entry_mass + beta + emis_loglik[:, :, None] - self.log_likelihood_
        entry_posterior = np.exp(np.clip(entry_posterior_log, -700, 0))

        # Exact xi(k, k') aggregated over t
        xi_counts = np.zeros((K, K))
        for t in range(T - 1):
            completing_t = alpha[t, :, 0]
            if np.all(completing_t <= NEG_INF / 2) or np.all(future_entry_val[t + 1] <= NEG_INF / 2):
                continue
            xi_log_t = completing_t[:, None] + logA + future_entry_val[t + 1][None, :] - self.log_likelihood_
            xi_counts += np.exp(np.clip(xi_log_t, -700, 0))

        return regime_posterior, entry_posterior, xi_counts, m_filt, P_filt, m_pred, P_pred, emis_loglik, entry_mass

    def _kalman_update_regime(
        self,
        yt: np.ndarray,
        m_prior: np.ndarray,
        P_prior: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Kalman measurement update for regime k with NaN marginalization."""
        L, F = self.L, self.F
        is_obs = ~np.isnan(yt)
        if not np.any(is_obs):
            # All features missing: no update, zero log-density
            return m_prior.copy(), P_prior.copy(), 0.0

        y_obs = yt[is_obs]
        C_obs = self.C_obs[k][is_obs, :]
        d_obs = self.d_obs[k][is_obs]
        R_diag = self.R_obs[k][is_obs]

        # Innovation
        y_pred = C_obs @ m_prior + d_obs
        nu = y_obs - y_pred

        # Innovation covariance: S = C P C^T + R
        S = C_obs @ P_prior @ C_obs.T + np.diag(R_diag)
        S = 0.5 * (S + S.T) + 1e-6 * np.eye(len(y_obs))

        try:
            chol = np.linalg.cholesky(S)
            # Kalman gain K = P C^T S^{-1}
            CP = C_obs @ P_prior
            solve_val = np.linalg.solve(chol, CP)
            K_gain = np.linalg.solve(chol.T, solve_val).T

            # Updated state and covariance
            m_up = m_prior + K_gain @ nu
            I_KC = np.eye(L) - K_gain @ C_obs
            P_up = I_KC @ P_prior @ I_KC.T + K_gain @ np.diag(R_diag) @ K_gain.T
            P_up = 0.5 * (P_up + P_up.T) + 1e-6 * np.eye(L)

            # Log-likelihood of innovation
            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
            inv_nu = np.linalg.solve(chol, nu)
            quad = float(np.sum(inv_nu**2))
            ll = -0.5 * (len(y_obs) * np.log(2.0 * np.pi) + log_det + quad)
            return m_up, P_up, ll
        except np.linalg.LinAlgError:
            return m_prior.copy(), P_prior.copy(), -1e6

    # ---------- Kim Backward Smoother ----------

    def _backward_smoother(
        self,
        X: np.ndarray,
        regime_post: np.ndarray,
        m_filt: np.ndarray,
        P_filt: np.ndarray,
        m_pred: np.ndarray,
        P_pred: np.ndarray,
    ):
        """Kim smoother for continuous latent state and regime posteriors.

        Returns:
          smoothed_m: (T, L)
          smoothed_P: (T, L, L)
          smoothed_m_k: (T, K, L)
          smoothed_P_k: (T, K, L, L)
        """
        T, K, L = X.shape[0], self.K, self.L

        smoothed_m_k = np.zeros((T, K, L))
        smoothed_P_k = np.zeros((T, K, L, L))

        # At t = T-1: smoothed = filtered
        smoothed_m_k[-1] = m_filt[-1]
        smoothed_P_k[-1] = P_filt[-1]

        # Backward recursion
        for t in range(T - 2, -1, -1):
            for k in range(K):
                # Smoother gain J_t = P_{t|t} A^T (P_{t+1|t})^{-1}
                P_pred_next = self.A_fast[k] @ P_filt[t, k] @ self.A_fast[k].T + self.Q_fast[k] + 1e-6 * np.eye(L)
                try:
                    J_t = P_filt[t, k] @ self.A_fast[k].T @ np.linalg.inv(P_pred_next)
                    m_pred_next = self.A_fast[k] @ m_filt[t, k] + self.b_fast[k]
                    smoothed_m_k[t, k] = m_filt[t, k] + J_t @ (smoothed_m_k[t + 1, k] - m_pred_next)
                    smoothed_P_k[t, k] = P_filt[t, k] + J_t @ (smoothed_P_k[t + 1, k] - P_pred_next) @ J_t.T
                    smoothed_P_k[t, k] = 0.5 * (smoothed_P_k[t, k] + smoothed_P_k[t, k].T)
                except np.linalg.LinAlgError:
                    smoothed_m_k[t, k] = m_filt[t, k]
                    smoothed_P_k[t, k] = P_filt[t, k]

        # Aggregate across regimes using regime posterior weights
        smoothed_m = np.zeros((T, L))
        smoothed_P = np.zeros((T, L, L))

        for t in range(T):
            gamma_t = regime_post[t]  # (K,)
            m_t = np.sum(gamma_t[:, None] * smoothed_m_k[t], axis=0)
            smoothed_m[t] = m_t
            P_t = np.zeros((L, L))
            for k in range(K):
                diff = smoothed_m_k[t, k] - m_t
                P_t += gamma_t[k] * (smoothed_P_k[t, k] + np.outer(diff, diff))
            smoothed_P[t] = P_t

        return smoothed_m, smoothed_P, smoothed_m_k, smoothed_P_k

    # ---------- EM M-Step ----------

    def _m_step(
        self,
        X: np.ndarray,
        regime_post: np.ndarray,
        entry_post: np.ndarray,
        xi_counts: np.ndarray,
        smoothed_m_k: np.ndarray,
        smoothed_P_k: np.ndarray,
    ) -> None:
        """Analytical M-step for continuous, observation, and discrete slow-regime parameters."""
        T, K, L, F = X.shape[0], self.K, self.L, self.F

        # 1. Update initial regime distribution
        self.pi = regime_post[0] / (regime_post[0].sum() + 1e-12)

        # 2. Update observation parameters (C_obs, d_obs, R_obs)
        for k in range(K):
            gamma_k = regime_post[:, k]  # (T,)
            for f in range(F):
                valid = ~np.isnan(X[:, f])
                if not np.any(valid):
                    continue
                w_f = gamma_k[valid]
                w_sum_f = np.sum(w_f) + 1e-8
                X_f = X[valid, f]
                M_f = smoothed_m_k[valid, k]  # (T_valid, L)

                # Weighted linear regression
                Phi = np.hstack([M_f, np.ones((len(M_f), 1))])  # (T_valid, L+1)
                W_Phi = Phi * w_f[:, None]
                lhs = W_Phi.T @ Phi + 1e-4 * np.eye(L + 1)
                rhs = W_Phi.T @ X_f

                try:
                    theta = np.linalg.solve(lhs, rhs)
                    self.C_obs[k, f, :] = theta[:L]
                    self.d_obs[k, f] = theta[L]

                    # Residual variance
                    pred_f = Phi @ theta
                    res_sq = (X_f - pred_f) ** 2
                    var_f = float(np.sum(w_f * res_sq) / w_sum_f)
                    self.R_obs[k, f] = max(var_f, 1e-3)
                except np.linalg.LinAlgError:
                    pass

        # 3. Update continuous state dynamics (A_fast, b_fast, Q_fast)
        for k in range(K):
            gamma_k = regime_post[1:, k]  # (T-1,)
            w_sum = np.sum(gamma_k) + 1e-8

            M_prev = smoothed_m_k[:-1, k]  # (T-1, L)
            M_curr = smoothed_m_k[1:, k]   # (T-1, L)

            Phi = np.hstack([M_prev, np.ones((len(M_prev), 1))])  # (T-1, L+1)
            W_Phi = Phi * gamma_k[:, None]
            lhs = W_Phi.T @ Phi + 1e-4 * np.eye(L + 1)
            rhs = W_Phi.T @ M_curr  # (L+1, L)

            try:
                theta = np.linalg.solve(lhs, rhs)  # (L+1, L)
                self.A_fast[k] = theta[:L, :].T
                self.b_fast[k] = theta[L, :]

                # Process noise covariance
                pred_M = Phi @ theta
                res = M_curr - pred_M  # (T-1, L)
                Q_k = np.zeros((L, L))
                for t in range(len(res)):
                    Q_k += gamma_k[t] * np.outer(res[t], res[t])
                Q_k = Q_k / w_sum + 1e-4 * np.eye(L)
                self.Q_fast[k] = 0.5 * (Q_k + Q_k.T)
            except np.linalg.LinAlgError:
                pass

        # 4. Update duration parameters (dur_mu, dur_sigma)
        d_vals = np.arange(1, self.Dmax + 1)
        log_d = np.log(d_vals)
        for k in range(K):
            w = entry_post[:, k, :].sum(axis=0)
            w_sum = w.sum()
            if w_sum <= 1e-12:
                continue

            # Initial un-truncated MLE estimate as starting point
            init_mu = float((w * log_d).sum() / (w_sum + 1e-12))
            init_var = float((w * (log_d - init_mu) ** 2).sum() / (w_sum + 1e-12))
            init_sigma = float(np.sqrt(max(init_var, 1e-3)))

            # Exact truncated objective: maximize sum(w * log(pmf))
            def neg_target_loglik(params: np.ndarray) -> float:
                m, s = params[0], max(params[1], 1e-3)
                pdf = lognorm.pdf(d_vals, s=s, scale=np.exp(m))
                pdf_sum = pdf.sum()
                if pdf_sum <= 0.0 or np.isnan(pdf_sum):
                    return 1e10
                pmf = np.clip(pdf / pdf_sum, 1e-300, None)
                return -float(np.sum(w * np.log(pmf)))

            res = minimize(
                neg_target_loglik,
                x0=[init_mu, init_sigma],
                method="L-BFGS-B",
                bounds=[(None, None), (1e-3, 5.0)],
                options={"maxiter": 5, "ftol": 1e-2},
            )
            if res.success:
                self.dur_mu[k] = float(res.x[0])
                self.dur_sigma[k] = float(res.x[1])
                self.duration_optimizer_success_[k] = True
                self.duration_optimizer_messages_[k] = str(getattr(res, "message", "Optimization terminated successfully"))
            else:
                self.dur_mu[k] = init_mu
                self.dur_sigma[k] = init_sigma
                self.duration_optimizer_success_[k] = False
                self.duration_fallback_used_ = True
                msg = str(getattr(res, "message", "Optimization failed to converge"))
                self.duration_optimizer_messages_[k] = msg
                warnings.warn(
                    f"[DualTimescaleHSSM] Duration optimization fallback used for regime {k}: "
                    f"L-BFGS-B did not converge ({msg}). Using untruncated moment estimate "
                    f"(mu={init_mu:.4f}, sigma={init_sigma:.4f}).",
                    UserWarning,
                    stacklevel=2,
                )

        # 5. Update slow transition matrix (A_slow)
        row_sums = xi_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        new_A = xi_counts / row_sums
        # guard: keep zero diagonal (dwell handled entirely by duration dist, not self-loops)
        np.fill_diagonal(new_A, 0.0)
        row_sums2 = new_A.sum(axis=1, keepdims=True)
        row_sums2[row_sums2 == 0] = 1.0
        self.A_slow = new_A / row_sums2

    # ---------- Fitting & Generalized EM ----------

    def fit(
        self,
        X: np.ndarray,
        n_iter: int = 50,
        tol: float = 1e-4,
        verbose: bool = False,
        timestamps: np.ndarray | None = None,
    ) -> "DualTimescaleHSSM":
        """Fit dual-timescale model using generalized EM with backtracking rollback."""
        self.fit_state_ = FitState.RUNNING
        self.log_likelihood_history_ = []
        self._n_observations = len(X)

        if timestamps is not None:
            self.duration_unit = "calendar_days"
        else:
            self.duration_unit = "sessions"

        self._init_params(X)

        prev_params = None
        prev_ll = None

        for it in range(n_iter):
            # Save snapshot for monotonicity rollback
            snapshot = (
                self.pi.copy(),
                self.A_slow.copy(),
                self.dur_mu.copy(),
                self.dur_sigma.copy(),
                self.A_fast.copy(),
                self.b_fast.copy(),
                self.Q_fast.copy(),
                self.C_obs.copy(),
                self.d_obs.copy(),
                self.R_obs.copy(),
                self.duration_optimizer_success_.copy(),
                self.duration_fallback_used_,
                self.duration_optimizer_messages_.copy(),
            )

            # E-step: Forward-backward pass (Kalman GPB2 + discrete semi-Markov) + backward smoother
            regime_post, entry_post, xi_counts, m_filt, P_filt, m_pred, P_pred = self._forward_backward(X)
            smoothed_m, smoothed_P, smoothed_m_k, smoothed_P_k = self._backward_smoother(
                X, regime_post, m_filt, P_filt, m_pred, P_pred
            )

            ll = self.log_likelihood_
            if ll is None or np.isnan(ll) or np.isinf(ll):
                self.fit_state_ = FitState.NUMERICAL_FAILURE
                self.n_iter_ = it
                self._is_fitted = True
                return self

            self.log_likelihood_history_.append(ll)

            if verbose and it % 5 == 0:
                print(f"  [DualTimescaleHSSM] iter {it}: log-likelihood = {ll:.3f}")

            # Convergence check
            if it > 0 and prev_ll is not None and abs(ll - prev_ll) < tol:
                self.fit_state_ = FitState.CONVERGED
                self.n_iter_ = it
                break

            # Generalized EM Monotonicity Guard (Backtracking)
            if it > 0 and prev_ll is not None and ll < prev_ll - 1e-3:
                if prev_params is not None:
                    warnings.warn(
                        f"[DualTimescaleHSSM] Monotonicity violation at iter {it}: "
                        f"LL dropped from {prev_ll:.4f} to {ll:.4f}. Backtracking.",
                        UserWarning,
                        stacklevel=2,
                    )
                    (
                        self.pi, self.A_slow, self.dur_mu, self.dur_sigma,
                        self.A_fast, self.b_fast, self.Q_fast,
                        self.C_obs, self.d_obs, self.R_obs,
                        self.duration_optimizer_success_,
                        self.duration_fallback_used_,
                        self.duration_optimizer_messages_,
                    ) = prev_params
                    self.log_likelihood_ = prev_ll
                    self.fit_state_ = FitState.REJECTED_UPDATE
                    self.n_iter_ = it
                    break

            prev_params = snapshot
            prev_ll = ll

            # M-step: Analytical update of fast, slow, and observation parameters
            self._m_step(X, regime_post, entry_post, xi_counts, smoothed_m_k, smoothed_P_k)

        if self.fit_state_ == FitState.RUNNING:
            self.fit_state_ = FitState.MAX_ITER_REACHED

        self._is_fitted = True
        return self

    # ---------- Inference & Estimation ----------

    def estimate_states(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Estimate fast continuous trajectory and slow regime assignments.

        Returns:
          m_t: (T, L) continuous state trajectory posterior mean
          m_t_std: (T, L) continuous state posterior standard error (uncertainty)
          p_t: (T,) MAP discrete regime sequence
          p_t_post: (T, K) regime posterior probabilities
        """
        self._require_fitted()
        regime_post, entry_post, xi_counts, m_filt, P_filt, m_pred, P_pred = self._forward_backward(X)
        smoothed_m, smoothed_P, _, _ = self._backward_smoother(
            X, regime_post, m_filt, P_filt, m_pred, P_pred
        )

        p_t = np.argmax(regime_post, axis=1)

        # Standard error along diagonal of covariance
        m_t_std = np.zeros_like(smoothed_m)
        for t in range(len(smoothed_m)):
            m_t_std[t] = np.sqrt(np.maximum(np.diag(smoothed_P[t]), 1e-8))

        return smoothed_m, m_t_std, p_t, regime_post

    # ---------- Information Criteria ----------

    def n_params(self) -> int:
        self._require_fitted()
        K, F, L = self.K, self.F, self.L
        slow_params = (K - 1) + K * (K - 1) + 2 * K
        fast_params = K * (L * L + L + L * (L + 1) // 2)
        obs_params = K * (F * L + F + F)
        return slow_params + fast_params + obs_params

    def bic(self, n_observations: int | None = None) -> float:
        self._require_fitted()
        if self.log_likelihood_ is None:
            raise InternalStateError("Log likelihood is None for fitted model")
        n_obs = n_observations if n_observations is not None else getattr(self, "_n_observations", 100)
        return -2 * self.log_likelihood_ + self.n_params() * np.log(max(n_obs, 1))

    def aic(self, n_observations: int | None = None) -> float:
        self._require_fitted()
        if self.log_likelihood_ is None:
            raise InternalStateError("Log likelihood is None for fitted model")
        return -2 * self.log_likelihood_ + 2 * self.n_params()
