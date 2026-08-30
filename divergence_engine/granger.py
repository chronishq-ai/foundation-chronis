"""
divergence_engine/granger.py

Sprint 8, Day 23 — Condition 2: within-regime Granger predictability, and the
hard 20-session-per-regime power gate (MP-09).

S79.1 FIX: this replaces the pre-segmented Bayesian VAR with a true Bayesian
Markov-Switching VAR (MS-VAR): hidden regime states z_t and regime-conditional
VAR coefficients (B_k, Sigma_k) are estimated JOINTLY via Gibbs sampling
(Hamilton filter forward pass + Kim multi-move backward sampling of z_t,
NIW-conjugate regime-conditional VAR draws, Dirichlet transition-matrix
draws). No regime pre-slicing anywhere in this file — the full (m_t, n_t)
session series is passed to the joint estimator, and regime membership is
inferred, not assumed.

IMPLEMENTATION NOTES (read before treating output as final):
  - Gibbs convergence is not verified automatically. Before trusting results
    on a new dataset/regime count, check trace plots / effective sample size
    on the retained B_k, Sigma_k, and z_t draws (not wired up here — add a
    diagnostics pass, e.g. via arviz on the raw draws, before production use).
  - Label switching: regime index k=0/1/... is not intrinsically meaningful
    across Gibbs iterations for a generic MS-VAR. This implementation avoids
    picking a specific k up front and instead lets `within_regime_granger_test`
    select "the regime containing the most recent session" per-iteration
    (see `target_regime="latest"`), which is invariant to label permutation
    draw-by-draw. If you need a different regime-selection semantics, do the
    selection per-draw, not by fixing an index.
  - With few effective sessions per regime, regime-conditional NIW posteriors
    fall back toward the prior (by construction) rather than becoming
    numerically unstable — but that also means MP-09's 20-session gate is
    necessary but not sufficient for a well-identified regime; interpret
    borderline cases (gate barely passed) cautiously.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np
from scipy.stats import invwishart

MIN_SESSIONS_PER_REGIME = 20  # MP-09: hard gate, no exceptions, no overrides.


# ======================================================================
# Result container
# ======================================================================
@dataclass(frozen=True)
class GrangerResult:
    ran: bool  # False if the power gate blocked the test entirely
    p_value_m_causes_n: Optional[float]
    p_value_n_causes_m: Optional[float]
    bonferroni_alpha: Optional[float]
    significant_m_causes_n: bool
    significant_n_causes_m: bool
    lag_order: Optional[int]
    n_regimes: Optional[int]
    n_sessions_in_regime: int  # effective (posterior-expected) count in target regime
    power_gate_passed: bool


def power_gate(n_sessions_in_regime: float) -> bool:
    """MP-09. Hard boolean. No soft score, ever."""
    return n_sessions_in_regime >= MIN_SESSIONS_PER_REGIME


# ======================================================================
# Bayesian Markov-Switching VAR (joint regime + VAR estimation, Gibbs)
# ======================================================================
@dataclass
class BayesianMarkovSwitchingVAR:
    """
    Bayesian MS-VAR(p) with K regimes. Each regime k has its own NIW-conjugate
    VAR(p) block (B_k, Sigma_k); regime membership z_t follows a first-order
    Markov chain with transition matrix P, estimated jointly with (B_k, Sigma_k)
    via Gibbs sampling:

      1. Given z, draw (B_k, Sigma_k) ~ NIW-posterior(regime-k rows of X, Y)
      2. Given (B_k, Sigma_k) for all k, run Hamilton forward filter to get
         P(z_t = k | Y_1:t, params) for all t
      3. Kim (1994) multi-move backward sampler: draw z_T ~ filtered dist,
         then for t = T-1..1 draw z_t ~ P(z_t | z_{t+1}, Y_1:t, params)
      4. Given z, draw each row of P from Dirichlet(prior_counts + transition_counts)

    Iterate; retain post-burn-in draws.
    """

    n_regimes: int
    p: int
    lambda_tightness: float = 0.2
    nu0_extra: int = 2
    include_const: bool = True
    n_iter: int = 500
    burn_in: int = 200
    dirichlet_prior_diag: float = 8.0   # prior stickiness for P's diagonal
    dirichlet_prior_offdiag: float = 1.0
    seed: Optional[int] = None

    # populated by fit()
    k_: int = field(init=False, default=0)
    n_regressors_: int = field(init=False, default=0)
    T_eff_: int = field(init=False, default=0)
    X_: Optional[np.ndarray] = field(init=False, default=None)
    Y_: Optional[np.ndarray] = field(init=False, default=None)
    B0_: Optional[np.ndarray] = field(init=False, default=None)
    Omega0_: Optional[np.ndarray] = field(init=False, default=None)
    Psi0_: Optional[np.ndarray] = field(init=False, default=None)
    nu0_: Optional[float] = field(init=False, default=None)

    # Gibbs draws retained post burn-in
    B_draws_: Optional[np.ndarray] = field(init=False, default=None)      # (n_kept, K, n_reg, k)
    Sigma_draws_: Optional[np.ndarray] = field(init=False, default=None)  # (n_kept, K, k, k)
    z_draws_: Optional[np.ndarray] = field(init=False, default=None)      # (n_kept, T_eff) int
    P_draws_: Optional[np.ndarray] = field(init=False, default=None)      # (n_kept, K, K)
    _fitted: bool = field(init=False, default=False)

    # ------------------------------------------------------------------
    # Design construction (regime-independent: X depends only on lags of Y)
    # ------------------------------------------------------------------
    @staticmethod
    def _as_2d(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        return a[:, None] if a.ndim == 1 else a

    def _build_joint_series(self, m_t: np.ndarray, n_t: np.ndarray) -> np.ndarray:
        m_t = self._as_2d(m_t)
        n_t = self._as_2d(n_t)
        if m_t.shape[0] != n_t.shape[0]:
            raise ValueError(
                f"m_t and n_t must share the session axis; got "
                f"{m_t.shape[0]} vs {n_t.shape[0]}"
            )
        self.d_m_ = m_t.shape[1]
        self.d_n_ = n_t.shape[1]
        return np.column_stack([m_t, n_t])  # (T, d_m + d_n), no averaging

    def _build_design(self, Y_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T, k = Y_full.shape
        p = self.p
        if T <= p:
            raise ValueError(f"Need more than p={p} sessions to form lags; got T={T}")
        Y = Y_full[p:, :]
        lag_blocks = [Y_full[p - i - 1 : T - i - 1, :] for i in range(p)]
        X_lags = np.column_stack(lag_blocks)
        X = np.column_stack([np.ones((Y.shape[0], 1)), X_lags]) if self.include_const else X_lags
        return X, Y

    def _default_prior(self, Y: np.ndarray) -> None:
        k = self.k_
        p = self.p
        n_reg = self.n_regressors_
        const_offset = 1 if self.include_const else 0

        own_var_proxy = np.var(Y, axis=0, ddof=1)
        own_var_proxy = np.where(own_var_proxy <= 1e-12, 1.0, own_var_proxy)

        B0 = np.zeros((n_reg, k))
        for j in range(k):
            B0[const_offset + j, j] = 1.0  # weak random-walk prior on own first lag

        diag_omega0 = np.zeros(n_reg)
        if self.include_const:
            diag_omega0[0] = 1e4
        lam = self.lambda_tightness
        for lag in range(1, p + 1):
            for series_j in range(k):
                row = const_offset + (lag - 1) * k + series_j
                own_scale = (lam / lag) ** 2
                cross_scales = [
                    (lam / lag) ** 2 * (own_var_proxy[oj] / own_var_proxy[series_j]) * 0.5
                    for oj in range(k) if oj != series_j
                ]
                diag_omega0[row] = max([own_scale] + cross_scales) if cross_scales else own_scale
        diag_omega0 = np.where(diag_omega0 <= 0, 1.0, diag_omega0)
        Omega0 = np.diag(diag_omega0)

        nu0 = k + 1 + self.nu0_extra
        Psi0 = np.diag(np.maximum(own_var_proxy * max(nu0 - k - 1, 1), 1e-8))

        self.B0_, self.Omega0_, self.Psi0_, self.nu0_ = B0, Omega0, Psi0, float(nu0)

    # ------------------------------------------------------------------
    # Regime-conditional NIW draw
    # ------------------------------------------------------------------
    def _draw_regime_niw(
        self, X_k: np.ndarray, Y_k: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Posterior NIW draw for one regime's rows. Falls back to the prior
        when a regime has too few assigned rows for a stable update (design
        remains proper: prior dominates, it never errors out)."""
        Omega0_inv = np.linalg.inv(self.Omega0_)

        if X_k.shape[0] == 0:
            Omega_n_inv = Omega0_inv
            Omega_n = self.Omega0_
            B_n = self.B0_
            Psi_n = self.Psi0_
            nu_n = self.nu0_
        else:
            XtX = X_k.T @ X_k
            XtY = X_k.T @ Y_k
            Omega_n_inv = Omega0_inv + XtX
            Omega_n = np.linalg.inv(Omega_n_inv)
            B_n = Omega_n @ (Omega0_inv @ self.B0_ + XtY)
            Psi_n = (
                self.Psi0_ + Y_k.T @ Y_k
                + self.B0_.T @ Omega0_inv @ self.B0_
                - B_n.T @ Omega_n_inv @ B_n
            )
            Psi_n = 0.5 * (Psi_n + Psi_n.T)
            eigvals, eigvecs = np.linalg.eigh(Psi_n)
            eigvals = np.clip(eigvals, 1e-10, None)
            Psi_n = eigvecs @ np.diag(eigvals) @ eigvecs.T
            nu_n = self.nu0_ + X_k.shape[0]
            Omega_n = 0.5 * (Omega_n + Omega_n.T)

        Sigma_k = invwishart.rvs(df=nu_n, scale=Psi_n, random_state=rng)
        Sigma_k = np.atleast_2d(Sigma_k)
        L_omega = np.linalg.cholesky(Omega_n)
        L_sigma = np.linalg.cholesky(Sigma_k)
        Z = rng.standard_normal(size=B_n.shape)
        B_k = B_n + L_omega @ Z @ L_sigma.T
        return B_k, Sigma_k

    # ------------------------------------------------------------------
    # Hamilton forward filter + Kim backward sampler
    # ------------------------------------------------------------------
    @staticmethod
    def _mvn_logpdf(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
        k = y.shape[0]
        diff = y - mean
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            cov = cov + np.eye(k) * 1e-8
            sign, logdet = np.linalg.slogdet(cov)
        solved = np.linalg.solve(cov, diff)
        quad = diff @ solved
        return -0.5 * (k * np.log(2 * np.pi) + logdet + quad)

    def _forward_filter(
        self, X: np.ndarray, Y: np.ndarray, B: np.ndarray, Sigma: np.ndarray, P: np.ndarray
    ) -> np.ndarray:
        """Returns filtered_prob, shape (T_eff, K): P(z_t=k | Y_1:t, params)."""
        T_eff = X.shape[0]
        K = self.n_regimes
        loglik = np.empty((T_eff, K))
        for t in range(T_eff):
            mean_t = X[t] @ B  # (K, k) via broadcasting below
        # vectorize properly: B is (K, n_reg, k)
        for t in range(T_eff):
            for k_idx in range(K):
                mean_tk = X[t] @ B[k_idx]
                loglik[t, k_idx] = self._mvn_logpdf(Y[t], mean_tk, Sigma[k_idx])

        filtered = np.empty((T_eff, K))
        pred = np.full(K, 1.0 / K)  # diffuse initial regime prior
        for t in range(T_eff):
            ll = loglik[t] - loglik[t].max()  # stabilize
            w = pred * np.exp(ll)
            w_sum = w.sum()
            if w_sum <= 0 or not np.isfinite(w_sum):
                w = pred.copy()
                w_sum = w.sum()
            filtered[t] = w / w_sum
            pred = filtered[t] @ P  # predict next step
        return filtered

    def _backward_sample(
        self, filtered: np.ndarray, P: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        T_eff, K = filtered.shape
        z = np.empty(T_eff, dtype=int)
        z[-1] = rng.choice(K, p=filtered[-1])
        for t in range(T_eff - 2, -1, -1):
            unnorm = filtered[t] * P[:, z[t + 1]]
            s = unnorm.sum()
            probs = unnorm / s if s > 0 else np.full(K, 1.0 / K)
            z[t] = rng.choice(K, p=probs)
        return z

    def _draw_transition_matrix(self, z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        K = self.n_regimes
        counts = np.zeros((K, K))
        for t in range(len(z) - 1):
            counts[z[t], z[t + 1]] += 1
        P = np.empty((K, K))
        for i in range(K):
            prior_row = np.full(K, self.dirichlet_prior_offdiag)
            prior_row[i] = self.dirichlet_prior_diag
            P[i] = rng.dirichlet(prior_row + counts[i])
        return P

    # ------------------------------------------------------------------
    # Fit (Gibbs sampler)
    # ------------------------------------------------------------------
    def fit(self, m_t: np.ndarray, n_t: np.ndarray) -> "BayesianMarkovSwitchingVAR":
        rng = np.random.default_rng(self.seed)
        Y_full = self._build_joint_series(m_t, n_t)
        T, k = Y_full.shape
        self.k_ = k
        self.n_regressors_ = k * self.p + (1 if self.include_const else 0)

        X, Y = self._build_design(Y_full)
        self.X_, self.Y_ = X, Y
        self.T_eff_ = X.shape[0]
        self._default_prior(Y)

        K = self.n_regimes
        n_reg = self.n_regressors_

        # Init: random regime assignment, uniform-ish transition matrix
        z = rng.integers(0, K, size=self.T_eff_)
        P = np.full((K, K), self.dirichlet_prior_offdiag)
        np.fill_diagonal(P, self.dirichlet_prior_diag)
        P = P / P.sum(axis=1, keepdims=True)

        total_iters = self.n_iter + self.burn_in
        kept_B, kept_Sigma, kept_z, kept_P = [], [], [], []

        for it in range(total_iters):
            B = np.empty((K, n_reg, k))
            Sigma = np.empty((K, k, k))
            for k_idx in range(K):
                mask = z == k_idx
                B[k_idx], Sigma[k_idx] = self._draw_regime_niw(X[mask], Y[mask], rng)

            filtered = self._forward_filter(X, Y, B, Sigma, P)
            z = self._backward_sample(filtered, P, rng)
            P = self._draw_transition_matrix(z, rng)

            if it >= self.burn_in:
                kept_B.append(B)
                kept_Sigma.append(Sigma)
                kept_z.append(z.copy())
                kept_P.append(P)

        self.B_draws_ = np.array(kept_B)          # (n_kept, K, n_reg, k)
        self.Sigma_draws_ = np.array(kept_Sigma)  # (n_kept, K, k, k)
        self.z_draws_ = np.array(kept_z)          # (n_kept, T_eff)
        self.P_draws_ = np.array(kept_P)          # (n_kept, K, K)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Regime-targeted posterior extraction
    # ------------------------------------------------------------------
    def target_regime_draws(
        self, target: Literal["latest", "largest"] = "latest"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        For each kept Gibbs draw, pick the regime label matching `target`
        (per-draw selection avoids label-switching bias across draws):
          - "latest":  regime occupied at the final session (z[-1])
          - "largest": regime with the most assigned sessions in that draw

        Returns
        -------
        B_target   : (n_kept, n_reg, k)
        Sigma_target : (n_kept, k, k)
        expected_n_sessions : (n_kept,) count of sessions assigned to the
            selected regime in that draw (mean of this = effective n for MP-09)
        """
        if not self._fitted:
            raise RuntimeError("Call fit(m_t, n_t) before target_regime_draws().")
        n_kept = self.z_draws_.shape[0]
        B_target = np.empty((n_kept, self.n_regressors_, self.k_))
        Sigma_target = np.empty((n_kept, self.k_, self.k_))
        n_sessions = np.empty(n_kept)

        for i in range(n_kept):
            z_i = self.z_draws_[i]
            if target == "latest":
                k_sel = z_i[-1]
            else:  # "largest"
                k_sel = np.bincount(z_i, minlength=self.n_regimes).argmax()
            B_target[i] = self.B_draws_[i, k_sel]
            Sigma_target[i] = self.Sigma_draws_[i, k_sel]
            n_sessions[i] = np.sum(z_i == k_sel)

        return B_target, Sigma_target, n_sessions


# ======================================================================
# Directional ROPE test + top-level Granger entry point
# ======================================================================
def _causing_block_indices(
    p: int, k: int, include_const: bool, causing_cols: range, target_cols: range
) -> tuple[np.ndarray, np.ndarray]:
    const_offset = 1 if include_const else 0
    rows = []
    for lag in range(1, p + 1):
        base = const_offset + (lag - 1) * k
        rows.extend(base + c for c in causing_cols)
    return np.array(rows, dtype=int), np.array(list(target_cols), dtype=int)


def _directional_rope_mass(
    B_samples: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    rope_half_width: float,
) -> float:
    """
    Fraction of posterior (regime-targeted) B draws for which EVERY
    coefficient in the causing block lies within the ROPE. Low mass =>
    posterior excludes zero for the block => evidence FOR directional
    causality. This is a Bayesian ROPE analog, not a frequentist p-value —
    see module notes above for how it's used against `bonferroni_alpha`.
    """
    block = B_samples[:, row_idx[:, None], col_idx[None, :]]
    within_rope = np.all(np.abs(block) <= rope_half_width, axis=(1, 2))
    return float(np.mean(within_rope))


def within_regime_granger_test(
    m_t: np.ndarray,
    n_t: np.ndarray,
    n_pairs_tested: int,
    n_regimes: int = 2,
    max_lag: int = 5,
    alpha: float = 0.05,
    n_iter: int = 500,
    burn_in: int = 200,
    target_regime: Literal["latest", "largest"] = "latest",
    rope_fraction_of_std: float = 0.05,
    random_state: Optional[int] = None,
) -> GrangerResult:
    """
    m_t: behavioral fast state, shape (T, d1) — FULL session series, not
         pre-sliced to a regime window. Regime membership is estimated
         jointly with the VAR coefficients by BayesianMarkovSwitchingVAR.
    n_t: narrative fast state, shape (T, d2) — same session alignment.

    MP-09 is enforced against the *effective* session count of the target
    regime: the posterior-mean number of sessions the Gibbs sampler assigns
    to that regime across kept draws. If that falls below 20, the test does
    not run (ran=False) — same hard-gate semantics as before, just computed
    from the joint model's regime assignments instead of a pre-sliced array
    length.

    Bonferroni: alpha / (2 * n_pairs_tested), covering the two directional
    sub-tests (m->n, n->m) nested within each tested pair.
    """
    if len(m_t) != len(n_t):
        raise ValueError("m_t and n_t must be aligned to the same session grid")

    T = len(m_t)
    max_lag_allowed = min(max_lag, T // 3 - 1)
    lag_order = max(1, max_lag_allowed)

    ms_var = BayesianMarkovSwitchingVAR(
        n_regimes=n_regimes,
        p=lag_order,
        n_iter=n_iter,
        burn_in=burn_in,
        seed=random_state,
    )
    ms_var.fit(m_t, n_t)

    B_target, _Sigma_target, n_sessions_draws = ms_var.target_regime_draws(target=target_regime)
    effective_n_sessions = float(np.mean(n_sessions_draws))

    gate_passed = power_gate(effective_n_sessions)
    if not gate_passed:
        return GrangerResult(
            ran=False,
            p_value_m_causes_n=None,
            p_value_n_causes_m=None,
            bonferroni_alpha=None,
            significant_m_causes_n=False,
            significant_n_causes_m=False,
            lag_order=lag_order,
            n_regimes=n_regimes,
            n_sessions_in_regime=int(round(effective_n_sessions)),
            power_gate_passed=False,
        )

    d_m = ms_var._as_2d(m_t).shape[1]
    d_n = ms_var._as_2d(n_t).shape[1]
    k = ms_var.k_
    m_cols, n_cols = range(0, d_m), range(d_m, k)

    row_m_to_n, col_m_to_n = _causing_block_indices(
        lag_order, k, ms_var.include_const, causing_cols=m_cols, target_cols=n_cols
    )
    row_n_to_m, col_n_to_m = _causing_block_indices(
        lag_order, k, ms_var.include_const, causing_cols=n_cols, target_cols=m_cols
    )

    n_std = np.std(ms_var._as_2d(n_t), axis=0).mean()
    m_std = np.std(ms_var._as_2d(m_t), axis=0).mean()
    rope_m_to_n = rope_fraction_of_std * (n_std if n_std > 1e-12 else 1.0)
    rope_n_to_m = rope_fraction_of_std * (m_std if m_std > 1e-12 else 1.0)

    p_value_m_causes_n = _directional_rope_mass(B_target, row_m_to_n, col_m_to_n, rope_m_to_n)
    p_value_n_causes_m = _directional_rope_mass(B_target, row_n_to_m, col_n_to_m, rope_n_to_m)

    bonferroni_alpha = alpha / (2 * max(1, n_pairs_tested))

    return GrangerResult(
        ran=True,
        p_value_m_causes_n=p_value_m_causes_n,
        p_value_n_causes_m=p_value_n_causes_m,
        bonferroni_alpha=bonferroni_alpha,
        significant_m_causes_n=bool(p_value_m_causes_n < bonferroni_alpha),
        significant_n_causes_m=bool(p_value_n_causes_m < bonferroni_alpha),
        lag_order=lag_order,
        n_regimes=n_regimes,
        n_sessions_in_regime=int(round(effective_n_sessions)),
        power_gate_passed=True,
    )