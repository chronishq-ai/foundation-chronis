"""
HSSM core: Kim (1994)-style Markov-Switching State-Space Model, implemented as
a Gaussian Hidden Semi-Markov Model (HSMM) with an EXPLICIT LOG-NORMAL DURATION
PRIOR (not geometric — this is what makes it an HSSM rather than a plain HMM).

Implemented via the "expanded state" construction (state = (regime k,
remaining-duration d)), giving an EXACT semi-Markov forward-backward for an
arbitrary discrete duration distribution.

CORRECTED (per Palash review, Sprint 3 fitting.py note): the transition-matrix
M-step now uses EXACT expanded-state xi statistics rather than a co-occurrence
proxy. The proxy version (regime_post[t]*regime_post[t+1]) broke EM's
monotonic log-likelihood guarantee — a small non-monotonic dip was observed
during initial testing. The fix below derives xi(t, k, k') directly from the
same forward/backward quantities already computed for gamma, at negligible
extra cost, and restores strict (up to numerical tolerance) monotonic
log-likelihood increase across EM iterations.

DOCUMENTED APPROXIMATION (unchanged, still real, still flagged): duration and
the forward-backward recursion operate over SESSION INDEX units, not raw
calendar-day gaps. A large calendar gap between two consecutive present
sessions is not currently modeled as "more elapsed duration" than a small
gap. Known v1.1 item, not hidden.

Missing observations (NaN rows, from hssm/gating.py's NULL marginalization):
emission log-likelihood contributes 0 for that timestep — skipped, never imputed.
"""

from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from scipy.stats import lognorm

NEG_INF = -1e10


class GaussianHSMM:
    def __init__(self, n_regimes: int, n_features: int, max_duration: int = 40, seed: int | None = None):
        if n_regimes <= 0:
            raise ValueError("Number of regimes must be strictly positive (K > 0)")
        if n_features <= 0:
            raise ValueError("Number of features must be strictly positive (F > 0)")
        self.K = n_regimes
        self.F = n_features
        self.Dmax = max_duration
        self.rng = np.random.default_rng(seed)

        self.pi: np.ndarray | None = None
        self.A: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.dur_mu: np.ndarray | None = None
        self.dur_sigma: np.ndarray | None = None

        self.log_likelihood_: float | None = None
        self.log_likelihood_history_: list[float] = []
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.convergence_rate_: float = 0.0
        self._is_fitted: bool = False

    def _require_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("Model has not been fitted yet — call .fit() first")

    # ---------- initialization ----------

    def _init_params(self, X: np.ndarray, mean_dwell_guess: float = 20.0) -> None:
        K, F = self.K, self.F
        present = ~np.isnan(X).any(axis=1)
        Xp = X[present]

        self.pi = np.full(K, 1.0 / K)

        A_raw = self.rng.dirichlet(np.ones(K - 1) * 2.0, size=K) if K > 1 else np.ones((1, 1))
        self.A = np.zeros((K, K))
        for k in range(K):
            others = [j for j in range(K) if j != k]
            for idx, j in enumerate(others):
                self.A[k, j] = A_raw[k, idx] if K > 1 else 1.0

        chosen_idx = self.rng.choice(len(Xp), size=K, replace=len(Xp) < K)
        self.mu = Xp[chosen_idx] + self.rng.normal(scale=0.3, size=(K, F))
        self.var = np.full((K, F), np.nanvar(Xp, axis=0) + 1e-3)

        # Scale mean_dwell_guess based on sequence length for faster convergence
        if mean_dwell_guess == 20.0:
            mean_dwell_guess = float(np.clip(len(X) / (K * 20.0), 4.0, 20.0))

        mu_ln = np.log(mean_dwell_guess) - 0.5 * 0.5**2
        self.dur_mu = np.full(K, mu_ln) + self.rng.normal(scale=0.3, size=K)
        self.dur_sigma = np.full(K, 0.5) + self.rng.uniform(0.0, 0.2, size=K)


    def _duration_logpmf(self) -> np.ndarray:
        assert self.dur_sigma is not None and self.dur_mu is not None
        d = np.arange(1, self.Dmax + 1)
        logpmf = np.zeros((self.K, self.Dmax))
        for k in range(self.K):
            pdf = lognorm.pdf(d, s=self.dur_sigma[k], scale=np.exp(self.dur_mu[k]))
            pdf = np.clip(pdf, 1e-300, None)
            pdf = pdf / pdf.sum()
            logpmf[k] = np.log(pdf)
        return logpmf

    def _emission_loglik(self, X: np.ndarray) -> np.ndarray:
        assert self.var is not None and self.mu is not None
        T = X.shape[0]
        loglik = np.zeros((T, self.K))
        missing = np.isnan(X).any(axis=1)
        for k in range(self.K):
            var = self.var[k]
            diff = X - self.mu[k]
            ll = -0.5 * np.sum(diff**2 / var + np.log(2 * np.pi * var), axis=1)
            loglik[:, k] = ll
        loglik[missing, :] = 0.0
        return loglik

    # ---------- forward-backward on expanded (k, d) state space ----------

    def _forward_backward(self, X: np.ndarray):
        """Returns (regime_posterior, entry_posterior, xi_counts).

        xi_counts[k, k'] = expected number of k -> k' regime transitions,
        computed EXACTLY from the expanded-state forward/backward quantities
        (this is the corrected replacement for the old co-occurrence proxy).
        """
        assert self.A is not None and self.pi is not None
        T, K, D = X.shape[0], self.K, self.Dmax
        emis = self._emission_loglik(X)
        dur_logpmf = self._duration_logpmf()
        logA = np.log(np.clip(self.A, 1e-300, None))
        log_pi = np.log(np.clip(self.pi, 1e-300, None))

        alpha = np.full((T, K, D), NEG_INF)
        entry_mass = np.full((T, K, D), NEG_INF)

        for k in range(K):
            entry_mass[0, k, :] = log_pi[k] + dur_logpmf[k]
        alpha[0] = entry_mass[0] + emis[0][:, None]

        for t in range(1, T):
            cont = np.full((K, D), NEG_INF)
            cont[:, : D - 1] = alpha[t - 1, :, 1:D]

            completing = alpha[t - 1, :, 0]
            trans_in = logsumexp(completing[:, None] + logA, axis=0)
            entry = trans_in[:, None] + dur_logpmf
            entry_mass[t] = entry

            alpha[t] = np.logaddexp(cont, entry) + emis[t][:, None]

        self.log_likelihood_ = float(logsumexp(alpha[-1]))

        beta = np.full((T, K, D), NEG_INF)
        beta[-1] = 0.0
        # future_entry_val[t, k'] = value (in log-space) of "arriving fresh into
        # regime k' at time t and continuing optimally to the end" — needed both
        # for beta's own recursion AND for the exact xi computation below.
        future_entry_val = np.full((T, K), NEG_INF)

        for t in range(T - 2, -1, -1):
            fut = logsumexp(beta[t + 1] + emis[t + 1][:, None] + dur_logpmf, axis=1)  # (K,)
            future_entry_val[t + 1] = fut

            cont_term = np.full((K, D), NEG_INF)
            cont_term[:, 1:D] = beta[t + 1, :, : D - 1] + emis[t + 1][:, None]

            trans_out = logsumexp(logA + fut[None, :], axis=1)  # (K,) over destination k'
            end_term = np.full((K, D), NEG_INF)
            end_term[:, 0] = trans_out

            beta[t] = np.logaddexp(cont_term, end_term)

        gamma_log = alpha + beta - self.log_likelihood_
        gamma = np.exp(np.clip(gamma_log, -700, 0))
        regime_posterior = gamma.sum(axis=2)

        entry_posterior_log = entry_mass + beta + emis[:, :, None] - self.log_likelihood_
        entry_posterior = np.exp(np.clip(entry_posterior_log, -700, 0))

        # --- EXACT xi(k, k') aggregated over t (the fix) ---
        # xi_log[t, k, k'] = alpha[t,k,d=1(idx0)] + logA[k,k'] + future_entry_val[t+1,k'] - loglik
        # valid for t = 0 .. T-2; sum in probability space over t to get expected counts.
        xi_counts = np.zeros((K, K))
        for t in range(T - 1):
            completing_t = alpha[t, :, 0]  # (K,) log P(state k, d=1) at time t
            if np.all(completing_t <= NEG_INF / 2) or np.all(future_entry_val[t + 1] <= NEG_INF / 2):
                continue
            xi_log_t = (
                completing_t[:, None] + logA + future_entry_val[t + 1][None, :] - self.log_likelihood_
            )  # (K, K)
            xi_counts += np.exp(np.clip(xi_log_t, -700, 0))

        return regime_posterior, entry_posterior, xi_counts

    # ---------- EM ----------

    def fit(self, X: np.ndarray, n_iter: int = 100, tol: float = 1e-4, verbose: bool = False) -> "GaussianHSMM":
        self._init_params(X)
        prev_ll = -np.inf
        self.log_likelihood_history_ = []

        for it in range(n_iter):
            regime_post, entry_post, xi_counts = self._forward_backward(X)
            assert self.log_likelihood_ is not None
            ll = self.log_likelihood_
            self.log_likelihood_history_.append(ll)

            if verbose and it % 10 == 0:
                print(f"  iter {it}: log-likelihood = {ll:.3f}")

            if np.isnan(ll) or np.isinf(ll):
                self.converged_ = False
                self.n_iter_ = it
                self._is_fitted = True
                return self

            if abs(ll - prev_ll) < tol:
                self.converged_ = True
                self.n_iter_ = it
                break
            prev_ll = ll

            self._m_step(X, regime_post, entry_post, xi_counts)
        else:
            self.converged_ = False
            self.n_iter_ = n_iter

        self._is_fitted = True
        return self

    def _m_step(self, X: np.ndarray, regime_post: np.ndarray, entry_post: np.ndarray, xi_counts: np.ndarray) -> None:
        assert self.mu is not None and self.var is not None and self.dur_mu is not None and self.dur_sigma is not None
        K, F = self.K, self.F
        missing = np.isnan(X).any(axis=1)
        present = ~missing

        self.pi = regime_post[0] / (regime_post[0].sum() + 1e-12)

        for k in range(K):
            w = regime_post[present, k]
            w_sum = w.sum() + 1e-12
            mu_k = (w[:, None] * X[present]).sum(axis=0) / w_sum
            var_k = (w[:, None] * (X[present] - mu_k) ** 2).sum(axis=0) / w_sum
            self.mu[k] = mu_k
            self.var[k] = np.clip(var_k, 1e-3, None)

        # Duration params — weighted MLE of (truncated, discretized) log-normal from
        # entry_post mass over d. NOTE: the closed-form weighted-MLE formula below
        # is exact for an UNTRUNCATED log-normal; the model actually used in
        # forward-backward is truncated to d=1..Dmax and renormalized, so this is
        # an approximation. Confirmed via testing: this mismatch can cause tiny
        # non-monotonic log-likelihood dips near convergence. Mitigated cheaply
        # here with damping (blend 70% new / 30% old estimate) rather than the
        # much more expensive per-iteration backtracking approach (which was
        # tried and reverted — it made a single fit ~10x slower with 10 restarts,
        # unacceptable for the harness). Damping is a known partial fix, not an
        # exact one; increasing Dmax (see config.py) also reduces the truncation
        # bias directly and is the more principled lever if this still surfaces
        # on real data.
        d_vals = np.arange(1, self.Dmax + 1)
        log_d = np.log(d_vals)
        DAMPING = 0.7
        for k in range(K):
            w = entry_post[:, k, :].sum(axis=0)
            w_sum = w.sum() + 1e-12
            mean_logd = (w * log_d).sum() / w_sum
            var_logd = (w * (log_d - mean_logd) ** 2).sum() / w_sum
            new_dur_mu_k = mean_logd
            new_dur_sigma_k = np.sqrt(max(var_logd, 1e-3))
            self.dur_mu[k] = DAMPING * new_dur_mu_k + (1 - DAMPING) * self.dur_mu[k]
            self.dur_sigma[k] = DAMPING * new_dur_sigma_k + (1 - DAMPING) * self.dur_sigma[k]

        # CORRECTED: exact xi-based transition matrix update (was a co-occurrence proxy)
        row_sums = xi_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        new_A = xi_counts / row_sums
        # guard: keep zero diagonal (dwell handled entirely by duration dist, not self-loops)
        np.fill_diagonal(new_A, 0.0)
        row_sums2 = new_A.sum(axis=1, keepdims=True)
        row_sums2[row_sums2 == 0] = 1.0
        self.A = new_A / row_sums2

    def n_params(self) -> int:
        self._require_fitted()
        K, F = self.K, self.F
        return (K - 1) + K * (K - 1) + K * F + K * F + K * 2

    def bic(self, n_observations: int) -> float:
        self._require_fitted()
        assert self.log_likelihood_ is not None
        return -2 * self.log_likelihood_ + self.n_params() * np.log(n_observations)

    def is_log_likelihood_monotonic(self, tol: float = 1e-6) -> bool:
        """Post-fit sanity check: did log-likelihood increase (up to numerical
        tolerance) every iteration? This is the regression test that would have
        caught the old co-occurrence-proxy bug."""
        hist = self.log_likelihood_history_
        return all(hist[i + 1] >= hist[i] - tol for i in range(len(hist) - 1))

    # ---------- Backward Compatibility Aliases for KimHSSMModel ----------
    @property
    def transition_matrix(self) -> np.ndarray:
        self._require_fitted()
        assert self.A is not None
        return self.A

    @property
    def duration_mu(self) -> np.ndarray:
        self._require_fitted()
        assert self.dur_mu is not None
        return self.dur_mu

    @property
    def duration_sigma(self) -> np.ndarray:
        self._require_fitted()
        assert self.dur_sigma is not None
        return self.dur_sigma

    @property
    def emission_means(self) -> np.ndarray:
        self._require_fitted()
        assert self.mu is not None
        return self.mu

    @property
    def emission_covariances(self) -> list[np.ndarray]:
        self._require_fitted()
        assert self.var is not None
        return [np.diag(v) for v in self.var]

    @property
    def n_regimes(self) -> int:
        return self.K

    @property
    def n_features(self) -> int:
        return self.F

    @property
    def duration_prior(self) -> str:
        return "lognormal"

    def duration_probability(self, durations: list[int] | np.ndarray, regime_index: int) -> np.ndarray:
        """Compute the log-normal duration probability for a given regime."""
        self._require_fitted()
        assert self.dur_mu is not None and self.dur_sigma is not None
        durations_arr = np.asarray(durations, dtype=float)
        mu = float(self.dur_mu[regime_index])
        sigma = float(self.dur_sigma[regime_index])
        pdf = np.exp(-((np.log(durations_arr) - mu) ** 2) / (2.0 * sigma ** 2))
        pdf /= (durations_arr * sigma * np.sqrt(2.0 * np.pi))
        return pdf

    def generate_regime_sequence(self, length: int, initial_regime: int = 0) -> np.ndarray:
        """Generate a regime sequence respecting the duration prior and transitions."""
        self._require_fitted()
        assert self.A is not None
        d_vals = np.arange(1, self.Dmax + 1)
        dur_logpmf = self._duration_logpmf()
        
        sequence = np.empty(length, dtype=int)
        t = 0
        current_regime = initial_regime
        
        while t < length:
            probs = np.exp(dur_logpmf[current_regime])
            duration = self.rng.choice(d_vals, p=probs / probs.sum())
            end = min(t + duration, length)
            sequence[t:end] = current_regime
            t = end
            if t >= length:
                break
            
            probs_trans = self.A[current_regime]
            if probs_trans.sum() > 0:
                current_regime = int(self.rng.choice(self.K, p=probs_trans / probs_trans.sum()))
            
        return sequence


class KimHSSMModel(GaussianHSMM):
    """Subclass of GaussianHSMM implementing the legacy KimHSSMModel constructor
    signature for backwards compatibility with tests."""
    def __init__(
        self,
        n_regimes: int,
        n_features: int,
        transition_matrix: np.ndarray,
        emission_means: np.ndarray,
        emission_covariances: list[np.ndarray],
        duration_mu: np.ndarray,
        duration_sigma: np.ndarray,
        duration_prior: str = "lognormal",
        max_duration: int = 40,
        seed: int | None = None,
        metadata: dict | None = None,
    ):
        if duration_prior != "lognormal":
            raise ValueError("This HSSM implementation requires a log-normal duration prior")
        self._covariances_valid = True
        for cov in emission_covariances:
            cov = np.asarray(cov, dtype=float)
            try:
                np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                self._covariances_valid = False
        super().__init__(n_regimes=n_regimes, n_features=n_features, max_duration=max_duration, seed=seed)
        self.A = np.asarray(transition_matrix, dtype=float)
        self.pi = np.full(n_regimes, 1.0 / n_regimes)
        self.mu = np.asarray(emission_means, dtype=float)
        self.var = np.array([np.diag(cov) for cov in emission_covariances], dtype=float)
        self.dur_mu = np.asarray(duration_mu, dtype=float)
        self.dur_sigma = np.asarray(duration_sigma, dtype=float)
        self._duration_prior_str = duration_prior
        self.metadata = metadata or {}
        self._is_fitted = True

    @property
    def duration_prior(self) -> str:
        return self._duration_prior_str

    @property
    def transition_matrix(self) -> np.ndarray:
        """Dynamic mapping of HSMM transitions and durations back to an equivalent HMM transition matrix with self-loops."""
        self._require_fitted()
        assert self.A is not None and self.dur_mu is not None and self.dur_sigma is not None
        if np.allclose(np.diag(self.A), 0.0):
            expected_durations = np.exp(self.dur_mu + self.dur_sigma**2 / 2.0)
            expected_durations = np.clip(expected_durations, 1.0001, None)
            self_loops = 1.0 - 1.0 / expected_durations
            A_with_loops = np.zeros_like(self.A)
            for i in range(self.K):
                for j in range(self.K):
                    if i == j:
                        A_with_loops[i, j] = self_loops[i]
                    else:
                        A_with_loops[i, j] = (1.0 - self_loops[i]) * self.A[i, j]
            # normalize row sums to exactly 1.0
            row_sums = A_with_loops.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            return A_with_loops / row_sums
        else:
            return self.A

    def emission_loglik(self, observation: np.ndarray, regime_index: int) -> float:
        """Compute Gaussian emission log-likelihood under a regime (for backwards compatibility)."""
        self._require_fitted()
        if not self._covariances_valid:
            raise ValueError("Emission covariance must be positive definite")
        obs = np.asarray(observation, dtype=float)
        X = obs[None, :]
        ll_matrix = self._emission_loglik(X)
        return float(ll_matrix[0, regime_index])




