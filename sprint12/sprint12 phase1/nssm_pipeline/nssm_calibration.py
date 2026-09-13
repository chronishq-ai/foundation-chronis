"""
CHRONIS — Team 4 (INVENTORS) — Sprint 7, Day 20
Person-Specific Calibration & NSSM Fitting

WHAT THIS FILE DOES, IN PLAIN ENGLISH
--------------------------------------
Day 19 gave us, per session, a soft class distribution + sigma_t (a
measurement-uncertainty number) for each of the 8 narrative dimensions.
Day 20 turns that into something the rest of the system can actually build
on:

  1. IDIOLECT NORMALIZATION — before we trust any raw lexical/prosodic
     number, we re-express it relative to THIS person's own rolling
     baseline, not some population average. Someone who always talks fast
     isn't "anxious" every session just because they talk fast; they're
     only unusual relative to their OWN normal.

  2. CONFORMAL CALIBRATION — Day 19's label model gives us a soft
     probability distribution, but "the model says 70% confident" is not
     automatically trustworthy — label models can be overconfident or
     underconfident. Conformal prediction takes a small number of
     person-specific "gold checks" (real corrections the person made via
     "Teach Chronis") and uses them to build a PREDICTION SET (e.g. "the
     true class is one of {agentive_hero, observer}") that is guaranteed,
     on average, to contain the true answer at least X% of the time (e.g.
     90%). This is a real, provable statistical guarantee — not a vibe.

  3. THE NSSM (Narrative State-Space Model) — this is the actual regime
     model for someone's self-story: a slow discrete "narrative regime"
     q_t (which chapter of the story are we in?) with a LOG-NORMAL
     duration prior (stories don't switch chapters on a fixed geometric
     schedule — they tend to run for a while and then meaningfully shift),
     plus a fast continuous narrative state n_t (session-to-session
     wobble within a chapter). This is the exact same model *class* as
     System A's HSSM (Sprint 3), so the two are finally comparable —
     that's the whole point of Sprint 7.

  4. FITTING HARNESS — we fit the NSSM the same disciplined way Sprint 3
     fit the HSSM: >=10 random restarts, keep the highest log-likelihood
     run, and pick the number of narrative regimes J via BIC over
     {2, 3, 4} rather than eyeballing it.

WHERE THIS PLUGS INTO THE REST OF THE PROGRAM
-----------------------------------------------
- Upstream: this file imports Day 19's WeakSupervisionLabelLayer output
  (soft distributions + sigma_t) as its raw material.
- [REQUIRES SPRINT 2 Z-SCORING UTILITIES]: idiolect normalization is
  supposed to reuse FOUNDRY's (Sprint 1 Day 3) per-person rolling
  z-scoring utility directly, not reimplement it. That utility isn't
  importable from this codebase yet, so this file ships an equivalent
  standalone implementation, clearly marked, to unblock Sprint 7 — swap
  it for the real import the moment it exists.
- [REQUIRES SPRINT 3 KIM FILTER]: BACKBONE (Sprint 3) already built a
  production Kim (1994) filter + EM fitting harness for the behavioral
  HSSM. The NSSM is supposed to be "identical model class to System A."
  In production this file should import and reuse that engine rather
  than maintaining a second copy. Until that import path exists, this
  file ships its own duration-augmented Kim filter (same math, same
  fitting discipline) so Sprint 7 isn't blocked on Sprint 3's internals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import lognorm

from nssm_pipeline.weak_supervision_label_layer import (
    ABSTAIN,
    DIMENSION_CLASSES,
    DimensionOutput,
    NarrativeDimension,
    SessionInput,
    WeakSupervisionLabelLayer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chronis.nssm")


# ===========================================================================
# STEP 0 — Turn Day 19's per-dimension soft distributions into a scalar
# observation vector the NSSM can fit on.
# ===========================================================================
# SIMPLIFICATION, STATED HONESTLY: the NSSM below models a single
# continuous fast state n_t that loads onto all 8 dimensions. To make each
# dimension a single real-valued "channel" for that loading, we collapse
# each dimension's soft class distribution into one scalar via a fixed
# ordinal encoding (expected value over each class's assigned position on
# a [-1, +1] axis). This is a real modeling choice, not a hidden one —
# flagging it here so the team can revisit it if the Bible wants a richer,
# multi-dimensional narrative emission model later.
ORDINAL_ENCODING: Dict[NarrativeDimension, Dict[str, float]] = {
    NarrativeDimension.SEMANTIC_DOMAIN_COVERAGE: {"career": -1.0, "relationships": -0.33, "health": 0.33, "other": 1.0},
    NarrativeDimension.CAUSAL_ATTRIBUTION: {"external_locus": -1.0, "mixed": 0.0, "internal_locus": 1.0},
    NarrativeDimension.SELF_ROLE: {"victim": -1.0, "observer": 0.0, "agentive_hero": 1.0},
    NarrativeDimension.TEMPORAL_FRAMING: {"past_dominant": -1.0, "present_dominant": 0.0, "future_dominant": 1.0},
    NarrativeDimension.MORAL_FRAMING: {"neutral": -1.0, "loyalty_betrayal": -0.33, "fairness_cheating": 0.33, "care_harm": 1.0},
    NarrativeDimension.NARRATIVE_ARC_TYPING: {"contaminated": -1.0, "ambivalent": -0.33, "stable": 0.33, "redemptive": 1.0},
    NarrativeDimension.CONTRADICTION_TOLERANCE: {"low_tolerance": -1.0, "high_tolerance": 1.0},
    NarrativeDimension.FUTURE_SELF_REHEARSAL: {"absent": -1.0, "inconsistent": 0.0, "consistent": 1.0},
}

NARRATIVE_DIMENSIONS_ORDERED: List[NarrativeDimension] = list(NarrativeDimension)
F_DIM = len(NARRATIVE_DIMENSIONS_ORDERED)  # 8 observation channels


def dimension_outputs_to_observation(
    dim_outputs: Dict[str, DimensionOutput],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert one session's {dimension_name: DimensionOutput} (Day 19 output)
    into:
      obs   -> shape (F_DIM,), the expected ordinal value per dimension
      sigma -> shape (F_DIM,), Day 19's learned measurement uncertainty
               per dimension (used directly as the heteroskedastic
               emission noise term in the NSSM below — never a constant).
    """
    obs = np.zeros(F_DIM)
    sigma = np.zeros(F_DIM)
    for f, dim in enumerate(NARRATIVE_DIMENSIONS_ORDERED):
        out = dim_outputs[dim.value]
        classes = DIMENSION_CLASSES[dim]
        ordinal_values = np.array([ORDINAL_ENCODING[dim][c] for c in classes])
        obs[f] = float(np.dot(out.distribution, ordinal_values))  # expected value
        sigma[f] = out.sigma_t
    return obs, sigma


# ===========================================================================
# STEP 1 — Idiolect normalization (person-specific rolling z-scoring).
# ===========================================================================
@dataclass
class IdiolectNormalizer:
    """
    Person-specific lexical/prosodic baselines, computed BEFORE any
    labeling function runs (per the directive, this sits upstream of
    Day 19's WSL in the real pipeline — it's included in this file because
    that's where Sprint 7's spec asked for it to be built).

    [REQUIRES SPRINT 2 Z-SCORING UTILITIES]
    Sprint 1 Day 3 (FOUNDRY) already built "per-person z-scoring utilities
    under one hard rule: never normalize across the population — only
    against a person's own rolling baseline." This class should import
    and call that utility directly. It isn't importable from this
    codebase yet, so the method below is a same-contract standalone
    implementation — swap the body of `rolling_zscore` for the real
    FOUNDRY import as soon as it's wired in; keep the method signature
    the same so nothing downstream has to change.
    """
    window_sessions: int = 30
    min_sessions_for_baseline: int = 5

    def rolling_zscore(self, values: np.ndarray) -> np.ndarray:
        """
        values: shape (T,) — one person's raw values for ONE feature,
        in session order.

        Returns: shape (T,) — each value expressed as
            (value - rolling_mean_of_own_past_values) / rolling_std
        using only that person's own trailing window (never population
        stats, never future sessions — that would leak information).

        Beginner note: for the first `min_sessions_for_baseline` sessions
        there isn't enough history to compute a trustworthy baseline yet,
        so we return NaN there rather than a fabricated z-score. Silence
        is a valid, correct output — this mirrors the Global Standard's
        "NULL/missing states are typed, never imputed" rule.
        """
        # [REQUIRES SPRINT 2 Z-SCORING UTILITIES] — replace this body with:
        #   from chronis_foundry.zscoring import rolling_zscore as foundry_zscore
        #   return foundry_zscore(values, window=self.window_sessions)
        t = len(values)
        z = np.full(t, np.nan)
        for i in range(t):
            window_start = max(0, i - self.window_sessions)
            history = values[window_start:i]  # strictly PAST values only
            if len(history) < self.min_sessions_for_baseline:
                continue
            mu = np.mean(history)
            sd = np.std(history)
            if sd < 1e-8:
                continue  # degenerate baseline (no variance yet) -> stay NaN, don't divide by ~0
            z[i] = (values[i] - mu) / sd
        return z

    def normalize_session_series(self, raw_observations: np.ndarray) -> np.ndarray:
        """
        raw_observations: shape (T, F) — one person's raw per-dimension
        observation series across sessions.

        Returns: shape (T, F) — each column independently rolling
        z-scored against that person's own history in that dimension.
        """
        t, f = raw_observations.shape
        normalized = np.zeros_like(raw_observations)
        for col in range(f):
            normalized[:, col] = self.rolling_zscore(raw_observations[:, col])
        return normalized


# ===========================================================================
# STEP 2 — Conformal calibration of Day 19's soft scores against sparse
# per-person gold checks.
# ===========================================================================
@dataclass
class GoldCheck:
    """One sparse, real ground-truth data point — e.g. a person confirming
    via 'Teach Chronis' (Sprint 17) that a given session really was
    'agentive_hero' for self_role. These are rare by nature; conformal
    calibration is specifically chosen because it doesn't need many of
    them to produce a valid guarantee."""
    session_id: str
    dimension: NarrativeDimension
    true_class: str
    soft_distribution: np.ndarray  # Day 19's predicted distribution for this session/dimension


@dataclass
class CalibratedPredictionSet:
    session_id: str
    dimension: str
    prediction_set: List[str]
    qhat: float
    target_coverage: float


class ConformalCalibrator:
    """
    Implements SPLIT CONFORMAL PREDICTION (Vovk et al.), the simplest form
    that still gives a real, provable guarantee.

    BEGINNER WALKTHROUGH OF THE MATH
    ---------------------------------
    1. For every gold-check session, compute a "nonconformity score":
           s = 1 - P_hat(true_class | session)
       This is just "how wrong was the model's confidence in the actual
       right answer?" A score near 0 means the model was confident AND
       correct. A score near 1 means the model was blindsided.

    2. Sort all these scores from the calibration set. Pick the
       ceil((n+1)*(1-alpha))/n empirical quantile of that sorted list —
       call it qhat. This one number, `qhat`, is the calibration.

    3. For any NEW session, build the prediction set:
           { class c : 1 - P_hat(c | session) <= qhat }
       i.e. keep every class the model didn't rule out "more confidently
       than qhat allows."

    WHY THIS IS A REAL GUARANTEE, NOT A VIBE
    ------------------------------------------
    Split-conformal prediction guarantees MARGINAL coverage: averaged
    across the calibration distribution, the true class falls inside the
    prediction set at least (1 - alpha) of the time (e.g. 90% for
    alpha=0.10). This is a population-level, finite-sample guarantee. It
    is NOT a promise about any single prediction ("this specific session's
    set has a 90% chance of containing the truth" is the WRONG reading —
    state the marginal version plainly, per the directive).

    SPARSE PER-PERSON GOLD CHECKS
    -------------------------------
    Because "Teach Chronis" corrections are rare, we may not have enough
    gold checks for a given person+dimension to calibrate per-person. If
    we don't, we fall back to a population-level calibration set (still a
    valid marginal guarantee, just averaged over more people, not just
    this one) and we say so explicitly in the returned metadata rather
    than silently pretending it's person-specific.
    """

    def __init__(self, alpha: float = 0.10, min_gold_checks_for_personal_calibration: int = 8):
        self.alpha = alpha
        self.target_coverage = 1.0 - alpha
        self.min_gold_checks_for_personal_calibration = min_gold_checks_for_personal_calibration

    def _nonconformity_scores(self, gold_checks: List[GoldCheck], dimension: NarrativeDimension) -> np.ndarray:
        classes = DIMENSION_CLASSES[dimension]
        scores = []
        for gc in gold_checks:
            if gc.dimension != dimension:
                continue
            true_idx = classes.index(gc.true_class)
            p_true = gc.soft_distribution[true_idx]
            scores.append(1.0 - p_true)
        return np.array(scores)

    def calibrate(self, gold_checks: List[GoldCheck], dimension: NarrativeDimension) -> Tuple[float, bool]:
        """
        Returns (qhat, was_person_specific).

        was_person_specific=False signals we fell back to a broader
        calibration pool because this person didn't have enough gold
        checks yet for this dimension — callers must propagate that
        honestly (e.g. into product copy / confidence-set metadata), not
        hide it.
        """
        scores = self._nonconformity_scores(gold_checks, dimension)
        n = len(scores)
        was_person_specific = n >= self.min_gold_checks_for_personal_calibration

        if n == 0:
            # No gold checks at all yet for this dimension -> cannot
            # calibrate. Return the most conservative possible qhat
            # (accept every class) rather than pretending we calibrated.
            logger.warning("No gold checks available for %s; returning maximally conservative qhat=1.0", dimension.value)
            return 1.0, False

        # The classic split-conformal quantile: ceil((n+1)*(1-alpha))/n,
        # clipped to 1.0 in the small-sample edge case.
        q_level = min(1.0, np.ceil((n + 1) * self.target_coverage) / n)
        qhat = float(np.quantile(scores, q_level, method="higher"))
        return qhat, was_person_specific

    def predict_set(
        self, soft_distribution: np.ndarray, qhat: float, dimension: NarrativeDimension,
        session_id: str,
    ) -> CalibratedPredictionSet:
        classes = DIMENSION_CLASSES[dimension]
        keep = [c for c, p in zip(classes, soft_distribution) if (1.0 - p) <= qhat]
        if not keep:
            # Numerical edge case: never return an empty set — that would
            # violate the coverage guarantee outright. Fall back to the
            # single most likely class.
            keep = [classes[int(np.argmax(soft_distribution))]]
        return CalibratedPredictionSet(
            session_id=session_id,
            dimension=dimension.value,
            prediction_set=keep,
            qhat=qhat,
            target_coverage=self.target_coverage,
        )


# ===========================================================================
# STEP 3 — Log-normal duration prior (what makes this an HSSM, not an HMM).
# ===========================================================================
class LogNormalDurationPrior:
    """
    Per-regime distribution over "how many sessions does this narrative
    chapter tend to last?" A plain HMM implicitly assumes GEOMETRIC
    duration (constant per-step chance of switching, i.e. "memoryless" —
    equally likely to end on day 1 of a chapter as day 30). Real narrative
    chapters aren't like that: they have inertia early on and become more
    (or less) likely to end as they mature. A log-normal duration prior
    captures that directly.
    """

    def __init__(self, mu: float, sigma: float):
        self.mu = mu       # mean of the underlying normal (in log-duration space)
        self.sigma = sigma  # std of the underlying normal
        self._dist = lognorm(s=max(sigma, 1e-3), scale=np.exp(mu))

    def pdf(self, d: int) -> float:
        return float(self._dist.pdf(d))

    def survival(self, d: int) -> float:
        """P(duration > d) — the chance a chapter that has already lasted
        d sessions keeps going past d."""
        if d <= 0:
            return 1.0
        return float(self._dist.sf(d))

    def hazard(self, d: int) -> float:
        """
        The HAZARD is the beginner-friendly heart of this whole idea:
        "given a chapter has already lasted d-1 sessions, what's the
        chance it ends on session d?" That's exactly
            hazard(d) = pdf(d) / survival(d-1)
        A log-normal's hazard function is NOT constant over d (unlike a
        geometric distribution's), which is precisely the behavior the
        directive requires.
        """
        s_prev = self.survival(d - 1)
        if s_prev < 1e-10:
            return 1.0  # essentially certain to have ended by now
        h = self.pdf(d) / s_prev
        return float(np.clip(h, 0.0, 1.0))


# ===========================================================================
# STEP 4 — Duration-augmented transition matrix.
# ===========================================================================
# We turn the semi-Markov (regime, duration) process into an ordinary,
# larger Markov chain over augmented states z = (regime j, duration d),
# d capped at D_MAX. This "duration augmentation" trick is a standard way
# to make a duration-dependent process tractable with an ordinary
# Kim-style filter, while still genuinely encoding the log-normal hazard
# (not a geometric one) in every transition probability.
def build_augmented_transition_matrix(
    j_count: int,
    d_max: int,
    duration_priors: List[LogNormalDurationPrior],
    cross_regime_transitions: np.ndarray,  # shape (J, J), zero diagonal, rows sum to 1
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    s_count = j_count * d_max
    index_of: Dict[Tuple[int, int], int] = {}
    idx = 0
    for j in range(j_count):
        for d in range(1, d_max + 1):
            index_of[(j, d)] = idx
            idx += 1

    trans = np.zeros((s_count, s_count))
    for j in range(j_count):
        for d in range(1, d_max + 1):
            row = index_of[(j, d)]
            hazard = 1.0 if d == d_max else duration_priors[j].hazard(d)
            stay_prob = 1.0 - hazard
            if d < d_max and stay_prob > 0:
                trans[row, index_of[(j, d + 1)]] = stay_prob
            for k in range(j_count):
                if k == j:
                    continue
                switch_prob = hazard * cross_regime_transitions[j, k]
                if switch_prob > 0:
                    trans[row, index_of[(k, 1)]] += switch_prob
            # Numerical safety: renormalize the row so it sums to exactly 1
            # (guards against floating-point drift in the hazard/survival
            # computation, never silently redistributes real probability
            # mass).
            row_sum = trans[row].sum()
            if row_sum > 0:
                trans[row] /= row_sum
    return trans, index_of


# ===========================================================================
# STEP 5 — The duration-augmented Kim (1994) filter.
# ===========================================================================
# [REQUIRES SPRINT 3 KIM FILTER]
# BACKBONE (Team 2, Sprint 3) already implements the production
# Kim-filter fitting engine for System A's HSSM via statsmodels, with the
# exact fitting discipline (>=10 random inits, highest-log-likelihood
# selection) this file also needs. The NSSM is explicitly supposed to be
# "identical model class to System A," so in production this class should
# be REUSED from Sprint 3's module, not reimplemented here. That import
# path isn't available in this codebase yet, so this class is a
# standalone equivalent (same math: predict -> update -> Kim collapsing)
# so Sprint 7 can be built and tested without waiting on Sprint 3's
# internals to be exposed as a shared library. Replace this class with
# `from chronis_backbone.kim_filter import KimFilter` the moment that's
# possible, and delete the duration-augmentation wrapper here in favor of
# whatever generic hook Sprint 3 exposes for a custom transition matrix.
class DurationAugmentedKimFilter:
    """
    Runs the Kim (1994) approximate filter over the duration-augmented
    state space built in Step 4. The "fast" continuous narrative state
    n_t is scalar; it loads onto all F=8 narrative-dimension observation
    channels via a per-regime loading vector C_j (this is what lets one
    latent "chapter intensity" number explain movement across all 8
    dimensions at once).

    BEGINNER WALKTHROUGH OF ONE FILTER STEP
    ------------------------------------------
    At each session t, for every PAIR of (previous augmented state i,
    current augmented state j):
      1. PREDICT where the continuous state would be if we were in
         pair (i, j): n_pred = a_j * n_prev[i]; grow the uncertainty by
         the regime's process noise q_j.
      2. Compare that prediction to what we actually observed (y_t) to
         get an "innovation" (prediction error) and how likely that
         error was under pair (i, j)'s implied noise model — this
         likelihood is the vote that pair (i, j) casts for "was this the
         right regime path?"
      3. UPDATE the continuous state estimate using that observation
         (a standard Kalman update, done in closed form here because a
         single scalar state observed through several independent-noise
         channels combines very cleanly — no big matrix inversion
         needed).
      4. COLLAPSE: Kim's filter would otherwise need to track every
         pair (i, j) forever, which blows up exponentially with t. The
         "collapsing" step approximates all the (i, j) pairs that ended
         up in the same current state j as a single Gaussian (mean +
         variance), which keeps the filter's cost constant over time.
    """

    def __init__(self, j_count: int, d_max: int, f_dim: int, diffuse_prior_var: float = 10.0):
        self.j_count = j_count
        self.d_max = d_max
        self.f_dim = f_dim
        self.s_count = j_count * d_max
        self.diffuse_prior_var = diffuse_prior_var

    def _regime_of_state(self) -> np.ndarray:
        """augmented-state index -> base regime index, e.g. [0,0,0,1,1,1,...]"""
        return np.repeat(np.arange(self.j_count), self.d_max)

    def run(
        self,
        observations: np.ndarray,      # shape (T, F)
        emission_noise_var: np.ndarray,  # shape (T, F) — Day 19's sigma_t**2, heteroskedastic
        a: np.ndarray,                  # shape (J,) AR coefficient per regime
        q: np.ndarray,                  # shape (J,) process noise variance per regime
        c: np.ndarray,                  # shape (J, F) emission loading per regime
        transition_matrix: np.ndarray,  # shape (S, S), from build_augmented_transition_matrix
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Returns:
          total_loglik: float — sum over t of log p(y_t | y_1..y_{t-1})
          regime_probs: shape (T, J) — filtered probability of each BASE
                        regime at each t (duration marginalized out)
          filtered_state: shape (T,) — filtered mean of the continuous
                        narrative state n_t
        """
        t_count = observations.shape[0]
        s_count = self.s_count
        regime_of = self._regime_of_state()  # (S,)

        # Expand per-regime params out to per-AUGMENTED-STATE params, since
        # every duration bucket of a given regime shares that regime's
        # dynamics.
        a_s = a[regime_of]           # (S,)
        q_s = q[regime_of]           # (S,)
        c_s = c[regime_of, :]        # (S, F)

        # Initial beliefs: no information yet, so start every regime's
        # duration=1 bucket equally likely, and the continuous state at 0
        # with a wide ("diffuse") variance since we have no prior pull.
        state_probs = np.zeros(s_count)
        for j in range(self.j_count):
            state_probs[j * self.d_max] = 1.0 / self.j_count  # duration=1 bucket for each regime
        n_filtered = np.zeros(s_count)
        p_filtered = np.full(s_count, self.diffuse_prior_var)

        total_loglik = 0.0
        regime_probs_over_time = np.zeros((t_count, self.j_count))
        state_over_time = np.zeros(t_count)

        for t in range(t_count):
            y_t = observations[t]            # (F,)
            r_t = emission_noise_var[t]      # (F,) heteroskedastic emission variance, from Day 19 sigma_t
            r_t = np.clip(r_t, 1e-6, None)    # numerical floor — never divide by exactly zero

            # ---- PREDICT step, vectorized over every (i, j) pair ----
            # n_pred[i, j] = a_s[j] * n_filtered[i]
            n_pred = np.outer(n_filtered, a_s)                       # (S, S)
            p_pred = np.outer(p_filtered, a_s ** 2) + q_s[np.newaxis, :]  # (S, S)

            # ---- Per-(t, j) quantities that don't depend on i ----
            # Sxx_j = sum_f C_j[f]^2 / R_t[f]   (how "informative" regime j's
            #         loading is, given this timestep's noise levels)
            sxx_j = (c_s ** 2 / r_t[np.newaxis, :]).sum(axis=1)       # (S,)
            # raw_data_info_j = sum_f C_j[f] * y_t[f] / R_t[f]
            data_info_j = (c_s * (y_t / r_t)[np.newaxis, :]).sum(axis=1)  # (S,)

            # ---- Innovation / likelihood, vectorized over (i, j, f) ----
            mean_pred = c_s[np.newaxis, :, :] * n_pred[:, :, np.newaxis]  # (S, S, F)
            innovation = y_t[np.newaxis, np.newaxis, :] - mean_pred        # (S, S, F)
            quad = (innovation ** 2 / r_t[np.newaxis, np.newaxis, :]).sum(axis=2)  # (S, S)

            denom = 1.0 + p_pred * sxx_j[np.newaxis, :]                    # (S, S), matrix-determinant-lemma term
            log_det_r = np.log(r_t).sum()                                   # scalar
            log_det_cov = log_det_r + np.log(np.clip(denom, 1e-12, None))   # (S, S)
            # Sherman-Morrison correction term for the quadratic form. This
            # uses the INNOVATION-weighted sum (not the raw-y-weighted
            # data_info_j from the update step below) — a different
            # projection of the same residual, computed fresh here:
            sxy_j_given_ij = (c_s[np.newaxis, :, :] * (innovation / r_t[np.newaxis, np.newaxis, :])).sum(axis=2)  # (S, S)
            mahalanobis = quad - (p_pred * sxy_j_given_ij ** 2) / denom

            loglik_pairs = -0.5 * (self.f_dim * np.log(2 * np.pi) + log_det_cov + mahalanobis)  # (S, S)

            # ---- Kim step: combine with the transition matrix ----
            with np.errstate(divide="ignore"):
                log_state_probs_prev = np.log(np.clip(state_probs, 1e-300, None))  # (S,)
                log_trans = np.log(np.clip(transition_matrix, 1e-300, None))       # (S, S)
            log_joint_pred = log_state_probs_prev[:, np.newaxis] + log_trans        # (S, S): log P(i_{t-1}, j_t | Y_{1:t-1})
            log_joint_post_unnorm = log_joint_pred + loglik_pairs                   # (S, S)

            # Total log-likelihood contribution of this timestep:
            #   log p(y_t | Y_{1:t-1}) = log sum_{i,j} exp(log_joint_post_unnorm)
            t_loglik = logsumexp(log_joint_post_unnorm)
            total_loglik += t_loglik

            log_joint_post = log_joint_post_unnorm - t_loglik  # now sums to 1 in prob space
            joint_post = np.exp(log_joint_post)                 # (S, S)

            # Posterior over the current augmented state j, marginalizing i:
            state_probs = joint_post.sum(axis=0)  # (S,)
            state_probs = state_probs / state_probs.sum()  # renormalize for numerical safety

            # ---- Kalman UPDATE (closed-form, scalar state / diagonal noise) ----
            p_post_inv = 1.0 / p_pred + sxx_j[np.newaxis, :]     # (S, S)
            p_post = 1.0 / p_post_inv                             # (S, S)
            n_post = p_post * (n_pred / p_pred + data_info_j[np.newaxis, :])  # (S, S)

            # ---- Kim COLLAPSING: (i, j) pairs -> single Gaussian per j ----
            new_n_filtered = np.zeros(s_count)
            new_p_filtered = np.full(s_count, self.diffuse_prior_var)
            for j in range(s_count):
                p_j = joint_post[:, j]  # (S,), P(i | j, Y_t) up to normalization
                mass = p_j.sum()
                if mass < 1e-12:
                    continue
                weights = p_j / mass
                mean_j = np.dot(weights, n_post[:, j])
                # Collapsed variance = within-pair variance + between-pair
                # variance of the means (the standard Kim 1994 formula) —
                # this is what keeps the filter from underestimating
                # uncertainty when several regime paths disagree.
                var_within = np.dot(weights, p_post[:, j])
                var_between = np.dot(weights, (n_post[:, j] - mean_j) ** 2)
                new_n_filtered[j] = mean_j
                new_p_filtered[j] = var_within + var_between

            n_filtered = new_n_filtered
            p_filtered = new_p_filtered

            # Record base-regime marginal probabilities (sum out duration)
            # and the overall filtered state mean for reporting/diagnostics.
            for j in range(self.j_count):
                regime_probs_over_time[t, j] = state_probs[j * self.d_max:(j + 1) * self.d_max].sum()
            state_over_time[t] = float(np.dot(state_probs, n_filtered))

        return float(total_loglik), regime_probs_over_time, state_over_time


# ===========================================================================
# STEP 6 — Parameter packing + the EM/MLE fitting harness.
# ===========================================================================
@dataclass
class NSSMParams:
    j_count: int
    a: np.ndarray                    # (J,)
    q: np.ndarray                    # (J,) process noise variance, > 0
    c: np.ndarray                    # (J, F) emission loadings
    duration_mu: np.ndarray          # (J,) log-normal duration mu per regime
    duration_sigma: np.ndarray       # (J,) log-normal duration sigma per regime, > 0
    cross_regime_transitions: np.ndarray  # (J, J), zero diagonal, rows sum to 1

    def duration_priors(self) -> List[LogNormalDurationPrior]:
        return [LogNormalDurationPrior(self.duration_mu[j], self.duration_sigma[j]) for j in range(self.j_count)]


def _param_count(j_count: int, f_dim: int) -> int:
    # a, log_q, C (F values), log_dur_mu, log_dur_sigma per regime,
    # plus (J-1) free cross-transition logits per regime row.
    return j_count * (2 + f_dim + 2) + j_count * (j_count - 1)


def _pack(params: NSSMParams) -> np.ndarray:
    j_count = params.j_count
    pieces = [
        np.arctanh(np.clip(params.a, -0.98, 0.98)),   # unconstrained -> (-1, 1) via tanh later
        np.log(np.clip(params.q, 1e-6, None)),
        params.c.flatten(),
        np.log(np.clip(params.duration_mu, 1e-6, None)),
        np.log(np.clip(params.duration_sigma, 1e-6, None)),
    ]
    # Cross-regime transition logits: for each row j, J-1 free logits
    # (softmax over the off-diagonal entries reproduces a valid
    # probability row that never puts mass back on the diagonal, since
    # self-transitions are handled entirely by the duration/hazard model).
    for j in range(j_count):
        off_diag = np.array([params.cross_regime_transitions[j, k] for k in range(j_count) if k != j])
        off_diag = np.clip(off_diag, 1e-6, 1.0)
        logits = np.log(off_diag) - np.log(off_diag[-1])  # anchor last logit at 0 for identifiability
        pieces.append(logits[:-1] if j_count > 2 else logits[:0])  # (J-2) free logits when J>2, else none extra beyond softmax-of-2
        # NOTE: for J=2 there is only one off-diagonal entry per row and it
        # must be 1.0 (only one other regime to switch to) -> 0 free params.
    return np.concatenate(pieces)


def _unpack(flat: np.ndarray, j_count: int, f_dim: int) -> NSSMParams:
    idx = 0
    a = np.tanh(flat[idx: idx + j_count]); idx += j_count
    q = np.exp(flat[idx: idx + j_count]); idx += j_count
    c = flat[idx: idx + j_count * f_dim].reshape(j_count, f_dim); idx += j_count * f_dim
    duration_mu = np.exp(flat[idx: idx + j_count]); idx += j_count
    duration_sigma = np.exp(flat[idx: idx + j_count]); idx += j_count

    cross = np.zeros((j_count, j_count))
    for j in range(j_count):
        others = [k for k in range(j_count) if k != j]
        n_free = max(len(others) - 1, 0)
        if n_free > 0:
            free_logits = flat[idx: idx + n_free]; idx += n_free
            logits = np.append(free_logits, 0.0)  # anchored last logit
        else:
            logits = np.zeros(len(others))  # J == 2 case: single option, prob 1
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        for k, p in zip(others, probs):
            cross[j, k] = p
    return NSSMParams(j_count, a, q, c, duration_mu, duration_sigma, cross)


def _random_init(j_count: int, f_dim: int, observations: np.ndarray, rng: np.random.Generator) -> NSSMParams:
    a = rng.uniform(-0.6, 0.6, size=j_count)
    q = rng.uniform(0.1, 1.0, size=j_count)
    c = rng.normal(loc=0.0, scale=np.std(observations) + 0.1, size=(j_count, f_dim))
    duration_mu = rng.uniform(1.5, 3.0, size=j_count)  # log-space -> median duration roughly e^mu sessions
    duration_sigma = rng.uniform(0.3, 0.8, size=j_count)
    cross = np.zeros((j_count, j_count))
    for j in range(j_count):
        others = [k for k in range(j_count) if k != j]
        weights = rng.dirichlet(np.ones(len(others)))
        for k, w in zip(others, weights):
            cross[j, k] = w
    return NSSMParams(j_count, a, q, c, duration_mu, duration_sigma, cross)


def _statsmodels_warm_start(observations: np.ndarray, j_count: int) -> Optional[NSSMParams]:
    """
    Uses `statsmodels.tsa.regime_switching.markov_regression.MarkovRegression`
    to fit a quick, DURATION-NAIVE (plain, geometric-implied) K-regime model
    on the mean observation series, purely to get a sane starting point for
    our own duration-augmented Kim filter's random restarts. This is one of
    the >=10 random inits — the rest stay genuinely random so the optimizer
    isn't only ever starting from statsmodels' particular local optimum.
    """
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError:
        logger.warning("statsmodels regime_switching module unavailable; skipping warm start")
        return None

    mean_series = observations.mean(axis=1)
    try:
        model = MarkovRegression(mean_series, k_regimes=j_count, trend="c", switching_variance=True)
        result = model.fit(search_reps=5, disp=False)
    except Exception:
        logger.warning("statsmodels warm-start fit failed; falling back to a random init", exc_info=False)
        return None

    # Use the smoothed regime probabilities to compute an empirical
    # per-regime mean of the FULL F-dim observation vector -> a sensible
    # starting guess for each regime's emission loading C_j.
    smoothed = np.asarray(result.smoothed_marginal_probabilities)  # (T, J)
    c_guess = np.zeros((j_count, F_DIM))
    for j in range(j_count):
        weights = smoothed[:, j]
        if weights.sum() > 1e-6:
            c_guess[j] = np.average(observations, axis=0, weights=weights)
    return NSSMParams(
        j_count=j_count,
        a=np.full(j_count, 0.3),
        q=np.full(j_count, 0.5),
        c=c_guess,
        duration_mu=np.full(j_count, 2.2),
        duration_sigma=np.full(j_count, 0.5),
        cross_regime_transitions=_uniform_cross_transitions(j_count),
    )


def _uniform_cross_transitions(j_count: int) -> np.ndarray:
    cross = np.zeros((j_count, j_count))
    for j in range(j_count):
        others = [k for k in range(j_count) if k != j]
        for k in others:
            cross[j, k] = 1.0 / len(others)
    return cross


def _neg_log_likelihood(
    flat_params: np.ndarray, j_count: int, f_dim: int, d_max: int,
    observations: np.ndarray, emission_noise_var: np.ndarray,
) -> float:
    params = _unpack(flat_params, j_count, f_dim)
    trans, _ = build_augmented_transition_matrix(
        j_count, d_max, params.duration_priors(), params.cross_regime_transitions
    )
    kim_filter = DurationAugmentedKimFilter(j_count, d_max, f_dim)
    loglik, _, _ = kim_filter.run(observations, emission_noise_var, params.a, params.q, params.c, trans)
    if not np.isfinite(loglik):
        return 1e10  # penalize numerically broken parameter regions instead of crashing the optimizer
    return -loglik


@dataclass
class NSSMFitResult:
    j_count: int
    params: NSSMParams
    log_likelihood: float
    bic: float
    n_params: int
    regime_probs: np.ndarray   # (T, J)
    filtered_state: np.ndarray  # (T,)


def fit_nssm_for_j(
    observations: np.ndarray,
    emission_noise_var: np.ndarray,
    j_count: int,
    d_max: int = 6,
    n_random_inits: int = 10,
    optimizer_maxiter: int = 60,
    seed: Optional[int] = None,
) -> NSSMFitResult:
    """
    Fits the NSSM for a FIXED number of regimes J, matching Sprint 3 Day
    8's exact fitting discipline: a hard minimum of `n_random_inits`
    restarts, keeping whichever converged solution has the highest
    log-likelihood — never the "best-looking" one.

    HONEST ENGINEERING NOTE ON "EM"
    ----------------------------------
    Classic Dempster-Laird-Rubin EM has a closed-form M-step when the
    duration prior is geometric (that's what makes plain HMMs/HSSMs with
    geometric duration EM-friendly). A LOG-NORMAL duration prior breaks
    that closed form — there's no clean sufficient-statistics update for
    log-normal (mu, sigma) from the E-step's soft state assignments alone.
    The field's standard practical response (used across the duration-HMM
    /HSMM literature) is to replace the M-step with direct numerical
    maximum-likelihood optimization of the SAME log-likelihood surface the
    Kim filter computes — which is exactly what `scipy.optimize.minimize`
    does here. We keep every other piece of the discipline identical to
    Sprint 3: many random restarts, keep the best log-likelihood, never
    the prettiest-looking regimes.
    """
    rng = np.random.default_rng(seed)
    t_count = observations.shape[0]
    f_dim = observations.shape[1]

    candidates: List[NSSMParams] = []
    warm = _statsmodels_warm_start(observations, j_count)
    if warm is not None:
        candidates.append(warm)
    while len(candidates) < n_random_inits:
        candidates.append(_random_init(j_count, f_dim, observations, rng))

    best_loglik = -np.inf
    best_params: Optional[NSSMParams] = None

    for i, init_params in enumerate(candidates):
        x0 = _pack(init_params)
        result = minimize(
            _neg_log_likelihood,
            x0,
            args=(j_count, f_dim, d_max, observations, emission_noise_var),
            method="L-BFGS-B",
            options={"maxiter": optimizer_maxiter},
        )
        loglik = -result.fun
        logger.info("  [J=%d] init %d/%d -> loglik=%.3f (converged=%s)", j_count, i + 1, len(candidates), loglik, result.success)
        if loglik > best_loglik:
            best_loglik = loglik
            best_params = _unpack(result.x, j_count, f_dim)

    if best_params is None:
        raise AssertionError("best_params is not None")
    trans, _ = build_augmented_transition_matrix(j_count, d_max, best_params.duration_priors(), best_params.cross_regime_transitions)
    kim_filter = DurationAugmentedKimFilter(j_count, d_max, f_dim)
    loglik, regime_probs, filtered_state = kim_filter.run(
        observations, emission_noise_var, best_params.a, best_params.q, best_params.c, trans
    )

    n_params = _param_count(j_count, f_dim)
    bic = -2.0 * loglik + n_params * np.log(t_count)

    return NSSMFitResult(
        j_count=j_count, params=best_params, log_likelihood=loglik, bic=bic,
        n_params=n_params, regime_probs=regime_probs, filtered_state=filtered_state,
    )


def canonical_sort_regimes(fit_result: NSSMFitResult) -> NSSMFitResult:
    """
    Label-switching fix, reusing Sprint 3 MP-02's exact convention: sort
    regimes by a canonical criterion (here, mean emission loading across
    all F dimensions, ascending) and lock that ordering. Two fits of the
    "same" underlying regimes can otherwise come out with regime 0 and
    regime 1 swapped between runs purely by chance — this makes results
    comparable across fits and across users.
    """
    params = fit_result.params
    order = np.argsort(params.c.mean(axis=1))
    sorted_params = NSSMParams(
        j_count=params.j_count,
        a=params.a[order],
        q=params.q[order],
        c=params.c[order, :],
        duration_mu=params.duration_mu[order],
        duration_sigma=params.duration_sigma[order],
        cross_regime_transitions=params.cross_regime_transitions[np.ix_(order, order)],
    )
    return NSSMFitResult(
        j_count=fit_result.j_count,
        params=sorted_params,
        log_likelihood=fit_result.log_likelihood,
        bic=fit_result.bic,
        n_params=fit_result.n_params,
        regime_probs=fit_result.regime_probs[:, order],
        filtered_state=fit_result.filtered_state,
    )


def fit_nssm(
    observations: np.ndarray,
    emission_noise_var: np.ndarray,
    j_candidates: Sequence[int] = (2, 3, 4),
    d_max: int = 6,
    n_random_inits: int = 10,
    optimizer_maxiter: int = 60,
    seed: Optional[int] = None,
) -> Tuple[NSSMFitResult, Dict[int, NSSMFitResult]]:
    """
    Top-level Day 20 entry point: fit the NSSM once per candidate J, then
    pick J by BIC — never because a particular J "gives more interesting
    regimes" (Sprint 3's own non-negotiable, reused verbatim here).

    Returns: (best_result, all_results_by_j) — the second is returned too
    so callers can log/inspect the full BIC comparison, not just the
    winner.
    """
    all_results: Dict[int, NSSMFitResult] = {}
    for j_count in j_candidates:
        logger.info("Fitting NSSM for J=%d ...", j_count)
        result = fit_nssm_for_j(
            observations, emission_noise_var, j_count,
            d_max=d_max, n_random_inits=n_random_inits,
            optimizer_maxiter=optimizer_maxiter, seed=seed,
        )
        result = canonical_sort_regimes(result)
        all_results[j_count] = result
        logger.info("  J=%d -> loglik=%.3f, n_params=%d, BIC=%.3f", j_count, result.log_likelihood, result.n_params, result.bic)

    best_j = min(all_results, key=lambda j: all_results[j].bic)
    logger.info("Selected J=%d by BIC (lowest BIC wins)", best_j)
    return all_results[best_j], all_results


# ===========================================================================
# STEP 7 — End-to-end smoke test.
# ===========================================================================
# NOTE ON SCALE: this __main__ block intentionally uses REDUCED settings
# (few sessions, small duration cap, fewer random inits, 2 candidate J
# values instead of the full {2,3,4}) purely so this file finishes
# executing quickly as a correctness check in a sandbox. Real per-user
# fitting must use this module's actual defaults (n_random_inits=10,
# j_candidates=(2,3,4)) against real session history.
if __name__ == "__main__":
    # --- Build synthetic sessions that alternate between two narrative
    # "chapters" so there's genuine regime structure for the NSSM to find.
    agentive_template = (
        "I decided to take charge of the situation at work. I chose to speak up and "
        "I'll be leading the new initiative. I see myself succeeding at this."
    )
    resigned_template = (
        "It happened to me again and I had no choice. They made me feel like it was "
        "my fault. I couldn't do anything, it just fell apart on its own."
    )
    synthetic_sessions: List[SessionInput] = []
    for day in range(16):
        chapter = agentive_template if (day // 6) % 2 == 0 else resigned_template
        prosody = (
            {"f0_contour_z": 0.7, "energy_envelope_z": 0.3}
            if chapter is agentive_template
            else {"f0_contour_z": -0.8, "energy_envelope_z": -0.6}
        )
        synthetic_sessions.append(SessionInput(session_id=f"day{day:02d}", transcript=chapter, prosody_features=prosody))

    # --- Day 19: run the WSL to get soft distributions + sigma_t ---
    wsl = WeakSupervisionLabelLayer()
    wsl.fit(synthetic_sessions)
    day19_output = wsl.transform(synthetic_sessions)

    # --- Convert Day 19 output into an (T, F) observation matrix + sigma ---
    obs_matrix = np.zeros((len(synthetic_sessions), F_DIM))
    sigma_matrix = np.zeros((len(synthetic_sessions), F_DIM))
    for i, session in enumerate(synthetic_sessions):
        obs, sigma = dimension_outputs_to_observation(day19_output[session.session_id])
        obs_matrix[i] = obs
        sigma_matrix[i] = sigma

    # --- Idiolect normalization ---
    normalizer = IdiolectNormalizer(window_sessions=30, min_sessions_for_baseline=5)
    normalized_obs = normalizer.normalize_session_series(obs_matrix)
    # Sessions before min_sessions_for_baseline come back as NaN (honest
    # "not enough history yet," not a fabricated baseline) -> fall back to
    # the raw (unnormalized) values just for those early sessions so the
    # NSSM has something to fit on in this demo; a production run would
    # instead simply not fit until enough history exists.
    fallback_mask = np.isnan(normalized_obs)
    normalized_obs[fallback_mask] = obs_matrix[fallback_mask]
    emission_var = np.clip(sigma_matrix ** 2, 1e-3, None)  # sigma_t -> variance, floored

    print("\n=== Idiolect-normalized observation matrix (first 3 sessions) ===")
    print(np.round(normalized_obs[:3], 2))

    # --- Conformal calibration demo with a handful of synthetic gold checks ---
    calibrator = ConformalCalibrator(alpha=0.10, min_gold_checks_for_personal_calibration=8)
    gold_checks = [
        GoldCheck("day00", NarrativeDimension.SELF_ROLE, "agentive_hero", day19_output["day00"]["self_role"].distribution),
        GoldCheck("day06", NarrativeDimension.SELF_ROLE, "victim", day19_output["day06"]["self_role"].distribution),
        GoldCheck("day01", NarrativeDimension.SELF_ROLE, "agentive_hero", day19_output["day01"]["self_role"].distribution),
        GoldCheck("day07", NarrativeDimension.SELF_ROLE, "victim", day19_output["day07"]["self_role"].distribution),
    ]
    qhat, was_personal = calibrator.calibrate(gold_checks, NarrativeDimension.SELF_ROLE)
    example_set = calibrator.predict_set(
        day19_output["day02"]["self_role"].distribution, qhat, NarrativeDimension.SELF_ROLE, "day02",
    )
    print(f"\n=== Conformal calibration (self_role) ===")
    print(f"qhat={qhat:.3f}, person_specific={was_personal} (target coverage={calibrator.target_coverage:.0%})")
    print(f"Prediction set for day02: {example_set.prediction_set}")

    # --- NSSM fitting (REDUCED settings for a fast smoke test — see note above) ---
    print("\n=== Fitting NSSM (reduced smoke-test settings) ===")
    best_result, all_results = fit_nssm(
        normalized_obs, emission_var,
        j_candidates=(2, 3),       # production default: (2, 3, 4)
        d_max=4,                    # production default: larger, e.g. 10+
        n_random_inits=4,           # production default: 10
        optimizer_maxiter=40,
        seed=7,
    )

    print("\nBIC by candidate J:")
    for j, res in sorted(all_results.items()):
        print(f"  J={j}: loglik={res.log_likelihood:.2f}, n_params={res.n_params}, BIC={res.bic:.2f}")
    print(f"\nSelected J={best_result.j_count}")
    print("Filtered regime probabilities (last 6 sessions):")
    print(np.round(best_result.regime_probs[-6:], 2))
    print("Filtered continuous narrative state n_t (last 6 sessions):")
    print(np.round(best_result.filtered_state[-6:], 2))
