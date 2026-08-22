"""
S56.2 -- best-effort regime-conditional fit_fn / predict_ll_fn pair for
`evaluate_generative_model_degradation` (phase_transition/degradation.py),
built directly from HSSM-shaped output (regime_sequence + observations,
the same two arrays `domain_emergence.hssm_adapter.HSSMAdapterOutput`
extracts from a real `backbone.hssm.fit_hssm` result -- see that
module's docstring for the shape contract).

HONESTY FLAG -- READ BEFORE USING IN PRODUCTION:
This is NOT the same object as "does the actual fitted HSSM predict
post-boundary data worse". A real HSSM's predictive likelihood for an
unseen post-boundary window comes from forward-filtering the model's
own transition matrix + emission distributions into the future (i.e.
it does not get to see the post-window's regime labels). What this
module does instead: given the regime labels HSSM already assigned
(regime_sequence, decoded over the FULL series, pre and post alike),
it fits per-regime Gaussian emission stats on the pre-boundary window
and tests whether those SAME per-regime emission stats still explain
the post-boundary observations for the regimes decoded there. That is
a real, meaningful degradation signal (emission-parameter drift within
a regime) -- but it is a narrower claim than genuine held-out
predictive likelihood, because it consumes post-window regime labels
that a true forward-predictive evaluation would have to infer.

This gap is exactly what genuine `backbone.hssm.fit_hssm` predictive
scoring would close, and closing it is outside what an adapter over
already-decoded output can honestly do. Building the real thing needs
Sprint 3-4's `backbone` package (has the forward-filtering / predictive
machinery); that package is not part of this sprint-5-6 zip -- same
blocker `hssm_adapter.py` (S56.6) already documents. This module is
therefore a code-complete, testable step UP from the generic scalar-
Gaussian `PredictiveFitDegradation`, not a claim of full doctrine
compliance. It has only been exercised against
`tests/fixtures/synthetic_hssm_fixture.py` (test-fixture-only,
KNOWN regime assignment) -- never against a real `backbone.hssm`
result. Swapping this in as the production Condition-2 model, and any
decision about what null baseline is "appropriate", remain Senior-
owned per the pack's Ownership Model ("Harnesses" tier) -- see
`evaluate_generative_model_degradation`'s docstring. This module ships
as an opt-in path (see `PredictiveFitDegradation` and
`PhaseTransitionGate` -- same "supplied means opt-in, omitted means
unchanged default" pattern already used for the S56.1 entropy metric),
never a default swap.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm

from phase_transition.degradation import evaluate_generative_model_degradation


def stack_regime_observations(regime_sequence, observations) -> np.ndarray:
    """Combine (T,) regime labels and (T, F) observations into a single
    (T, F+1) array -- observations in columns [:-1], regime label (as
    float) in the last column -- so regime identity travels alongside
    each row through `evaluate_generative_model_degradation`'s generic
    windowing/slicing, without changing that harness's signature."""
    observations = np.asarray(observations, dtype=float)
    regime_sequence = np.asarray(regime_sequence, dtype=float).reshape(-1, 1)
    if observations.ndim == 1:
        observations = observations.reshape(-1, 1)
    if len(regime_sequence) != len(observations):
        raise ValueError(
            "regime_sequence and observations must have the same length "
            f"(got {len(regime_sequence)} and {len(observations)})"
        )
    return np.hstack([observations, regime_sequence])


def _split_labels(stacked: np.ndarray):
    stacked = np.asarray(stacked, dtype=float)
    return stacked[:, :-1], stacked[:, -1].astype(int)


def make_regime_conditional_fit_predict(min_var: float = 1e-6):
    """Returns (fit_fn, predict_ll_fn) for use with
    `evaluate_generative_model_degradation`. Both expect the `data`
    argument fed to the harness to be the output of
    `stack_regime_observations` (regime label riding in the last
    column) -- NOT raw observations alone.

    fit_fn: per-regime diagonal-Gaussian mean/var, estimated from
    whichever regimes actually appear in the pre-boundary window.
    Rows with any NaN feature (missing session, per HSSM convention --
    not imputed) are dropped before fitting, matching how
    `HSSMAdapterOutput.observations` represents dropout.

    predict_ll_fn: total log-likelihood of post-window rows under the
    per-regime stats their OWN decoded regime label points to; a
    post-window regime never seen pre-boundary falls back to a pooled
    (all-regimes) Gaussian fit from the pre-window, rather than
    silently contributing 0 -- a genuinely new/unseen regime is itself
    evidence of degraded fit, not something to skip."""

    def fit_fn(pre_data):
        feats, labels = _split_labels(pre_data)
        valid = ~np.isnan(feats).any(axis=1)
        feats, labels = feats[valid], labels[valid]

        regime_means, regime_vars = {}, {}
        for r in np.unique(labels):
            rows = feats[labels == r]
            if len(rows) == 0:
                continue
            regime_means[r] = rows.mean(axis=0)
            regime_vars[r] = rows.var(axis=0) + min_var

        if len(feats) > 0:
            fallback_mean = feats.mean(axis=0)
            fallback_var = feats.var(axis=0) + min_var
        else:
            F = feats.shape[1] if feats.ndim == 2 else 1
            fallback_mean = np.zeros(F)
            fallback_var = np.ones(F)

        return {
            "regime_means": regime_means,
            "regime_vars": regime_vars,
            "fallback_mean": fallback_mean,
            "fallback_var": fallback_var,
        }

    def predict_ll_fn(model_state, post_data):
        feats, labels = _split_labels(post_data)
        total_ll = 0.0
        for row, r in zip(feats, labels):
            if np.isnan(row).any():
                continue  # missing session -- skip, not imputed
            mean = model_state["regime_means"].get(r, model_state["fallback_mean"])
            var = model_state["regime_vars"].get(r, model_state["fallback_var"])
            total_ll += float(np.sum(norm(mean, np.sqrt(var)).logpdf(row)))
        return total_ll

    return fit_fn, predict_ll_fn


def pooled_gaussian_null_ll(pre_data, post_data) -> float:
    """Null baseline for `evaluate_generative_model_degradation`:
    single pooled (regime-blind) diagonal Gaussian fit on the
    pre-window, scored on the post-window. Answers "does knowing the
    regime label help at all, beyond just a plain Gaussian on the raw
    features" -- a reasonable, cheap null, but per the harness's own
    docstring the choice of null is Senior-owned; this is offered as a
    default, not a mandate."""
    pre_feats, _ = _split_labels(pre_data)
    post_feats, _ = _split_labels(post_data)
    pre_feats = pre_feats[~np.isnan(pre_feats).any(axis=1)]

    if len(pre_feats) == 0:
        return 0.0

    mean = pre_feats.mean(axis=0)
    var = pre_feats.var(axis=0) + 1e-6

    total_ll = 0.0
    for row in post_feats:
        if np.isnan(row).any():
            continue
        total_ll += float(np.sum(norm(mean, np.sqrt(var)).logpdf(row)))
    return total_ll


def evaluate_regime_conditional_degradation(
    regime_sequence,
    observations,
    candidate_t: int,
    timestamps: list[float] | None = None,
    pre_window: int = 20,
    post_window: int = 20,
    min_var: float = 1e-6,
) -> dict:
    """Convenience wrapper: builds the stacked input + regime-
    conditional fit/predict/null functions above and runs them through
    the S56.2 harness (`evaluate_generative_model_degradation`),
    additionally reporting an `in_sample_ll_per_sample` /
    `out_sample_ll_per_sample` / `degradation` triple in the same
    sign convention as `PredictiveFitDegradation.degradation_score`
    (positive = post-boundary fit degraded) so callers used to that
    class's output shape can compare directly.

    See module docstring for the honesty flag on what this can and
    cannot claim relative to true HSSM predictive likelihood."""
    stacked = stack_regime_observations(regime_sequence, observations)
    fit_fn, predict_ll_fn = make_regime_conditional_fit_predict(min_var=min_var)

    result = evaluate_generative_model_degradation(
        stacked.tolist(), candidate_t, fit_fn, predict_ll_fn,
        timestamps=timestamps, pre_window=pre_window, post_window=post_window,
        null_baseline_ll_fn=pooled_gaussian_null_ll,
    )
    if not result["valid"]:
        return result

    pre_start = max(0, candidate_t - pre_window)
    pre_data = stacked[pre_start:candidate_t]
    model_state = fit_fn(pre_data)
    in_sample_ll = predict_ll_fn(model_state, pre_data)
    n_pre_valid = int((~np.isnan(pre_data[:, :-1]).any(axis=1)).sum())
    n_post_valid = int(round(
        result["post_predictive_ll"] / result["post_predictive_ll_per_sample"]
    )) if result["post_predictive_ll_per_sample"] != 0 else 0

    in_sample_ll_per_sample = in_sample_ll / n_pre_valid if n_pre_valid else 0.0
    result["in_sample_ll_per_sample"] = in_sample_ll_per_sample
    result["out_sample_ll_per_sample"] = result["post_predictive_ll_per_sample"]
    result["degradation"] = in_sample_ll_per_sample - result["post_predictive_ll_per_sample"]
    return result


def is_regime_conditional_degraded(
    regime_sequence,
    observations,
    candidate_t: int,
    timestamps: list[float] | None = None,
    pre_window: int = 20,
    post_window: int = 20,
    threshold: float = 2.0,
) -> bool:
    """Threshold gate matching `PredictiveFitDegradation.is_degraded`'s
    interface/semantics, for the regime-conditional model above."""
    result = evaluate_regime_conditional_degradation(
        regime_sequence, observations, candidate_t,
        timestamps=timestamps, pre_window=pre_window, post_window=post_window,
    )
    if not result.get("valid"):
        return False
    return result["degradation"] > threshold
