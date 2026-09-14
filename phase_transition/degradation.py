import numpy as np

from phase_transition.diagnostics.scalar_gaussian_degradation import (
    PredictiveFitDegradation,
)

__all__ = ["evaluate_generative_model_degradation", "PredictiveFitDegradation"]

# HONESTY FLAG (S56.2): doctrine requires testing whether the ACTUAL
# fitted behavioral generative model (HSSM/domain representation)
# predicts post-boundary data worse -- a generic scalar Gaussian fit is
# not the same claim. Swapping in the real generative model + choosing
# the null/seasonal baseline is senior-owned (Ownership Model:
# "Harnesses" tier). `evaluate_generative_model_degradation` below is
# the harness: it accepts pluggable fit/predict functions so a senior
# can wire the real HSSM in without touching harness plumbing.
#
# RELOCATED (item 4 / R2-S56.4 mechanical sub-fix, this pass): the
# generic scalar-Gaussian `PredictiveFitDegradation` implementation now
# physically lives in `phase_transition/diagnostics/`, per the doc's
# "keep the scalar-Gaussian path only under diagnostics/" ask -- it is
# re-exported here ONLY for backward compatibility with existing
# imports (`from phase_transition.degradation import
# PredictiveFitDegradation`); do not add new logic to this module for
# it, edit `diagnostics/scalar_gaussian_degradation.py` instead.
#
# See phase_transition/hssm_degradation.py for a best-effort
# regime-conditional fit_fn/predict_ll_fn pair built on top of this
# harness -- code-complete and tested against the synthetic HSSM
# fixture, but explicitly NOT the same claim as genuine HSSM predictive
# likelihood (see that module's docstring), and NOT verified against a
# real backbone.hssm result (backbone/ is not part of this zip -- same
# S56.6 blocker). Wired into PhaseTransitionGate as an opt-in path
# only, same pattern as the S56.1 entropy metric below.


def evaluate_generative_model_degradation(
    data: list[float],
    candidate_t: int,
    fit_fn,
    predict_ll_fn,
    timestamps: list[float] | None = None,
    pre_window: int = 20,
    post_window: int = 20,
    null_baseline_ll_fn=None,
    model_id: str | None = None,
    model_version: str | None = None,
) -> dict:
    """Harness (S56.2, Harnesses tier). Held-out predictive-likelihood
    evaluation, generic over the model:

      fit_fn(pre_data) -> model_state
      predict_ll_fn(model_state, post_data) -> float (total log-lik)
      null_baseline_ll_fn(pre_data, post_data) -> float, optional --
        e.g. a seasonal/naive baseline log-lik for comparison. Senior
        defines what 'appropriate null' means; harness just plumbs it
        through if supplied.

    model_id / model_version (item 4 / R2-S56.4 mechanical sub-fix,
    this pass): caller-supplied identifiers persisted verbatim into the
    result dict so a decision record can never carry an anonymous
    model. The harness does not invent or validate these -- it only
    refuses to silently drop them; omitting both leaves `model_id=None`
    visible in the result rather than fabricating a default.

    Records exact pre/post window boundaries in machine-readable form
    (S56.2 Test Sheet T2) for the reproducibility manifest. Does NOT
    decide which fit_fn/predict_ll_fn is 'correct' -- caller supplies
    the real HSSM/domain-representation functions; a senior approves
    that wiring."""
    data = np.asarray(data)
    pre_start = max(0, candidate_t - pre_window)
    post_end = min(len(data), candidate_t + post_window)

    if candidate_t - pre_start < 1 or post_end - candidate_t < 1:
        return {"valid": False, "reason": "insufficient window"}

    pre_data = data[pre_start:candidate_t]
    post_data = data[candidate_t:post_end]

    model_state = fit_fn(pre_data)
    post_ll = predict_ll_fn(model_state, post_data)

    window_info = {
        "pre_window_start_idx": int(pre_start),
        "pre_window_end_idx": int(candidate_t),
        "post_window_start_idx": int(candidate_t),
        "post_window_end_idx": int(post_end),
    }
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype=float)
        window_info.update({
            "pre_window_start_ts": float(timestamps[pre_start]),
            "pre_window_end_ts": float(timestamps[candidate_t]),
            "post_window_start_ts": float(timestamps[candidate_t]),
            "post_window_end_ts": float(timestamps[post_end - 1]),
        })

    result = {
        "valid": True,
        "model_id": model_id,
        "model_version": model_version,
        "post_predictive_ll": float(post_ll),
        "post_predictive_ll_per_sample": float(post_ll) / len(post_data),
        **window_info,
    }

    if null_baseline_ll_fn is not None:
        null_ll = null_baseline_ll_fn(pre_data, post_data)
        result["null_baseline_ll"] = float(null_ll)
        result["ll_vs_null_baseline"] = float(post_ll) - float(null_ll)

    return result