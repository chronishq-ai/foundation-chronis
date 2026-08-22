import numpy as np
from scipy.stats import norm

# HONESTY FLAG (S56.2): doctrine requires testing whether the ACTUAL
# fitted behavioral generative model (HSSM/domain representation)
# predicts post-boundary data worse -- a generic scalar Gaussian fit is
# not the same claim. Swapping in the real generative model + choosing
# the null/seasonal baseline is senior-owned (Ownership Model:
# "Harnesses" tier). `evaluate_generative_model_degradation` below is
# the harness: it accepts pluggable fit/predict functions so a senior
# can wire the real HSSM in without touching harness plumbing. The
# PredictiveFitDegradation class below remains the (documented-generic)
# current implementation -- not removed, not silently promoted.
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
) -> dict:
    """Harness (S56.2, Harnesses tier). Held-out predictive-likelihood
    evaluation, generic over the model:

      fit_fn(pre_data) -> model_state
      predict_ll_fn(model_state, post_data) -> float (total log-lik)
      null_baseline_ll_fn(pre_data, post_data) -> float, optional --
        e.g. a seasonal/naive baseline log-lik for comparison. Senior
        defines what 'appropriate null' means; harness just plumbs it
        through if supplied.

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
        "post_predictive_ll": float(post_ll),
        "post_predictive_ll_per_sample": float(post_ll) / len(post_data),
        **window_info,
    }

    if null_baseline_ll_fn is not None:
        null_ll = null_baseline_ll_fn(pre_data, post_data)
        result["null_baseline_ll"] = float(null_ll)
        result["ll_vs_null_baseline"] = float(post_ll) - float(null_ll)

    return result


class PredictiveFitDegradation:
    """
    Condition 2 of 3 for phase-transition gate.
    Fits a simple Gaussian model on pre-boundary window, tests its
    predictive log-likelihood on post-boundary window. Sharp degradation
    = evidence the underlying regime actually changed, not just noise.

    HONESTY FLAG (S56.2): this is a generic scalar-Gaussian fit, not the
    actual fitted HSSM/domain generative model doctrine calls for. See
    `evaluate_generative_model_degradation` for the real-model harness.
    """

    def __init__(self, min_window: int = 10):
        self.min_window = min_window

    def fit_window(self, data: np.ndarray) -> dict:
        """Fit Gaussian (mean, std) on a window of data."""
        return {"mean": float(np.mean(data)), "std": float(np.std(data) + 1e-8)}

    def log_predictive_likelihood(self, model: dict, data: np.ndarray) -> float:
        """Total log-likelihood of data under a fitted model."""
        return float(np.sum(norm(model["mean"], model["std"]).logpdf(data)))

    def degradation_score(self, data: list[float], candidate_t: int,
                            pre_window: int = 20, post_window: int = 20) -> dict:
        """
        Fit on [candidate_t - pre_window, candidate_t), test log predictive
        likelihood on [candidate_t, candidate_t + post_window).
        Returns per-sample degradation: how much worse (more negative)
        the post-boundary log-likelihood is vs in-sample expectation.
        """
        data = np.asarray(data)
        pre_start = max(0, candidate_t - pre_window)
        post_end = min(len(data), candidate_t + post_window)

        if candidate_t - pre_start < self.min_window or post_end - candidate_t < self.min_window:
            return {"valid": False, "reason": "insufficient window"}

        pre_data = data[pre_start:candidate_t]
        post_data = data[candidate_t:post_end]

        model = self.fit_window(pre_data)

        in_sample_ll = self.log_predictive_likelihood(model, pre_data) / len(pre_data)
        out_sample_ll = self.log_predictive_likelihood(model, post_data) / len(post_data)

        degradation = in_sample_ll - out_sample_ll  # positive = degraded fit

        return {
            "valid": True,
            "in_sample_ll_per_sample": in_sample_ll,
            "out_sample_ll_per_sample": out_sample_ll,
            "degradation": degradation,
        }

    def is_degraded(self, data: list[float], candidate_t: int,
                      pre_window: int = 20, post_window: int = 20,
                      threshold: float = 2.0) -> bool:
        """Condition 2 gate: True if predictive fit degraded past threshold."""
        result = self.degradation_score(data, candidate_t, pre_window, post_window)
        if not result["valid"]:
            return False
        return result["degradation"] > threshold