import numpy as np
from scipy.stats import norm


class PredictiveFitDegradation:
    """
    Condition 2 of 3 for phase-transition gate.
    Fits a simple Gaussian model on pre-boundary window, tests its
    predictive log-likelihood on post-boundary window. Sharp degradation
    = evidence the underlying regime actually changed, not just noise.
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