import numpy as np
from phase_transition.degradation import (
    PredictiveFitDegradation, evaluate_generative_model_degradation,
)


def test_degrades_on_real_mean_shift():
    np.random.seed(42)
    pre = np.random.normal(0, 1, 100)
    post = np.random.normal(5, 1, 100)
    signal = np.concatenate([pre, post]).tolist()

    deg = PredictiveFitDegradation()
    assert deg.is_degraded(signal, candidate_t=100, threshold=2.0)


def test_no_degradation_on_stationary_noise():
    np.random.seed(1)
    signal = np.random.normal(0, 1, 200).tolist()

    deg = PredictiveFitDegradation()
    assert not deg.is_degraded(signal, candidate_t=100, threshold=2.0)


def _mean_fit(data):
    return {"mean": np.mean(data)}


def _neg_sq_error_ll(model, data):
    data = np.asarray(data)
    return -float(np.sum((data - model["mean"]) ** 2))


def test_evaluate_generative_model_degradation_reports_windows():
    """S56.2 T2: exact window boundaries present in machine-readable form."""
    np.random.seed(0)
    signal = list(np.random.normal(0, 1, 100)) + list(np.random.normal(5, 1, 100))
    timestamps = list(range(200))
    result = evaluate_generative_model_degradation(
        signal, candidate_t=100, fit_fn=_mean_fit, predict_ll_fn=_neg_sq_error_ll,
        timestamps=timestamps, pre_window=20, post_window=20,
    )
    assert result["valid"]
    assert result["pre_window_start_idx"] == 80
    assert result["pre_window_end_idx"] == 100
    assert result["post_window_start_idx"] == 100
    assert result["post_window_end_idx"] == 120
    assert "pre_window_start_ts" in result


def test_evaluate_generative_model_degradation_with_null_baseline():
    np.random.seed(0)
    signal = list(np.random.normal(0, 1, 100)) + list(np.random.normal(5, 1, 100))

    def null_ll(pre_data, post_data):
        return _neg_sq_error_ll({"mean": np.mean(pre_data)}, post_data) - 5.0

    result = evaluate_generative_model_degradation(
        signal, candidate_t=100, fit_fn=_mean_fit, predict_ll_fn=_neg_sq_error_ll,
        null_baseline_ll_fn=null_ll,
    )
    assert "null_baseline_ll" in result
    assert "ll_vs_null_baseline" in result


def test_evaluate_generative_model_degradation_insufficient_window():
    result = evaluate_generative_model_degradation(
        [1.0, 2.0], candidate_t=0, fit_fn=_mean_fit, predict_ll_fn=_neg_sq_error_ll,
    )
    assert not result["valid"]