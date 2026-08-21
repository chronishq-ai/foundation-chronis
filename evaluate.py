"""
Pod B v0.2 - Calibration evaluation pipeline.

Compares LLM confidence and naive-baseline confidence against blind human
ground truth, using the metric and pass condition locked in protocol.py
BEFORE this file is ever run against real data. Per spec section 6:
present objective numbers, not narrative conclusions.

Only (event, variable) pairs with an actual human rating are scored - this
file does not fabricate ground truth for unrated pairs, and does not run
the LLM against events nobody has rated yet (saves API quota).
"""
from __future__ import annotations

import json
from typing import Optional

import protocol
from analyzer import analyze_event
from baseline import baseline_predict
from dataset import get_by_id
from ground_truth import load_ratings
from logger import EvaluationLogger

_REPORT_PATH = "logs/calibration_report.json"
_eval_logger = EvaluationLogger()


def _extract_confidence(prediction: dict, variable: str) -> float:
    """
    Confidence for a variable from an EventSignals-shaped dict. A variable
    absent from `signals` means the model/baseline found no signal for it -
    treated as confidence 0, not skipped, since the human rated it as
    relevant and silence is itself a (possibly wrong) confidence claim.
    """
    return prediction.get("signals", {}).get(variable, {}).get("confidence", 0.0)


def _mean_absolute_error(pairs: list[tuple[float, float]]) -> Optional[float]:
    if not pairs:
        return None
    return sum(abs(pred - truth) for pred, truth in pairs) / len(pairs)


def run_evaluation() -> dict:
    """
    Runs the full v0.2 calibration evaluation and returns the report dict.
    Also writes it to logs/calibration_report.json.
    """
    ratings = load_ratings()
    if not ratings:
        raise RuntimeError(
            "No ground-truth ratings found. Run `python ground_truth.py` "
            "to collect blind human ratings before evaluating."
        )

    # Only pairs that have real ground truth get scored.
    rated_pairs = {(r["event_id"], r["variable"]) for r in ratings}
    # Latest rating per pair (mirrors ground_truth.get_rating's tie-break).
    latest_rating: dict[tuple[str, str], float] = {}
    for r in ratings:
        latest_rating[(r["event_id"], r["variable"])] = r["human_rating"]

    event_ids = sorted({event_id for event_id, _ in rated_pairs})

    llm_predictions: dict[str, dict] = {}
    baseline_predictions: dict[str, dict] = {}
    failed_events: list[dict] = []

    for event_id in event_ids:
        entry = get_by_id(event_id)
        text = entry["text"]

        baseline_predictions[event_id] = baseline_predict(text)

        try:
            llm_predictions[event_id] = analyze_event(text)
        except RuntimeError as exc:
            failed_events.append({"event_id": event_id, "error": str(exc)})
            llm_predictions[event_id] = {"signals": {}}

    llm_pairs: list[tuple[float, float]] = []
    baseline_pairs: list[tuple[float, float]] = []

    for event_id, variable in rated_pairs:
        truth = latest_rating[(event_id, variable)]
        llm_conf = _extract_confidence(llm_predictions[event_id], variable)
        baseline_conf = _extract_confidence(baseline_predictions[event_id], variable)
        pair_error = next(
            (f["error"] for f in failed_events if f["event_id"] == event_id), None
        )

        llm_pairs.append((llm_conf, truth))
        baseline_pairs.append((baseline_conf, truth))

        _eval_logger.log(
            event_id=event_id,
            variable=variable,
            human_rating=truth,
            baseline_prediction=baseline_conf,
            llm_prediction=llm_conf,
            metric_value=abs(llm_conf - truth),
            error=pair_error,
        )

    llm_mae = _mean_absolute_error(llm_pairs)
    baseline_mae = _mean_absolute_error(baseline_pairs)

    # protocol.PASS_CONDITION is documentation of the rule; evaluated here
    # directly rather than via eval() of the string, to avoid any ambiguity
    # about what's actually being computed.
    llm_beats_baseline = (
        llm_mae is not None and baseline_mae is not None and llm_mae < baseline_mae
    )

    report = {
        "metric": protocol.METRIC,
        "pass_condition": protocol.PASS_CONDITION,
        "num_rated_pairs": len(rated_pairs),
        "num_events_evaluated": len(event_ids),
        "num_events_failed": len(failed_events),
        "llm_mae": llm_mae,
        "baseline_mae": baseline_mae,
        "llm_beats_baseline": llm_beats_baseline,
        "result": "PASS" if llm_beats_baseline else "FAIL",
        "failed_events": failed_events,
    }

    with open(_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    return report


def print_report(report: dict) -> None:
    """Objective numbers only - no narrative, per spec section 6."""
    print(f"metric: {report['metric']}")
    print(f"pass_condition: {report['pass_condition']}")
    print(f"num_rated_pairs: {report['num_rated_pairs']}")
    print(f"num_events_evaluated: {report['num_events_evaluated']}")
    print(f"num_events_failed: {report['num_events_failed']}")
    print(f"llm_mae: {report['llm_mae']}")
    print(f"baseline_mae: {report['baseline_mae']}")
    print(f"llm_beats_baseline: {report['llm_beats_baseline']}")
    print(f"result: {report['result']}")


if __name__ == "__main__":
    print_report(run_evaluation())