import mlflow

from tracking.mlflow_tracker import MLflowTracker


def test_mlflow_logs_experiment():

    tracker = MLflowTracker(
        experiment_name="chronis-test"
    )

    tracker.log_experiment(
        name="feature_pipeline_test",
        dataset_hash="dataset_123",
        parameters={
            "window": "5min",
            "feature_version": "v1"
        },
        metrics={
            "accuracy": 0.90,
            "missing_rate": 0.02
        }
    )

    experiments = tracker.get_experiments()

    assert len(experiments) == 1

    experiment = experiments[0]

    assert experiment["name"] == "feature_pipeline_test"
    assert experiment["dataset_hash"] == "dataset_123"
    assert experiment["parameters"]["window"] == "5min"
    assert experiment["metrics"]["accuracy"] == 0.90
    assert "timestamp" in experiment


def test_mlflow_run_exists():

    experiment = mlflow.get_experiment_by_name(
        "chronis-test"
    )

    assert experiment is not None

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    assert len(runs) >= 1

    assert "tags.dataset_hash" in runs.columns
    assert "metrics.accuracy" in runs.columns
    assert "tags.experiment_timestamp" in runs.columns

    assert runs.iloc[0]["tags.dataset_hash"] == "dataset_123"
    assert runs.iloc[0]["metrics.accuracy"] == 0.90