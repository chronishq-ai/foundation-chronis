from feature_store.database import FeatureStore


def test_insert_and_retrieve_feature():

    store = FeatureStore()


    store.insert_feature(
        "user_001",
        "2026-08-16T10:00:00",
        "heart_rate",
        75
    )


    result = store.get_features(
        "user_001"
    )


    assert len(result) == 1
    assert result[0]["feature_name"] == "heart_rate"
    assert result[0]["value"] == 75

from tracking.mlflow_tracker import MLflowTracker


def test_experiment_tracking():

    tracker = MLflowTracker()


    tracker.log_experiment(
        name="feature_pipeline_v1",
        dataset_hash="abc123",
        parameters={
            "window":"5min"
        },
        metrics={
            "missing_rate":0.02
        }
    )


    result = tracker.get_experiments()


    assert len(result) == 1
    assert result[0]["name"] == "feature_pipeline_v1"
    assert result[0]["dataset_hash"] == "abc123"