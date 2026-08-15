from preprocessing.cleaner import clean_record
from preprocessing.normalizer import normalize_features

from features.feature_pipeline import generate_feature_vector

from alignment.temporal_alignment import align_features

from feature_store.database import FeatureStore

from tracking.mlflow_tracker import MLflowTracker


def test_complete_pipeline():

    # Input data

    record = {
        "user_id": "user_001",
        "timestamp": "2026-08-16T10:00:00",
        "features": {
            "heart_rate": 80,
            "movement": 5
        }
    }


    # Cleaning

    cleaned = clean_record(record)

    assert cleaned["user_id"] == "user_001"


    # Normalization

    normalized = normalize_features(
        cleaned["features"],
        {
            "heart_rate":70,
            "movement":3
        }
    )

    assert normalized["heart_rate"] == 10


    # Feature extraction

    features = generate_feature_vector(
        imu_data=[
            0.2,
            0.5,
            0.8
        ],

        heart_rates=[
            70,
            75,
            80
        ],

        audio_data={
            "speech_segments":10,
            "duration_seconds":60,
            "pauses":[1,2],
            "energy_values":[0.5,0.7]
        }
    )


    assert "movement_mean" in features


    # Alignment

    aligned = align_features(
        {
            "sensor":[
                {
                    "timestamp":"10:00",
                    "value":0.5
                }
            ]
        }
    )

    assert len(aligned) == 1


    # Feature Store

    store = FeatureStore()

    store.insert_feature(
        "user_001",
        "10:00",
        "movement",
        0.5
    )

    assert len(
        store.get_features("user_001")
    ) == 1


    # Tracking

    tracker = MLflowTracker()

    tracker.log_experiment(
        "pipeline_test",
        "hash123",
        {},
        {
            "accuracy":0.9
        }
    )

    assert len(
        tracker.get_experiments()
    ) == 1