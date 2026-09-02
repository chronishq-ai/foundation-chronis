from preprocessing.validator import validate_record


def test_valid_record():

    record = {
        "user_id": "001",
        "timestamp": "2026-08-16T10:00:00",
        "features": {
            "heart_rate": 80
        }
    }

    assert validate_record(record) is True


def test_missing_user():

    record = {
        "timestamp": "2026-08-16T10:00:00",
        "features": {}
    }

    try:
        validate_record(record)
        assert False
    except ValueError:
        assert True
from preprocessing.cleaner import clean_record


def test_clean_preserves_missing_values():

    record = {
        "user_id": "001",
        "timestamp": "2026-08-16T10:00:00",
        "features": {
            "heart_rate": 80,
            "movement": None
        }
    }

    result = clean_record(record)

    assert result["features"]["movement"] is None


def test_clean_removes_invalid_feature_names():

    record = {
        "user_id": "001",
        "timestamp": "2026-08-16T10:00:00",
        "features": {
            "heart_rate": 80,
            "": 20
        }
    }

    result = clean_record(record)

    assert "" not in result["features"]

from preprocessing.normalizer import normalize_features


def test_personal_normalization():

    features = {
        "heart_rate": 80,
        "movement": 5
    }

    baseline = {
        "heart_rate": 70,
        "movement": 3
    }

    result = normalize_features(
        features,
        baseline
    )

    assert result["heart_rate"] == 10
    assert result["movement"] == 2


def test_normalization_preserves_missing():

    features = {
        "heart_rate": None
    }

    baseline = {
        "heart_rate": 70
    }

    result = normalize_features(
        features,
        baseline
    )

    assert result["heart_rate"] is None