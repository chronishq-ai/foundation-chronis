from feature_store.database import FeatureStore


def test_get_features_by_time_range():

    store = FeatureStore()

    # Clean previous test data so repeated pytest runs
    # do not accumulate duplicate PostgreSQL records.
    store.delete_features("user_001")

    store.insert_feature(
        "user_001",
        "2026-08-16T10:00:00",
        "heart_rate",
        75
    )

    store.insert_feature(
        "user_001",
        "2026-08-16T11:00:00",
        "heart_rate",
        80
    )

    store.insert_feature(
        "user_001",
        "2026-08-16T12:00:00",
        "heart_rate",
        85
    )

    result = store.get_features_by_time_range(
        "user_001",
        "2026-08-16T10:00:00",
        "2026-08-16T11:00:00"
    )

    assert len(result) == 2
    assert result[0]["value"] == 75
    assert result[1]["value"] == 80


def test_feature_store_rejects_invalid_timestamp():

    store = FeatureStore()

    try:
        store.insert_feature(
            "user_001",
            "invalid-time",
            "heart_rate",
            75
        )
        assert False
    except ValueError:
        assert True


def test_feature_store_rejects_empty_user():

    store = FeatureStore()

    try:
        store.insert_feature(
            "",
            "2026-08-16T10:00:00",
            "heart_rate",
            75
        )
        assert False
    except ValueError:
        assert True


def test_feature_store_rejects_empty_feature_name():

    store = FeatureStore()

    try:
        store.insert_feature(
            "user_001",
            "2026-08-16T10:00:00",
            "",
            75
        )
        assert False
    except ValueError:
        assert True


def test_feature_store_rejects_invalid_time_range():

    store = FeatureStore()

    try:
        store.get_features_by_time_range(
            "user_001",
            "2026-08-16T12:00:00",
            "2026-08-16T10:00:00"
        )
        assert False
    except ValueError:
        assert True