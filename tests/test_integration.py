from feature_store.database import FeatureStore


def cleanup_user(store, user_id):
    store.delete_features(user_id)


def test_get_features_by_time_range():

    store = FeatureStore()

    user_id = "feature_store_test_user"

    cleanup_user(store, user_id)

    store.insert_feature(
        user_id,
        "2026-08-16T10:00:00",
        "heart_rate",
        75
    )

    store.insert_feature(
        user_id,
        "2026-08-16T11:00:00",
        "heart_rate",
        80
    )

    store.insert_feature(
        user_id,
        "2026-08-16T12:00:00",
        "heart_rate",
        85
    )

    result = store.get_features_by_time_range(
        user_id,
        "2026-08-16T10:00:00",
        "2026-08-16T11:00:00"
    )

    assert len(result) == 2
    assert result[0]["value"] == 75
    assert result[1]["value"] == 80

    cleanup_user(store, user_id)


def test_feature_store_rejects_invalid_timestamp():

    store = FeatureStore()

    try:
        store.insert_feature(
            "feature_store_invalid_timestamp_user",
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
            "feature_store_empty_feature_user",
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
            "feature_store_invalid_range_user",
            "2026-08-16T12:00:00",
            "2026-08-16T10:00:00"
        )
        assert False
    except ValueError:
        assert True