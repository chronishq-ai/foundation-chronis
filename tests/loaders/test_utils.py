from datetime import UTC

from chronis_ml.loaders.utils import (
    normalize_feature_name,
    parse_timestamp,
)


def test_normalize_feature_name() -> None:
    assert normalize_feature_name("Heart Rate / Mean") == "heart_rate_mean"


def test_parse_timestamp_is_timezone_aware() -> None:
    timestamp = parse_timestamp("2026-08-16T12:00:00")

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() is not None


def test_parse_timestamp_uses_utc() -> None:
    timestamp = parse_timestamp("2026-08-16T12:00:00")

    assert timestamp.tzinfo == UTC
