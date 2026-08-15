from chronis_ml.health import health_check


def test_health_check() -> None:
    assert health_check() == "chronis-ml-ok"
