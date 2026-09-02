from typing import Dict, Any


def validate_record(record: Dict[str, Any]) -> bool:
    """
    Validate standardized Chronis feature records.

    Expected format:

    {
        "user_id": "001",
        "timestamp": "2026-08-16T10:00:00",
        "features": {
            "heart_rate": 78
        }
    }
    """

    required_fields = [
        "user_id",
        "timestamp",
        "features"
    ]

    for field in required_fields:
        if field not in record:
            raise ValueError(
                f"Missing required field: {field}"
            )

    if not isinstance(record["features"], dict):
        raise TypeError(
            "Features must be a dictionary"
        )

    return True