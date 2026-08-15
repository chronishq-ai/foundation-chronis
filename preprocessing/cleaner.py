from typing import Dict, Any


def clean_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean standardized Chronis feature records.

    Rules:
    - Preserve missing values
    - Remove invalid empty feature names
    - Keep valid feature values unchanged
    """

    cleaned_features = {}

    for name, value in record["features"].items():

        # Ignore invalid feature names
        if not name or not isinstance(name, str):
            continue

        # Preserve None values
        cleaned_features[name] = value


    cleaned_record = {
        "user_id": record["user_id"],
        "timestamp": record["timestamp"],
        "features": cleaned_features
    }

    return cleaned_record