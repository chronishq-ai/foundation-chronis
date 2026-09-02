from typing import Dict, Any


def normalize_features(
    features: Dict[str, float],
    baseline: Dict[str, float]
) -> Dict[str, float]:
    """
    Normalize features using user's personal baseline.

    Formula:

    normalized_value =
        current_value - personal_baseline

    Example:

    heart_rate:
        current = 80
        baseline = 70

        output = 10
    """

    normalized = {}

    for feature, value in features.items():

        # Skip missing values
        if value is None:
            normalized[feature] = None
            continue

        if feature not in baseline:
            raise ValueError(
                f"Missing baseline for feature: {feature}"
            )

        normalized[feature] = (
            value - baseline[feature]
        )

    return normalized