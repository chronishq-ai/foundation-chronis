from typing import List, Dict


def extract_imu_features(
    acceleration: List[float]
) -> Dict[str, float]:
    """
    Extract movement features from IMU acceleration data.

    Features:
    - Mean movement intensity
    - Movement variance
    """

    if not acceleration:
        raise ValueError(
            "Acceleration data cannot be empty"
        )

    mean_value = sum(acceleration) / len(acceleration)

    variance = sum(
        (x - mean_value) ** 2
        for x in acceleration
    ) / len(acceleration)

    return {
        "movement_mean": round(mean_value, 3),
        "movement_variance": round(variance, 3)
    }