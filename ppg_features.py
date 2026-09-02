from typing import List, Dict


def extract_ppg_features(
    heart_rates: List[float]
) -> Dict[str, float]:
    """
    Extract basic features from PPG heart-rate data.

    Features:
    - Mean heart rate
    - Heart rate variability
    """

    if not heart_rates:
        raise ValueError(
            "Heart rate data cannot be empty"
        )

    mean_hr = sum(heart_rates) / len(heart_rates)

    variability = sum(
        (x - mean_hr) ** 2
        for x in heart_rates
    ) / len(heart_rates)

    return {
        "heart_rate_mean": round(mean_hr, 3),
        "heart_rate_variability": round(variability, 3)
    }