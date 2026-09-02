from typing import Dict, Any

from features.imu_features import extract_imu_features
from features.ppg_features import extract_ppg_features
from features.audio_features import extract_audio_features


def generate_feature_vector(
    imu_data,
    heart_rates,
    audio_data: Dict[str, Any]
) -> Dict[str, float]:
    """
    Combine multimodal features into one ML feature vector.
    """

    features = {}

    features.update(
        extract_imu_features(
            imu_data
        )
    )

    features.update(
        extract_ppg_features(
            heart_rates
        )
    )

    features.update(
        extract_audio_features(
            **audio_data
        )
    )

    return features