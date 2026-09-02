from features.imu_features import extract_imu_features


def test_imu_feature_extraction():

    acceleration = [
        0.2,
        0.5,
        0.8,
        0.4
    ]

    result = extract_imu_features(
        acceleration
    )

    assert "movement_mean" in result
    assert "movement_variance" in result
    assert result["movement_mean"] == 0.475

from features.ppg_features import extract_ppg_features


def test_ppg_feature_extraction():

    heart_rates = [
        72,
        75,
        78,
        74
    ]

    result = extract_ppg_features(
        heart_rates
    )

    assert "heart_rate_mean" in result
    assert "heart_rate_variability" in result
    assert result["heart_rate_mean"] == 74.75

from features.audio_features import extract_audio_features


def test_audio_feature_extraction():

    result = extract_audio_features(
        speech_segments=12,
        duration_seconds=60,
        pauses=[1.2, 0.8, 2.0],
        energy_values=[0.5, 0.7, 0.6]
    )

    assert "speaking_rate" in result
    assert "average_pause_duration" in result
    assert "average_energy" in result

    assert result["average_energy"] == 0.6

from features.feature_pipeline import generate_feature_vector


def test_feature_pipeline():

    result = generate_feature_vector(
        imu_data=[
            0.2,
            0.5,
            0.8
        ],

        heart_rates=[
            70,
            75,
            80
        ],

        audio_data={
            "speech_segments":10,
            "duration_seconds":60,
            "pauses":[1,2],
            "energy_values":[0.5,0.7]
        }
    )

    assert "movement_mean" in result
    assert "heart_rate_mean" in result
    assert "speaking_rate" in result