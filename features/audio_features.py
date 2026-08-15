from typing import List, Dict


def extract_audio_features(
    speech_segments: int,
    duration_seconds: float,
    pauses: List[float],
    energy_values: List[float]
) -> Dict[str, float]:
    """
    Extract basic audio/prosody features.

    Features:
    - Speaking rate
    - Average pause duration
    - Average audio energy
    """

    if duration_seconds <= 0:
        raise ValueError(
            "Duration must be greater than zero"
        )

    if not energy_values:
        raise ValueError(
            "Energy values cannot be empty"
        )

    speaking_rate = (
        speech_segments / duration_seconds
    )

    average_pause = (
        sum(pauses) / len(pauses)
        if pauses
        else 0
    )

    average_energy = (
        sum(energy_values)
        / len(energy_values)
    )

    return {
        "speaking_rate": round(
            speaking_rate, 3
        ),
        "average_pause_duration": round(
            average_pause, 3
        ),
        "average_energy": round(
            average_energy, 3
        )
    }