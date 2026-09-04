"""S1.2 re-verification — original test sheet, run fresh.

Test Sheet — S1.2 (from the original Intern Remediation & Test Pack):
  T1: Fixture matching the "not worn" IMU+PPG signature -> Validator
      classifies as not_worn, not generic sensor_failure
  T2: Fixture matching a mic-off event only -> Only audio NULL'd with
      audio_paused; other modalities untouched
"""

from chronis_ml.schema.models import MissingnessSignals, MissingReason
from chronis_ml.schema.validation import classify_missing_reason


def test_s12_t1_not_worn_signature_classifies_correctly() -> None:
    """T1: not_worn signature -> not_worn, not sensor_failure."""
    signals = MissingnessSignals(imu_stillness=True, ppg_dropout=True)

    result = classify_missing_reason(modality="imu", signals=signals)

    assert result is MissingReason.NOT_WORN
    assert result is not MissingReason.SENSOR_FAILURE


def test_s12_t2_mic_off_only_audio_nulled() -> None:
    """T2: mic-off event -> only audio gets AUDIO_PAUSED; other
    modalities are untouched by the same signal."""
    signals = MissingnessSignals(mic_off_event=True)

    audio_result = classify_missing_reason(modality="audio", signals=signals)
    imu_result = classify_missing_reason(modality="imu", signals=signals)
    ppg_result = classify_missing_reason(modality="ppg", signals=signals)

    assert audio_result is MissingReason.AUDIO_PAUSED
    assert imu_result is not MissingReason.AUDIO_PAUSED
    assert ppg_result is not MissingReason.AUDIO_PAUSED
