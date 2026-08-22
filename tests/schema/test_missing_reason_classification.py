"""Tests for the S1.2 typed-missingness decision table."""

from chronis_ml.schema.models import MissingnessSignals, MissingReason
from chronis_ml.schema.validation import classify_missing_reason


def test_imu_stillness_and_ppg_dropout_classifies_not_worn() -> None:
    """S1.2 T1: a fixture matching the 'not worn' IMU+PPG signature must
    classify as NOT_WORN, not generic SENSOR_FAILURE."""

    signals = MissingnessSignals(imu_stillness=True, ppg_dropout=True)

    result = classify_missing_reason(modality="imu", signals=signals)

    assert result is MissingReason.NOT_WORN


def test_imu_stillness_alone_is_not_sufficient() -> None:
    """Stillness alone (participant genuinely still but device worn) must
    NOT be classified as not_worn — both signals are required together."""

    signals = MissingnessSignals(imu_stillness=True, ppg_dropout=False)

    result = classify_missing_reason(modality="imu", signals=signals)

    assert result is MissingReason.SENSOR_FAILURE


def test_ppg_dropout_alone_is_not_sufficient() -> None:
    """A brief PPG dropout (e.g. motion artifact on a worn device) must
    NOT alone be classified as not_worn."""

    signals = MissingnessSignals(imu_stillness=False, ppg_dropout=True)

    result = classify_missing_reason(modality="ppg", signals=signals)

    assert result is MissingReason.SENSOR_FAILURE


def test_mic_off_event_classifies_audio_as_audio_paused() -> None:
    """S1.2 T2 (audio side): a discrete mic-off event must classify the
    audio modality's missing reading as AUDIO_PAUSED."""

    signals = MissingnessSignals(mic_off_event=True)

    result = classify_missing_reason(modality="audio", signals=signals)

    assert result is MissingReason.AUDIO_PAUSED


def test_mic_off_event_does_not_affect_other_modalities() -> None:
    """S1.2 T2: a mic-off event must NOT explain a missing reading in any
    non-audio modality — other streams must remain classified on their
    own terms (falling back to SENSOR_FAILURE here, since no other
    signal is present)."""

    signals = MissingnessSignals(mic_off_event=True)

    imu_result = classify_missing_reason(modality="imu", signals=signals)
    ppg_result = classify_missing_reason(modality="ppg", signals=signals)

    assert imu_result is MissingReason.SENSOR_FAILURE
    assert ppg_result is MissingReason.SENSOR_FAILURE


def test_mic_off_event_takes_priority_over_not_worn_signature() -> None:
    """If a mic-off event AND the not_worn IMU+PPG signature are both
    present simultaneously, the audio modality's own explicit mic-off
    event takes priority over the generic not_worn signature."""

    signals = MissingnessSignals(
        imu_stillness=True,
        ppg_dropout=True,
        mic_off_event=True,
    )

    audio_result = classify_missing_reason(modality="audio", signals=signals)
    imu_result = classify_missing_reason(modality="imu", signals=signals)

    assert audio_result is MissingReason.AUDIO_PAUSED
    assert imu_result is MissingReason.NOT_WORN


def test_no_signals_falls_back_to_sensor_failure() -> None:
    """With no distinguishing signal at all, the default remains
    SENSOR_FAILURE — the decision table never invents a more specific
    reason without evidence."""

    signals = MissingnessSignals()

    result = classify_missing_reason(modality="heart_rate", signals=signals)

    assert result is MissingReason.SENSOR_FAILURE
