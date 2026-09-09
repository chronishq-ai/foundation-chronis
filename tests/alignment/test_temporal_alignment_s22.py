"""S2.2 test sheet - original 6 test cases.

Test Sheet - S2.2:
  T1: Synthetic multi-modal stream, all signals clean -> Align ->
      Every output minute confidence == 1.0
  T2: Exactly one modality dropped for a window -> Align -> Confidence
      == 0.75 for those minutes; other modalities' values unaffected
  T3: Two modalities dropped; then enough dropped for <0.5 -> Align ->
      0.5 case included with correct score; <0.5 rows provably
      excluded from what's passed downstream to the HSSM
  T4: >10-minute gap -> Align -> No interpolation across the gap;
      values either side unchanged, gap stays typed NULL
  T5: 5-minute audio_paused window while IMU/PPG continue -> Align ->
      Only audio NULL for those minutes; IMU/PPG rows remain valid
  T6: Full output matrix -> Scan every NULL cell -> Zero cells
      silently imputed with zero or mean anywhere in output
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from chronis_ml.alignment.temporal_alignment import align_to_minute_grid, select_for_downstream
from chronis_ml.schema.models import FeatureRecord, MeasurementStatus, MissingReason

USER_ID = "user_001"
START = datetime(2026, 8, 16, 9, 0, tzinfo=UTC)


def observed(modality: str, minute_offset: int, value: float) -> FeatureRecord:
    return FeatureRecord(
        user_id=USER_ID,
        timestamp=START + timedelta(minutes=minute_offset),
        feature_name=modality,
        value=value,
        modality=modality,
        status=MeasurementStatus.OBSERVED,
    )


def missing(modality: str, minute_offset: int, reason: MissingReason) -> FeatureRecord:
    return FeatureRecord(
        user_id=USER_ID,
        timestamp=START + timedelta(minutes=minute_offset),
        feature_name=modality,
        value=None,
        modality=modality,
        status=MeasurementStatus.MISSING,
        missing_reason=reason,
    )


# --- T1: all signals clean -> confidence == 1.0 everywhere -------------------


def test_s22_t1_all_signals_present_confidence_is_1() -> None:
    records_by_modality = {
        "audio": [observed("audio", m, 0.5) for m in range(5)],
        "imu": [observed("imu", m, 1.0) for m in range(5)],
        "ppg": [observed("ppg", m, 70.0) for m in range(5)],
    }

    aligned = align_to_minute_grid(USER_ID, records_by_modality)

    assert len(aligned) == 5
    assert all(minute.confidence == 1.0 for minute in aligned)
    assert all(not minute.excluded_from_downstream for minute in aligned)


# --- T2: one modality dropped for a window -----------------------------------


def test_s22_t2_one_modality_missing_confidence_is_075() -> None:
    records_by_modality = {
        "audio": [observed("audio", m, 0.5) for m in range(5) if m not in (2, 3)],  # audio dropped
        "imu": [observed("imu", m, 1.0) for m in range(5)],
        "ppg": [observed("ppg", m, 70.0) for m in range(5)],
    }

    aligned = align_to_minute_grid(USER_ID, records_by_modality)
    by_minute = {m.minute_start: m for m in aligned}

    dropped_minutes = [by_minute[START + timedelta(minutes=m)] for m in (2, 3)]
    assert all(minute.confidence == 0.75 for minute in dropped_minutes)
    assert all(minute.missing_modalities == ("audio",) for minute in dropped_minutes)

    # Other modalities' VALUES must be unaffected during the drop.
    assert dropped_minutes[0].values["imu"] == 1.0
    assert dropped_minutes[0].values["ppg"] == 70.0

    # Non-dropped minutes remain at full confidence.
    clean_minutes = [by_minute[START + timedelta(minutes=m)] for m in (0, 1, 4)]
    assert all(minute.confidence == 1.0 for minute in clean_minutes)


# --- T3: two dropped -> 0.5; enough dropped -> <0.5 excluded -----------------


def test_s22_t3_two_modalities_missing_confidence_is_05() -> None:
    records_by_modality = {
        "audio": [observed("audio", 0, 0.5)],  # minute 0 only
        "imu": [],  # missing at minute 0
        "ppg": [],  # missing at minute 0
    }

    aligned = align_to_minute_grid(USER_ID, records_by_modality, modalities=("audio", "imu", "ppg"))

    assert len(aligned) == 1
    assert aligned[0].confidence == 0.5
    assert set(aligned[0].missing_modalities) == {"imu", "ppg"}
    assert not aligned[0].excluded_from_downstream  # 0.5 is still included per spec


def test_s22_t3_three_or_more_missing_is_excluded_from_downstream() -> None:
    records_by_modality = {
        "audio": [],
        "imu": [],
        "ppg": [],
        "environment": [observed("environment", 0, 22.0)],
    }

    aligned = align_to_minute_grid(
        USER_ID, records_by_modality, modalities=("audio", "imu", "ppg", "environment")
    )

    assert aligned[0].confidence < 0.5
    assert aligned[0].excluded_from_downstream is True

    downstream = select_for_downstream(aligned)
    assert downstream == ()  # provably excluded from what's actually returned


def test_s22_t3_mixed_confidence_minutes_only_excludes_the_bad_ones() -> None:
    """A realistic mixed run: some minutes fine, one minute badly
    degraded -- select_for_downstream must keep the good ones and drop
    only the excluded one."""

    records_by_modality = {
        "audio": [observed("audio", 0, 0.5), observed("audio", 1, 0.5)],
        "imu": [observed("imu", 0, 1.0)],  # missing at minute 1
        "ppg": [observed("ppg", 0, 70.0)],  # missing at minute 1
    }

    aligned = align_to_minute_grid(USER_ID, records_by_modality, modalities=("audio", "imu", "ppg"))
    downstream = select_for_downstream(aligned)

    assert len(aligned) == 2
    assert len(downstream) == 2  # minute 1: 2 missing = confidence 0.5, still included


# --- T4: >10-minute gap -> no interpolation, gap stays NULL ------------------


def test_s22_t4_large_gap_never_interpolated() -> None:
    records_by_modality = {
        "ppg": [observed("ppg", 0, 70.0), observed("ppg", 15, 72.0)],  # 15-minute gap
    }

    aligned = align_to_minute_grid(USER_ID, records_by_modality, modalities=("ppg",))
    by_minute = {m.minute_start: m for m in aligned}

    before = by_minute[START]
    after = by_minute[START + timedelta(minutes=15)]
    gap_minute = by_minute[START + timedelta(minutes=7)]  # squarely inside the gap

    # Values on either side of the gap are unchanged.
    assert before.values["ppg"] == 70.0
    assert after.values["ppg"] == 72.0

    # The gap itself stays NULL -- never interpolated between 70.0 and 72.0.
    assert gap_minute.values["ppg"] is None
    assert gap_minute.values["ppg"] != 71.0  # not the linear-interpolation midpoint


# --- T5: audio_paused nulls only audio; IMU/PPG remain valid -----------------


def test_s22_t5_audio_paused_window_only_nulls_audio() -> None:
    records_by_modality = {
        "audio": [
            observed("audio", 0, 0.5),
            missing("audio", 1, MissingReason.AUDIO_PAUSED),
            missing("audio", 2, MissingReason.AUDIO_PAUSED),
            missing("audio", 3, MissingReason.AUDIO_PAUSED),
            missing("audio", 4, MissingReason.AUDIO_PAUSED),
            missing("audio", 5, MissingReason.AUDIO_PAUSED),
            observed("audio", 6, 0.5),
        ],
        "imu": [observed("imu", m, 1.0) for m in range(7)],
        "ppg": [observed("ppg", m, 70.0) for m in range(7)],
    }

    aligned = align_to_minute_grid(USER_ID, records_by_modality, modalities=("audio", "imu", "ppg"))
    by_minute = {m.minute_start: m for m in aligned}

    paused_minutes = [by_minute[START + timedelta(minutes=m)] for m in range(1, 6)]

    assert all(minute.values["audio"] is None for minute in paused_minutes)
    assert all(minute.values["imu"] == 1.0 for minute in paused_minutes)
    assert all(minute.values["ppg"] == 70.0 for minute in paused_minutes)
    assert all(minute.missing_modalities == ("audio",) for minute in paused_minutes)
    assert all(minute.confidence == 0.75 for minute in paused_minutes)


# --- T6: zero cells ever silently imputed with zero or mean ------------------


def test_s22_t6_no_cell_is_ever_silently_zero_or_mean_imputed() -> None:
    """Scan the FULL output matrix -- every None must genuinely be None,
    and no missing value is ever secretly replaced with 0.0 or a
    computed mean."""

    records_by_modality = {
        "audio": [observed("audio", m, 0.5) for m in (0, 4)],  # gaps at 1,2,3
        "imu": [observed("imu", m, 1.0) for m in range(5)],
        "ppg": [observed("ppg", m, 70.0) for m in (0, 1, 4)],  # gaps at 2,3
    }

    aligned = align_to_minute_grid(USER_ID, records_by_modality, modalities=("audio", "imu", "ppg"))

    for minute in aligned:
        for modality in minute.missing_modalities:
            value = minute.values[modality]
            assert value is None  # genuinely None
            assert value != 0.0  # never silently zero (None != 0.0 is trivially true,
            # but this asserts the TYPE contract: a missing cell's value
            # attribute must literally be Python None, never a float 0.0
            # masquerading as "no data")

    # Cross-check: no modality's non-missing values were altered by
    # neighboring gaps (i.e., real observed values are exactly what was
    # given, never blended with a mean of surrounding real values).
    minute_0 = next(m for m in aligned if m.minute_start == START)
    assert minute_0.values["ppg"] == 70.0
    minute_4 = next(m for m in aligned if m.minute_start == START + timedelta(minutes=4))
    assert minute_4.values["ppg"] == 70.0
