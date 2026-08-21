from phase_transition.rupture import RuptureDetector, SensorSnapshot


def test_all_four_conditions_met_declares_rupture():
    det = RuptureDetector()
    snap = SensorSnapshot(
        timestamp=100.0,
        voice_energy_sigma=3.5,
        ppg_hr_pct_above_baseline=45.0,
        cse_salience_level=5,
        cse_salience_duration_min=12.0,
        imu_motion_disruption=True,
    )
    result = det.is_rupture(snap)
    assert result["rupture_declared"]


def test_missing_one_condition_blocks_rupture():
    det = RuptureDetector()
    # everything met EXCEPT imu motion disruption
    snap = SensorSnapshot(
        timestamp=100.0,
        voice_energy_sigma=3.5,
        ppg_hr_pct_above_baseline=45.0,
        cse_salience_level=5,
        cse_salience_duration_min=12.0,
        imu_motion_disruption=False,
    )
    result = det.is_rupture(snap)
    assert not result["rupture_declared"]
    assert result["cond_imu_motion"] is False


def test_cse_duration_too_short_blocks_rupture():
    det = RuptureDetector()
    snap = SensorSnapshot(
        timestamp=100.0,
        voice_energy_sigma=3.5,
        ppg_hr_pct_above_baseline=45.0,
        cse_salience_level=5,
        cse_salience_duration_min=5.0,  # under 10 min threshold
        imu_motion_disruption=True,
    )
    result = det.is_rupture(snap)
    assert not result["rupture_declared"]


def test_ordinary_baseline_data_never_declares():
    det = RuptureDetector()
    snap = SensorSnapshot(
        timestamp=100.0,
        voice_energy_sigma=0.5,
        ppg_hr_pct_above_baseline=5.0,
        cse_salience_level=2,
        cse_salience_duration_min=3.0,
        imu_motion_disruption=False,
    )
    result = det.is_rupture(snap)
    assert not result["rupture_declared"]