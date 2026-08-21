from dataclasses import dataclass


@dataclass
class SensorSnapshot:
    """One moment's multi-modal readings, already feature-extracted upstream."""
    timestamp: float
    voice_energy_sigma: float      # how many sigma above personal mean
    ppg_hr_pct_above_baseline: float  # e.g. 45.0 means 45% above baseline
    cse_salience_level: int        # 0-5
    cse_salience_duration_min: float  # minutes spent at current salience level
    imu_motion_disruption: bool    # significant motion disruption flag


class RuptureDetector:
    """
    Module 4.11. A rupture (acute bifurcation event) is declared only if
    ALL 4 conditions hold simultaneously -- hard AND, never a weighted
    score. Slow, gradual shifts are honestly NOT caught here -- this is
    an acute-event detector only.
    """

    VOICE_ENERGY_SIGMA_THRESHOLD = 3.0
    PPG_HR_PCT_THRESHOLD = 40.0
    CSE_SALIENCE_LEVEL_THRESHOLD = 5
    CSE_SALIENCE_DURATION_MIN_THRESHOLD = 10.0

    def is_rupture(self, snapshot: SensorSnapshot) -> dict:
        cond_voice = snapshot.voice_energy_sigma > self.VOICE_ENERGY_SIGMA_THRESHOLD
        cond_ppg = snapshot.ppg_hr_pct_above_baseline > self.PPG_HR_PCT_THRESHOLD
        cond_cse = (snapshot.cse_salience_level >= self.CSE_SALIENCE_LEVEL_THRESHOLD and
                    snapshot.cse_salience_duration_min > self.CSE_SALIENCE_DURATION_MIN_THRESHOLD)
        cond_imu = snapshot.imu_motion_disruption

        declared = cond_voice and cond_ppg and cond_cse and cond_imu

        return {
            "timestamp": snapshot.timestamp,
            "cond_voice_energy": cond_voice,
            "cond_ppg_hr": cond_ppg,
            "cond_cse_salience": cond_cse,
            "cond_imu_motion": cond_imu,
            "rupture_declared": declared,
        }