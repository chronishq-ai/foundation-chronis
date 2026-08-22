import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "bocd"))

import numpy as np
from bocd import bocd, GaussianUnknownMean


class ChangepointDetector:
    """
    Wraps Adams & MacKay 2007 BOCPD (gwgundersen/bocd reference impl).
    IMPORTANT: changepoint_prob output here is ONLY condition 1 of 3.
    Never call this a declared transition on its own — Day 14 adds
    conditions 2 (predictive fit degradation) and 3 (regime stability),
    composed in gate.py.
    """

    def __init__(self, hazard: float = 1/250, mean0: float = 0.0,
                 var0: float = 2.0, varx: float = 1.0):
        self.hazard = hazard
        self.mean0 = mean0
        self.var0 = var0
        self.varx = varx

    def run(self, data: list[float]) -> dict:
        """Run BOCPD over full data array. Returns run-length posterior
        matrix R plus predictive mean/var."""
        model = GaussianUnknownMean(self.mean0, self.var0, self.varx)
        R, pmean, pvar = bocd(data, model, self.hazard)
        return {"R": R, "pmean": pmean, "pvar": pvar}

    def run_length_map(self, data: list[float]) -> np.ndarray:
        """MAP (most likely) run length at each timestep.
        After a real changepoint, this collapses back toward 0."""
        result = self.run(data)
        R = result["R"]
        T = len(data)
        return np.array([np.argmax(R[t, :t + 1]) for t in range(1, T + 1)])

    def short_runlength_mass(self, data: list[float], window: int = 3) -> np.ndarray:
        """P(run length <= window) at each t — the real changepoint signal.
        Spikes right after a genuine changepoint since belief mass
        concentrates on short run lengths."""
        result = self.run(data)
        R = result["R"]
        T = len(data)
        return np.array([R[t, :min(window + 1, t + 1)].sum() for t in range(1, T + 1)])

    def candidate_changepoints(self, data: list[float],
                                 window: int = 3, threshold: float = 0.5) -> list[int]:
        """Timesteps where short-run-length mass crosses threshold.
        CANDIDATES only — feeds condition 1 of the 3-condition gate."""
        mass = self.short_runlength_mass(data, window=window)
        return [t for t, p in enumerate(mass) if p > threshold]


def hazard_sensitivity_sweep(
    data: list[float],
    hazard_values: list[float],
    window: int = 3,
    threshold: float = 0.5,
    mean0: float = 0.0,
    var0: float = 2.0,
    varx: float = 1.0,
) -> dict:
    """S56.3 (Harnesses tier, intern-owned): sweeps a fixed set of hazard
    rates over the SAME data/seed and reports the resulting candidate
    changepoint count/timing per hazard, so a single arbitrary hazard's
    influence on timing is visible rather than assumed. Diagnostic report
    only -- no pass/fail gate, senior interprets and sets any calibration
    policy (per S56.3 Test Sheet: 'output is a report only')."""
    curve = {}
    for hazard in hazard_values:
        det = ChangepointDetector(hazard=hazard, mean0=mean0, var0=var0, varx=varx)
        cps = det.candidate_changepoints(data, window=window, threshold=threshold)
        curve[hazard] = {"n_candidates": len(cps), "candidate_timesteps": cps}
    return curve