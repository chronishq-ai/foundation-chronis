"""Hyperparameters for attractor detection/calibration — kept out of code so
they're inspectable and tunable, never hidden magic numbers."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class AttractorConfig:
    neighborhood_radius_multiplier: float = 0.75  # applied to norm(std) of regime's m_t, see detector.py


@dataclass(frozen=True)
class CalibrationConfig:
    n_timesteps: int = 300
    n_trials: int = 20
    target_precision: float = 0.85
    attractor_strength: float = 0.85
    # N split into N_revisit / N_dwell after a real bug: these two stats live on
    # very different numeric scales and cannot share one literal threshold.
    N_revisit_grid: tuple = (2, 4, 6, 9, 13)
    N_dwell_grid: tuple = (0.5, 1.0, 1.5, 2.5, 4.0)
    T_grid_multipliers: tuple = (0.25, 0.5, 0.75, 1.0, 1.5)  # scaled by model's own regime variance


DEFAULT_ATTRACTOR_CONFIG = AttractorConfig()
DEFAULT_CALIBRATION_CONFIG = CalibrationConfig()
