"""
upstream_interfaces.py

CONTRACT FILE -- not a deliverable of any upstream sprint.

This file defines the exact data shapes Sprint 11 (Auxiliary Intelligence
Modules) expects to receive from earlier sprints. Nothing in this file does
real fitting/modeling -- it only describes shapes, the same pattern Mansi's
team used in Sprint 8/9's own upstream_interfaces.py.

Sources these shapes are based on (real samples received, not guesses):
  - Sprint 3 (BACKBONE / Palash):  BehavioralStateRecord (m_t, p_t)
  - Sprint 6 (CHRONOS / Rohit):    DomainRecord, AlignmentResult
  - Sprint 1 (FOUNDRY / Alok):     FeatureRecord (timestamp, missing-data typing)
  - Sprint 9 (INVENTORS / Mansi):  Claim, ClaimLevel, GateEvaluation

When the real upstream packages are wired in:
  - Delete this file.
  - Import the real types directly from each team's module instead.
  - echo_detection.py / weather_forecast.py / etc. only depend on the
    shapes below, never on how m_t/p_t/domains were actually fit -- so
    the swap should be a pure import change, nothing else.

WHAT IS STILL MISSING (do not build against a guess for these):
  - Social context per session (who was present, what kind of setting) --
    needed by Echo Detection's "matching social context" condition.
    Not yet received from FOUNDRY. Left as an explicit TODO below.
  - Turn-taking + PPG data for Silence Map -- not yet received.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# From Sprint 3 (BACKBONE / Palash) -- m_t / p_t per timestamp per user.
# Matches the real sample JSON Palash sent on Aug 23.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeState:
    """p_t -- the discrete behavioral regime at one timestamp."""
    regime_label: int                    # integer in [0, K-1]
    regime_posterior: Sequence[float]    # length K, sums to ~1.0


@dataclass(frozen=True)
class BehavioralStateRecord:
    """One user, one timestamp: m_t (continuous) + p_t (discrete regime)."""
    user_id: str
    timestamp: datetime
    m_t: Sequence[float]        # 8-15 floats, PCA-reduced behavioral vector
    p_t: RegimeState

    # TODO (blocked on FOUNDRY): social context is not yet part of this
    # record. Echo Detection needs it. Do not fake this field -- leave it
    # absent until a real sample arrives, and gate Echo Detection's
    # "context match" condition accordingly (see echo_detection.py).


# ---------------------------------------------------------------------------
# From Sprint 6 (CHRONOS / Rohit) -- domain_emergence output.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainRecord:
    """Mirrors Rohit's DomainRecord (domain_emergence/domain_lifecycle.py)."""
    domain_id: int
    active: bool
    status: str                 # "active" | "candidate" (domain_confidence.py)
    confidence: float
    parent_ids: Sequence[int] = field(default_factory=list)
    child_ids: Sequence[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# From Sprint 1 (FOUNDRY / Alok) -- canonical FeatureRecord.
# ---------------------------------------------------------------------------

class MeasurementStatus(Enum):
    OBSERVED = "observed"
    MISSING = "missing"


class MissingReason(Enum):
    SENSOR_FAILURE = "sensor_failure"
    NOT_WORN = "not_worn"
    AUDIO_PAUSED = "audio_paused"


@dataclass(frozen=True)
class FeatureRecord:
    """Mirrors Alok's schema.models.FeatureRecord. Never treat MISSING as 0."""
    user_id: str
    timestamp: datetime
    feature_name: str
    value: Optional[float]
    modality: str
    status: MeasurementStatus
    missing_reason: Optional[MissingReason] = None


# ---------------------------------------------------------------------------
# From Sprint 9 (INVENTORS / Mansi) -- claims_engine/claim_levels.py.
# Re-declared here (not imported) so Sprint 11 can be built/tested before
# the real claims_engine package is on our PYTHONPATH.
# ---------------------------------------------------------------------------

class ClaimLevel(Enum):
    LEVEL_0 = 0
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class GateEvaluation:
    level: ClaimLevel
    admissible: bool
    checks: Sequence[GateCheck]


@dataclass(frozen=True)
class Claim:
    """Mirrors Mansi's Claim. For Behavioral DNA export: filter on
    level == LEVEL_3 AND gate_evaluation.admissible == True -- a Claim
    object can exist and still be inadmissible, so level alone is not
    enough."""
    claim_id: str
    user_id: str
    domain_id: str
    level: ClaimLevel
    dominant_divergence_type: Optional[str]
    gate_evaluation: GateEvaluation
    created_at: datetime


@dataclass(frozen=True)
class SessionExcerpt:
    """Mirrors Mansi's SessionExcerpt (upstream_interfaces.py, Sprint 8/9).
    A retrievable, citable unit of evidence for grounded generation."""
    session_id: str
    user_id: str
    timestamp: datetime
    text: str
    contribution_score: float
    is_near_miss: bool = False