"""
upstream_interfaces.py

CONTRACT FILE — not a deliverable of Sprint 8 or Sprint 9.

Sprint 8 (Divergence Engine) and Sprint 9 (Claims Engine) are built downstream of:
  - Sprint 3: HSSM  -> p_t (discrete behavioral regime), m_t (continuous behavioral state)
  - Sprint 4: Attractor detection -> AttractorRecord (per user/regime/context)
  - Sprint 6: Domain emergence -> Domain (behavioral+narrative aligned domain)
  - Sprint 7: NSSM  -> q_t (discrete narrative regime), n_t (continuous narrative state)

This file defines the exact data shapes Sprint 8/9 code expects from those sprints,
as small, typed dataclasses / Protocols, with NO real fitting logic inside them.

When you wire this into the real chronis-ml repo:
  - Delete this file.
  - Import the real HSSM/NSSM/Attractor/Domain types from Sprint 3/4/6/7 modules instead.
  - Everything in divergence_engine/ and claims_engine/ only depends on the shapes below,
    never on how p_t/q_t were actually fit — so the swap should be a pure import change.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence
import numpy as np


@dataclass(frozen=True)
class RegimeSeries:
    """
    A fitted slow-regime process for one user, one system (behavioral or narrative).

    Mirrors Sprint 3 Day 7-9 (HSSM: p_t) and Sprint 7 Day 20 (NSSM: q_t) output shape.
    """
    user_id: str
    system: str  # "behavioral" | "narrative"
    session_ids: Sequence[str]            # session index -> session id, length T
    timestamps: Sequence[datetime]        # length T, NTP-aligned
    regime_labels: np.ndarray             # shape (T,), dtype int, canonical-sorted regime id per session
    n_regimes: int                        # K (behavioral) or J (narrative)
    fast_state: np.ndarray                # shape (T, d) — m_t (behavioral) or n_t (narrative)
    in_fit_set: np.ndarray                # shape (T,), dtype bool.
    # For behavioral: True once cold-start gate (>=30 sessions) is cleared.
    # For narrative: True only for sessions counted toward |S| (Sprint 7's narrative-density gate).
    gated: bool                           # True if this user/system never cleared its cold-start gate at all


@dataclass(frozen=True)
class AttractorRecord:
    """Mirrors Sprint 4 output: a declared attractor for one user/regime/context."""
    user_id: str
    regime_id: int
    context_key: str
    revisit_count: int
    mean_dwell_time: float
    transition_stability: float
    declared: bool  # True only if all 3 stats cleared person-calibrated N/T (hard AND)


@dataclass(frozen=True)
class Domain:
    """Mirrors Sprint 6 output: a behaviorally+narratively aligned domain."""
    domain_id: str
    user_id: str
    label: str
    behavioral_regime_ids: Sequence[int]
    narrative_regime_ids: Sequence[int]
    confidence: float           # observation count + persistence + cross-phase survival + coherence
    active: bool                # False if superseded by a split (parent kept inactive, never deleted)
    high_ignorance_prior: bool  # behavioral cluster, no narrative partner
    aspirational_or_hypothetical: bool  # narrative cluster, no behavioral partner


@dataclass(frozen=True)
class SessionExcerpt:
    """A retrievable, citable unit of evidence — a transcript/session slice."""
    session_id: str
    user_id: str
    timestamp: datetime
    text: str                 # wearer-only transcript excerpt (already policy-cleared)
    contribution_score: float # how strongly this session supports the pattern being cited
    is_near_miss: bool = False  # approached the attractor basin but did not enter it


@dataclass(frozen=True)
class SelfReflectionSessionFlag:
    """Mirrors whatever Sprint-9-adjacent tagging marks a session as self-reflection-mode."""
    session_id: str
    user_id: str
    timestamp: datetime
    is_self_reflection_mode: bool
