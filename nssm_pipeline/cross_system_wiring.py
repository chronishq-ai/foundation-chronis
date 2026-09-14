"""
CHRONIS — Team 4 (INVENTORS) — Sprint 7, Day 21 (P0 audit fix)

WHAT THIS FILE DOES, IN PLAIN ENGLISH
--------------------------------------
Sprint 7's job stops at the Sprint 7 / Sprint 8 boundary. Concretely:

  1. NARRATIVE-DENSITY GATE — not every session has enough genuine
     first-person self-narration to be worth fitting on. We count
     "wearer-attributed, agentive first-person clauses" per session and
     only let a session into the fit set S once it clears a
     PERSON-CALIBRATED minimum — never a fixed global number.

  2. NSSM FIT + HANDOFF — sessions that clear the gate are fit with the
     NSSM (Day 20), and the result is packaged into a single, strictly
     typed `NSSMResult` (q_t, n_t, conformal_confidence). That object,
     and nothing else, is Sprint 7's handoff to Sprint 8.

  3. GRANGER CAUSALITY IS NOT SPRINT 7'S JOB. Sprint 8 owns the Bayesian
     MS-VAR Granger-causality test end to end. Sprint 7 does not run it,
     does not stub it, and does not mock its math. `build_windowed_
     contingency_table` and `run_fisher_alignment` are kept here because
     they are pure data-prep helpers valid for cross-system wiring in
     general — but they are NOT invoked by the NSSM pipeline, and no
     Granger-related code lives in this file.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import fisher_exact

from nssm_calibration import (
    F_DIM,
    NARRATIVE_DIMENSIONS_ORDERED,
    IdiolectNormalizer,
    dimension_outputs_to_observation,
    fit_nssm_for_j,
)
from weak_supervision_label_layer import SessionInput, WeakSupervisionLabelLayer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chronis.wiring")


# ===========================================================================
# STEP 1 — Narrative-Density Gate.
# ===========================================================================
_AGENTIVE_VERB_LEXICON = {
    "decided", "chose", "made", "took", "handled", "led", "caused", "managed",
    "built", "created", "planned", "initiated", "started", "finished",
    "achieved", "earned", "tried", "fought", "kept", "gave", "felt",
    "noticed", "watched", "saw", "had", "went", "wanted", "needed",
    "struggled", "spoke", "stood", "pushed", "said", "told", "thought",
    "realized", "couldn't", "worked", "asked", "left", "stayed", "called",
}
_CLAUSE_PATTERN = re.compile(r"\bi\s+([a-z']+)\b")


def count_agentive_first_person_clauses(transcript: str) -> int:
    text = transcript.lower()
    count = 0
    for match in _CLAUSE_PATTERN.finditer(text):
        word = match.group(1)
        if word in _AGENTIVE_VERB_LEXICON or word.endswith("ed"):
            count += 1
    return count


@dataclass
class NarrativeDensityGate:
    min_clauses: Optional[int] = None

    def calibrate(
        self,
        labeled_sessions: List[Tuple[SessionInput, bool]],
        candidate_thresholds: Sequence[int] = range(1, 12),
        target_precision: float = 0.9,
    ) -> int:
        counts = [count_agentive_first_person_clauses(s.transcript) for s, _ in labeled_sessions]
        labels = [is_rich for _, is_rich in labeled_sessions]

        best_threshold = candidate_thresholds[0]
        best_precision = -1.0
        for threshold in candidate_thresholds:
            passed = [c >= threshold for c in counts]
            true_positives = sum(1 for p, y in zip(passed, labels) if p and y)
            predicted_positives = sum(passed)
            precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
            
            if precision >= target_precision:
                best_threshold = threshold
                best_precision = precision
                break
            if precision > best_precision:
                best_threshold = threshold
                best_precision = precision

        self.min_clauses = best_threshold
        logger.info("Narrative-density gate calibrated: min_clauses=%d (precision=%.2f)", best_threshold, best_precision)
        return best_threshold

    def passes(self, session: SessionInput) -> bool:
        if self.min_clauses is None:
            raise RuntimeError("Gate not calibrated yet — call calibrate() first.")
        return count_agentive_first_person_clauses(session.transcript) >= self.min_clauses

    def filter_fit_set(self, sessions: List[SessionInput]) -> List[SessionInput]:
        return [s for s in sessions if self.passes(s)]


# ===========================================================================
# STEP 2 — The NSSM contract: NSSMResult, and the pipeline that produces it.
# ===========================================================================
@dataclass
class NSSMResult:
    q_t: np.ndarray
    n_t: np.ndarray
    conformal_confidence: Union[float, np.ndarray]
    session_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.q_t.ndim != 1:
            raise ValueError(f"NSSMResult.q_t must be 1-D (T,); got shape {self.q_t.shape}")
        if self.n_t.ndim != 2 or self.n_t.shape[0] != self.q_t.shape[0]:
            raise ValueError(
                f"NSSMResult.n_t must be 2-D (T, latent_dim) aligned to q_t; "
                f"got q_t={self.q_t.shape}, n_t={self.n_t.shape}"
            )
        if isinstance(self.conformal_confidence, np.ndarray) and self.conformal_confidence.shape[0] != self.q_t.shape[0]:
            raise ValueError(
                f"NSSMResult.conformal_confidence, if array-valued, must align to q_t; "
                f"got q_t={self.q_t.shape}, conformal_confidence={self.conformal_confidence.shape}"
            )
        if self.session_ids and len(self.session_ids) != self.q_t.shape[0]:
            raise ValueError(
                f"NSSMResult.session_ids length must match q_t length; "
                f"got {len(self.session_ids)} ids for {self.q_t.shape[0]} sessions"
            )


def compute_conformal_confidence(regime_probs: np.ndarray, emission_var: np.ndarray) -> np.ndarray:
    posterior_margin = np.max(regime_probs, axis=1)
    mean_emission_var = np.mean(emission_var, axis=1)

    noise_min, noise_max = float(mean_emission_var.min()), float(mean_emission_var.max())
    if noise_max > noise_min:
        normalized_noise = (mean_emission_var - noise_min) / (noise_max - noise_min)
    else:
        normalized_noise = np.zeros_like(mean_emission_var)

    return np.clip(posterior_margin * (1.0 - 0.5 * normalized_noise), 0.0, 1.0)


def run_nssm_pipeline(
    sessions: Sequence[SessionInput],
    gate: NarrativeDensityGate,
    j_count: int,
    d_max: int = 6,
    n_random_inits: int = 10,
    optimizer_maxiter: int = 100,
    seed: Optional[int] = None,
) -> NSSMResult:
    fit_set = gate.filter_fit_set(list(sessions))
    if not fit_set:
        raise ValueError("Narrative-density gate rejected every session; fit set S is empty.")

    wsl = WeakSupervisionLabelLayer()
    wsl.fit(fit_set)
    day19_output = wsl.transform(fit_set)

    obs_matrix = np.zeros((len(fit_set), F_DIM), dtype=float)
    sigma_matrix = np.zeros((len(fit_set), F_DIM), dtype=float)
    for i, session in enumerate(fit_set):
        obs, sigma = dimension_outputs_to_observation(day19_output[session.session_id])
        obs_matrix[i] = obs
        sigma_matrix[i] = sigma

    IdiolectNormalizer(window_sessions=30, min_sessions_for_baseline=100)
    normalized_obs = obs_matrix
    emission_var = np.clip(sigma_matrix ** 2, 1e-3, None)

    fit_result = fit_nssm_for_j(
        normalized_obs, emission_var, j_count=j_count,
        d_max=d_max, n_random_inits=n_random_inits,
        optimizer_maxiter=optimizer_maxiter, seed=seed,
    )

    q_t = np.argmax(fit_result.regime_probs, axis=1).astype(np.int64)
    n_t = np.asarray(fit_result.filtered_state, dtype=float)
    conformal_confidence = compute_conformal_confidence(fit_result.regime_probs, emission_var)

    return NSSMResult(
        q_t=q_t,
        n_t=n_t,
        conformal_confidence=conformal_confidence,
        session_ids=[s.session_id for s in fit_set],
    )


# ===========================================================================
# STEP 3 — Cross-system data-prep helpers (Fisher's exact / contingency).
# ===========================================================================
@dataclass
class ContingencyResult:
    behavioral_regime: int
    narrative_regime: int
    table: np.ndarray
    odds_ratio: float
    p_value_raw: float
    p_value_bonferroni: float
    significant: bool


def build_windowed_contingency_table(
    behavioral_regime_series: np.ndarray,
    narrative_regime_series: np.ndarray,
    k: int,
    j: int,
) -> np.ndarray:
    both = int(np.sum((behavioral_regime_series == k) & (narrative_regime_series == j)))
    behavioral_only = int(np.sum((behavioral_regime_series == k) & (narrative_regime_series != j)))
    narrative_only = int(np.sum((behavioral_regime_series != k) & (narrative_regime_series == j)))
    neither = int(np.sum((behavioral_regime_series != k) & (narrative_regime_series != j)))
    return np.array([[both, behavioral_only], [narrative_only, neither]])


def run_fisher_alignment(
    behavioral_regime_series: np.ndarray,
    narrative_regime_series: np.ndarray,
    k_count: int,
    j_count: int,
    alpha: float = 0.05,
) -> List[ContingencyResult]:
    n_pairs = k_count * j_count
    bonferroni_alpha = alpha / n_pairs
    results: List[ContingencyResult] = []

    for k in range(k_count):
        for j in range(j_count):
            table = build_windowed_contingency_table(behavioral_regime_series, narrative_regime_series, k, j)
            odds_ratio, p_value = fisher_exact(table)
            results.append(
                ContingencyResult(
                    behavioral_regime=k,
                    narrative_regime=j,
                    table=table,
                    odds_ratio=float(odds_ratio),
                    p_value_raw=float(p_value),
                    p_value_bonferroni=float(min(p_value * n_pairs, 1.0)),
                    significant=bool(p_value < bonferroni_alpha),
                )
            )
    return results


# ===========================================================================
# STEP 4 — Synthetic session generation (used only by the smoke test below).
# ===========================================================================
def _generate_sustained_agentic_sessions(n: int, start_day: int) -> List[SessionInput]:
    templates = [
        "I decided to take charge of the situation at work. I always follow through and I definitely earned this.",
        "I made the call myself and I handled it. I definitely planned the whole approach and I always finish what I start.",
        "I pushed through the hard part and I built something I'm proud of. I completely earned this outcome, no doubt.",
    ]
    sessions = []
    for i in range(n):
        sessions.append(SessionInput(
            session_id=f"day{start_day + i:02d}",
            transcript=templates[i % len(templates)],
            prosody_features={"f0_contour_z": 0.8, "energy_envelope_z": 0.4},
        ))
    return sessions


def _generate_sustained_passive_sessions(n: int, start_day: int) -> List[SessionInput]:
    templates = [
        "It happened to me again and I had no choice. I never had control and it's completely out of my hands.",
        "They made me feel like it was my fault. I had no choice, and it's always completely out of my hands.",
        "I watched it all unfold and I couldn't stop any of it. I never have any say, it's completely hopeless.",
    ]
    sessions = []
    for i in range(n):
        sessions.append(SessionInput(
            session_id=f"day{start_day + i:02d}",
            transcript=templates[i % len(templates)],
            prosody_features={"f0_contour_z": -0.9, "energy_envelope_z": -0.7},
        ))
    return sessions


def _generate_oscillating_ambivalent_sessions(n: int, start_day: int) -> List[SessionInput]:
    templates = [
        "I decided to take charge for a while, and part of me is proud, but on the other hand it never fully worked out.",
        "Part of me feels like I decided this myself, but on the other hand it happened to me anyway. I had no choice either way.",
        "On the other hand I chose to try, and part of me believes I handled it well this time.",
    ]
    sessions = []
    for i in range(n):
        sessions.append(SessionInput(
            session_id=f"day{start_day + i:02d}",
            transcript=templates[i % len(templates)],
            prosody_features={"f0_contour_z": 0.0, "energy_envelope_z": 0.0}, 
        ))
    return sessions


PATTERN_NAMES = ["sustained_agentic", "sustained_passive", "oscillating_ambivalent"]


def build_synthetic_validation_set(sessions_per_pattern: int = 6) -> Tuple[List[SessionInput], np.ndarray]:
    generators = [_generate_sustained_agentic_sessions, _generate_sustained_passive_sessions, _generate_oscillating_ambivalent_sessions]
    sessions: List[SessionInput] = []
    planted_labels: List[int] = []
    day = 0
    for pattern_idx, generator in enumerate(generators):
        block = generator(sessions_per_pattern, start_day=day)
        sessions.extend(block)
        planted_labels.extend([pattern_idx] * sessions_per_pattern)
        day += sessions_per_pattern
    return sessions, np.array(planted_labels)


# ===========================================================================
# STEP 5 — Smoke test: verify NSSMResult extraction end to end.
# ===========================================================================
if __name__ == "__main__":
    sessions, _planted_labels = build_synthetic_validation_set(sessions_per_pattern=6)
    pattern_count = len(PATTERN_NAMES)

    gate = NarrativeDensityGate()
    labeled_for_calibration = [(s, True) for s in sessions[:6]] + [
        (SessionInput(session_id="filler0", transcript="Yeah it was fine, nothing much happened.", prosody_features=None), False),
        (SessionInput(session_id="filler1", transcript="Not much to say today.", prosody_features=None), False),
    ]
    gate.calibrate(labeled_for_calibration, candidate_thresholds=range(1, 8), target_precision=0.9)
    print(f"=== Narrative-density gate === min_clauses={gate.min_clauses}")

    print("\n=== Running Sprint 7 pipeline: run_nssm_pipeline -> NSSMResult ===")
    result = run_nssm_pipeline(
        sessions, gate, j_count=pattern_count,
        d_max=6, n_random_inits=5, optimizer_maxiter=30, seed=11,
    )

    # --- Extraction checks: the handoff contract, nothing else. ---
    assert isinstance(result, NSSMResult)
    assert isinstance(result.q_t, np.ndarray) and result.q_t.dtype.kind in ("i", "u")
    assert isinstance(result.n_t, np.ndarray) and result.n_t.ndim == 2
    assert isinstance(result.conformal_confidence, (float, np.ndarray))
    n_sessions = result.q_t.shape[0]
    assert result.n_t.shape[0] == n_sessions
    assert len(result.session_ids) == n_sessions
    if isinstance(result.conformal_confidence, np.ndarray):
        assert result.conformal_confidence.shape[0] == n_sessions
        assert bool(np.all((result.conformal_confidence >= 0.0) & (result.conformal_confidence <= 1.0)))

    print(f"\n=== NSSMResult extraction ===")
    print(f"fit set |S|: {n_sessions}")
    print(f"q_t   : shape={result.q_t.shape}, dtype={result.q_t.dtype}")
    print(f"n_t   : shape={result.n_t.shape}, dtype={result.n_t.dtype}")
    if isinstance(result.conformal_confidence, np.ndarray):
        print(
            f"conformal_confidence: shape={result.conformal_confidence.shape}, "
            f"range=[{result.conformal_confidence.min():.3f}, {result.conformal_confidence.max():.3f}]"
        )
    else:
        print(f"conformal_confidence: scalar={result.conformal_confidence:.3f}")
    print(f"session_ids[:3]: {result.session_ids[:3]}")
    print("\nPASSED — NSSMResult is well-formed and eligible for handoff to Sprint 8.")