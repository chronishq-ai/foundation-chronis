"""
CHRONIS — Team 4 (INVENTORS) — Sprint 7, Day 21
Narrative-Density Gate, Cross-System Wiring, Synthetic Validation

WHAT THIS FILE DOES, IN PLAIN ENGLISH
--------------------------------------
Day 19 built the label layer. Day 20 built the NSSM (the narrative regime
model). Day 21 does three things to actually close out Sprint 7:

  1. NARRATIVE-DENSITY GATE — not every session has enough genuine
     first-person self-narration to be worth fitting on. A session that's
     mostly small talk shouldn't count toward the NSSM's fit set any more
     than a sensor gap with less than 10 minutes of data should count as
     real signal (Sprint 1's rule). We count "wearer-attributed, agentive
     first-person clauses" (roughly: "I <verb>..." constructions) per
     session and only let a session into the fit set S once it clears a
     PERSON-CALIBRATED minimum — never a fixed global number, for exactly
     the same reason Sprint 4's attractor thresholds N and T are
     person-calibrated: two people at very different baseline verbosity
     shouldn't be held to the same bar.

  2. CROSS-SYSTEM WIRING FOR SPRINT 8 — Sprint 8 (Days 22-24) needs two
     things from us, unmodified: (a) a windowed contingency table between
     the behavioral regime p_t and the narrative regime q_t, tested with
     Fisher's exact test and Bonferroni-corrected across every pair
     tested — reusing Sprint 6 Day 17's exact procedure; and (b) prepared
     (m_t, n_t) series ready for a Bayesian MS-VAR Granger-causality test,
     which Sprint 8 owns. We build the plumbing and the interfaces here;
     we do not reinvent Sprint 6's or Sprint 8's actual math.

  3. THE 20-OBSERVATION-PER-REGIME POWER GATE (MP-09) — below 20
     observations in a given regime, the Granger test simply does not
     run, and no Level 2 claim can be created for that pair. No
     exceptions, no overrides, computed over |S| (the gated fit set from
     step 1) — never the raw session count.

  4. SYNTHETIC VALIDATION — before Sprint 8 is allowed to build on any of
     this, we plant 3 known narrative regime patterns (sustained agentic,
     sustained passive, oscillating/ambivalent) into synthetic sessions,
     run the FULL Day 19 -> Day 20 -> Day 21 pipeline on them blind (the
     NSSM never sees the planted labels), decode which regime the NSSM
     assigned to each session, and check that its guesses line up with
     the planted ground truth more than 75% of the time. This is the same
     75% bar Sprint 7's own Definition of Done and Sprint 8's Divergence
     Engine both have to clear — proving it here, on data we control, is
     what earns this module the right to be trusted downstream.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
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
# "Agentive first-person clause" here is used in the broad, GRAMMATICAL
# sense the directive intends for a density/verbosity gate: any clause
# where "I" is the subject of a real action or experience verb. This is
# deliberately NOT the same thing as the "self_role" dimension from Day 19
# (which asks HOW the person frames their agency — hero, victim, or
# observer). Density asks a simpler, prior question: is there enough
# first-person material here at all to fit a model on? A person narrating
# a rough week in passive-sounding language ("I couldn't stop it") still
# has real first-person narrative density; the gate should not
# systematically punish that framing — that judgment belongs to self_role,
# not to this gate.
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
    """
    Counts "I <verb>" constructions in the wearer's own transcript. Uses a
    lexicon lookup OR a regular past-tense ("...ed") match as a broad,
    cheap catch-all, since this is a density signal, not a fine-grained
    parse.
    """
    text = transcript.lower()
    count = 0
    for match in _CLAUSE_PATTERN.finditer(text):
        word = match.group(1)
        if word in _AGENTIVE_VERB_LEXICON or word.endswith("ed"):
            count += 1
    return count


@dataclass
class NarrativeDensityGate:
    """
    Person-calibrated minimum clause count. A session only enters the NSSM
    fit set S once it clears `min_clauses` for THIS person.

    [REQUIRES SPRINT 4 CALIBRATION HARNESS]
    Sprint 4 Day 11 already built the real calibration procedure: a
    synthetic-trajectory sampler grid-searches candidate thresholds against
    planted true/false conditions and picks the operating point that hits
    a target precision for that user's own data density — this is
    explicitly called out in Day 21's spec as the SAME harness the density
    gate should reuse. That harness lives in Sprint 4's module and isn't
    importable from this codebase yet, so `calibrate()` below is a
    same-shape standalone stand-in (grid search + precision target) to
    unblock Sprint 7. Replace its body with a call into Sprint 4's real
    harness the moment it's exposed as a shared library.
    """
    min_clauses: Optional[int] = None  # set by calibrate(); None means "not yet calibrated"

    def calibrate(
        self,
        labeled_sessions: List[Tuple[SessionInput, bool]],  # (session, is_genuinely_content_rich)
        candidate_thresholds: Sequence[int] = range(1, 12),
        target_precision: float = 0.9,
    ) -> int:
        """
        [REQUIRES SPRINT 4 CALIBRATION HARNESS] — standalone stand-in.

        Grid-searches `candidate_thresholds`, and for each one measures:
          precision = P(session really is content-rich | gate says "pass")
        exactly mirroring Sprint 4 Day 11's own methodology (grid-search
        N/T against false-positive/false-negative rates on synthetic
        trajectories, pick the operating point hitting the target
        precision for THIS person's own data density).
        """
        counts = [count_agentive_first_person_clauses(s.transcript) for s, _ in labeled_sessions]
        labels = [is_rich for _, is_rich in labeled_sessions]

        best_threshold = candidate_thresholds[0]
        best_precision = -1.0
        for threshold in candidate_thresholds:
            passed = [c >= threshold for c in counts]
            true_positives = sum(1 for p, y in zip(passed, labels) if p and y)
            predicted_positives = sum(passed)
            precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
            # Prefer the lowest threshold that still clears the target
            # precision (a stricter-than-needed gate silently throws away
            # real sessions, which is its own kind of data loss).
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
            raise RuntimeError("Gate not calibrated yet — call calibrate() first (or set min_clauses directly).")
        return count_agentive_first_person_clauses(session.transcript) >= self.min_clauses

    def filter_fit_set(self, sessions: List[SessionInput]) -> List[SessionInput]:
        """Returns the subset S of sessions that clear the gate. Below-gate
        sessions are silently EXCLUDED from S, not zero-filled or
        imputed — this is the "no output at all," not "low-confidence
        output," discipline the Global Standard requires."""
        return [s for s in sessions if self.passes(s)]


# ===========================================================================
# STEP 2 — Cross-system wiring: windowed contingency table + Fisher's exact.
# ===========================================================================
@dataclass
class ContingencyResult:
    behavioral_regime: int
    narrative_regime: int
    table: np.ndarray  # 2x2: [[co-occur, behavioral-only], [narrative-only, neither]]
    odds_ratio: float
    p_value_raw: float
    p_value_bonferroni: float
    significant: bool


def build_windowed_contingency_table(
    behavioral_regime_series: np.ndarray,  # (T,) — placeholder for Sprint 3's p_t
    narrative_regime_series: np.ndarray,   # (T,) — this file's q_t (decoded NSSM regime)
    k: int,  # which behavioral regime
    j: int,  # which narrative regime
) -> np.ndarray:
    """
    Builds one 2x2 contingency table for a single (behavioral regime k,
    narrative regime j) pair, over the SAME time window for both series
    (both series must already be aligned/windowed the same way before
    calling this — that alignment itself is Sprint 6's concern, not
    rebuilt here).

        [[ both k and j active,      k active, j NOT active ],
         [ j active, k NOT active,   neither k nor j active ]]

    A large, lopsided table (most mass on the diagonal) means the two
    regimes tend to show up together far more than chance would predict —
    exactly the "regime co-occupancy" signal Sprint 8 needs.
    """
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
    """
    [REQUIRES SPRINT 6 FISHER'S EXACT]
    Sprint 6 Day 17 already built the production version of this exact
    procedure (windowed contingency table -> Fisher's exact ->
    Bonferroni correction across every pair tested), used there to align
    behavioral clusters with narrative topic clusters into domains. Day
    21's spec calls for reusing that procedure UNMODIFIED for the
    behavioral-regime / narrative-regime comparison Sprint 8 needs. That
    module isn't importable from this codebase yet, so this function is a
    same-math standalone equivalent — swap the call to
    `scipy.stats.fisher_exact` + Bonferroni logic below for a direct call
    into Sprint 6's module as soon as it's exposed.

    BEGINNER NOTE ON FISHER'S EXACT + BONFERRONI
    ------------------------------------------------
    Fisher's exact test asks: "if these two regimes were truly
    independent, how surprising is it that we saw a 2x2 table this
    lopsided?" A small p-value means "very surprising under
    independence" -> real co-occurrence. Because we're about to test
    EVERY (behavioral regime, narrative regime) pair — k_count * j_count
    of them — running each at the usual 0.05 cutoff would let us find
    "significant" results by pure chance far too often (this is the
    classic multiple-comparisons problem). Bonferroni correction fixes
    this bluntly but reliably: divide alpha by the number of tests run,
    so the EFFECTIVE bar each individual test must clear gets stricter
    as we test more pairs.
    """
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
# STEP 3 — The 20-observation-per-regime power gate (MP-09).
# ===========================================================================
def regime_observation_counts(regime_series: np.ndarray, regime_count: int) -> np.ndarray:
    """How many sessions in the GATED fit set S landed in each regime."""
    return np.array([int(np.sum(regime_series == r)) for r in range(regime_count)])


def power_gate_ok(n_obs_behavioral: int, n_obs_narrative: int, min_obs: int = 20) -> bool:
    """
    MP-09, enforced exactly as the directive states it: below threshold,
    the Granger test does not run at all, and no Level 2 claim can be
    created for that pair — no exceptions, no overrides. This function is
    the single hard boolean gate everything else in this file routes
    through before calling the (stubbed) Granger test.
    """
    return n_obs_behavioral >= min_obs and n_obs_narrative >= min_obs


# ===========================================================================
# STEP 4 — Prepared inputs for Sprint 8's Granger / Bayesian MS-VAR test.
# ===========================================================================
@dataclass
class GrangerInputBundle:
    behavioral_regime: int
    narrative_regime: int
    m_t: np.ndarray  # behavioral fast state series, restricted to sessions in this regime pair's window
    n_t: np.ndarray  # narrative fast state series (this file's Day 20 NSSM output), same window
    n_obs: int


def prepare_granger_inputs(
    m_t_full: np.ndarray,  # (T,) — placeholder for Sprint 3's fast behavioral state
    n_t_full: np.ndarray,  # (T,) — this file's Day 20 filtered narrative state
    behavioral_regime_series: np.ndarray,
    narrative_regime_series: np.ndarray,
    k: int,
    j: int,
    min_obs: int = 20,
) -> Optional[GrangerInputBundle]:
    """
    Restricts (m_t, n_t) to the sessions where this specific
    (behavioral regime k, narrative regime j) pair is jointly active, and
    enforces the power gate BEFORE handing anything to Sprint 8. Returns
    None (not an empty-but-truthy bundle) when the gate fails — callers
    must treat None as "do not run Granger, do not create a Level 2
    claim for this pair," never as "run it anyway with less power."
    """
    mask = (behavioral_regime_series == k) & (narrative_regime_series == j)
    n_obs = int(mask.sum())
    if not power_gate_ok(n_obs, n_obs, min_obs=min_obs):
        logger.info("Power gate FAILED for (behavioral=%d, narrative=%d): n_obs=%d < %d -> no Granger test", k, j, n_obs, min_obs)
        return None
    return GrangerInputBundle(behavioral_regime=k, narrative_regime=j, m_t=m_t_full[mask], n_t=n_t_full[mask], n_obs=n_obs)


@dataclass
class GrangerTestResult:
    behavioral_regime: int
    narrative_regime: int
    f_statistic: float
    p_value: float
    p_value_bonferroni: float
    significant: bool
    direction: str  # "behavioral_predicts_narrative" | "narrative_predicts_behavioral" | "bidirectional" | "none"


def run_granger_ms_var(bundle: GrangerInputBundle, n_pairs_tested: int, alpha: float = 0.05) -> GrangerTestResult:
    """
    [REQUIRES SPRINT 8 GRANGER MS-VAR]
    Sprint 8 Day 23 owns the real test here: a Bayesian Markov-Switching
    VAR Granger-causality test (Droumaguet, Warne & Wozniak 2017),
    AIC-selected lag order, reporting F-statistic and p-value PER
    DIRECTION, Bonferroni-corrected across domain pairs tested. That is
    real, nontrivial statistical machinery that belongs to Sprint 8, not
    reimplemented here.

    This function is a WIRING-LEVEL STUB ONLY: it proves the interface
    Sprint 8 will be called through actually works end-to-end (bundle in,
    typed result out, power gate already enforced upstream), using a
    plain OLS Granger-causality test as a placeholder computation so this
    file's synthetic validation can run without Sprint 8's engine. The
    p-values and direction this stub reports are NOT the calibrated,
    regime-switching-aware numbers the real system will use downstream —
    do not wire this stub's output into any Level 2 claim.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
        import pandas as pd

        data = pd.DataFrame({"m": bundle.m_t, "n": bundle.n_t})
        # m -> n direction
        result_m_to_n = grangercausalitytests(data[["n", "m"]], maxlag=1, verbose=False)
        f_stat, p_val = result_m_to_n[1][0]["ssr_ftest"][0], result_m_to_n[1][0]["ssr_ftest"][1]
    except Exception:
        logger.warning("Granger stub failed (likely too few observations for OLS); returning null result", exc_info=False)
        f_stat, p_val = 0.0, 1.0

    p_bonf = min(p_val * n_pairs_tested, 1.0)
    significant = p_bonf < alpha
    return GrangerTestResult(
        behavioral_regime=bundle.behavioral_regime,
        narrative_regime=bundle.narrative_regime,
        f_statistic=float(f_stat),
        p_value=float(p_val),
        p_value_bonferroni=float(p_bonf),
        significant=bool(significant),
        direction="behavioral_predicts_narrative" if significant else "none",
    )


# ===========================================================================
# STEP 5 — Orchestration: everything Sprint 8 will actually call.
# ===========================================================================
def prepare_cross_system_inputs(
    behavioral_regime_series: np.ndarray,
    narrative_regime_series: np.ndarray,
    m_t_full: np.ndarray,
    n_t_full: np.ndarray,
    k_count: int,
    j_count: int,
    min_obs: int = 20,
) -> Tuple[List[ContingencyResult], List[GrangerTestResult]]:
    """Top-level Day 21 entry point Sprint 8 wires into: condition 1
    (Fisher's-exact regime co-occupancy) and condition 2 (Granger
    predictability, power-gated) for every (behavioral, narrative) regime
    pair."""
    contingency_results = run_fisher_alignment(behavioral_regime_series, narrative_regime_series, k_count, j_count)

    n_pairs_tested = k_count * j_count
    granger_results: List[GrangerTestResult] = []
    for k in range(k_count):
        for j in range(j_count):
            bundle = prepare_granger_inputs(m_t_full, n_t_full, behavioral_regime_series, narrative_regime_series, k, j, min_obs=min_obs)
            if bundle is None:
                continue  # power gate failed -> no Granger test, no Level 2 claim, by design
            granger_results.append(run_granger_ms_var(bundle, n_pairs_tested=n_pairs_tested))

    return contingency_results, granger_results


# ===========================================================================
# STEP 6 — Synthetic validation: 3 planted narrative regime patterns.
# ===========================================================================
def _generate_sustained_agentic_sessions(n: int, start_day: int) -> List[SessionInput]:
    # Each template also carries >=2 absolutist words ("always"/"definitely"/
    # "completely") so the contradiction-tolerance LF has real evidence for
    # "low_tolerance" here, not just "high_tolerance" evidence from the
    # oscillating block elsewhere — see the module docstring note on why
    # one-directional-only LF evidence silently breaks Dawid-Skene.
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
    """
    The oscillating/ambivalent pattern's STABLE signature is hedging
    narrative style (every template below trips the contradiction-
    tolerance LF via an exact "part of me" / "on the other hand" phrase),
    not a fixed self-role. Which side of self-role a given session leans
    toward is deliberately alternated session-to-session — that alternation
    IS the "oscillating" behavior, while the hedging language is what gives
    this regime a single, recognizable, stable identity for the NSSM to
    lock onto (distinct from either sustained pattern, which never hedges).
    Phrasing was checked directly against the Day 19 labeling functions'
    exact keyword lists so the intended signal actually fires (a phrase
    like "it just happened to me" silently misses the LF's literal
    "it happened to me" match — worth testing LF firing on any new
    synthetic template before trusting it).
    """
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
            prosody_features={"f0_contour_z": 0.0, "energy_envelope_z": 0.0},  # deliberately ambiguous prosody
        ))
    return sessions


PATTERN_NAMES = ["sustained_agentic", "sustained_passive", "oscillating_ambivalent"]


def build_synthetic_validation_set(sessions_per_pattern: int = 6) -> Tuple[List[SessionInput], np.ndarray]:
    """
    Builds one contiguous block per pattern and returns:
      sessions       -> the full session list, in day order
      planted_labels -> shape (T,), the ground-truth pattern index (0/1/2)
                         per session — used ONLY for evaluation afterward,
                         never fed into the WSL or the NSSM fit itself.
    """
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


def best_permutation_accuracy(confusion: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    BEGINNER NOTE: the NSSM has no idea our planted patterns are called
    "sustained_agentic" etc — it just outputs regime index 0, 1, 2 in
    whatever order its own internal fitting happened to land on (this is
    exactly the "label switching" problem Sprint 3 MP-02 already
    solves for the behavioral HSSM). Before we can score "recovery
    accuracy," we first have to find the BEST one-to-one matching between
    decoded regime indices and planted pattern indices — i.e. the
    matching that gives the model the most credit, not an arbitrary one
    that might accidentally compare regime 0 against the wrong pattern.

    This is a classic assignment problem, solved exactly (not
    approximately) via the Hungarian algorithm
    (`scipy.optimize.linear_sum_assignment`), maximizing the number of
    matching sessions on the confusion matrix's diagonal after
    permutation.
    """
    cost = -confusion  # linear_sum_assignment MINIMIZES cost, we want to MAXIMIZE matches
    row_idx, col_idx = linear_sum_assignment(cost)
    matches = confusion[row_idx, col_idx].sum()
    total = confusion.sum()
    accuracy = float(matches / total) if total > 0 else 0.0
    return accuracy, col_idx  # col_idx[i] = which decoded regime best maps to planted pattern i


def evaluate_regime_recovery(planted_labels: np.ndarray, decoded_labels: np.ndarray, pattern_count: int) -> Tuple[float, np.ndarray]:
    confusion = np.zeros((pattern_count, pattern_count), dtype=int)
    for planted, decoded in zip(planted_labels, decoded_labels):
        confusion[planted, decoded] += 1
    accuracy, _ = best_permutation_accuracy(confusion)
    return accuracy, confusion


def assert_regime_recovery_accuracy(accuracy: float, threshold: float = 0.75) -> None:
    """
    Sprint 7's own Definition of Done, and the exact bar Sprint 8 Day 24
    reuses for the Divergence Engine's own type-score validation: regime-
    recovery accuracy on planted synthetic scripts must exceed 75%. This
    is a hard assertion, not a warning — a failing run must stop the
    handoff to Sprint 8, per the Global Standard's "a green checkbox that
    isn't actually true is worse than a red one."
    """
    assert accuracy > threshold, (
        f"Regime-recovery accuracy {accuracy:.1%} did not clear the {threshold:.0%} bar. "
        f"Per the Global Standard: this is NOT eligible for handoff to Sprint 8 until it does."
    )


# ===========================================================================
# STEP 7 — End-to-end synthetic validation run.
# ===========================================================================
# NOTE ON SCALE: as in Day 20, this smoke test intentionally uses REDUCED
# NSSM-fitting settings (fewer random inits, small duration cap) purely so
# it finishes quickly in a sandbox. A real Sprint 7 sign-off run should use
# this module's full defaults (n_random_inits=10) and more planted
# sessions per pattern for a statistically sturdier check.
if __name__ == "__main__":
    sessions_per_pattern = 6
    sessions, planted_labels = build_synthetic_validation_set(sessions_per_pattern=sessions_per_pattern)
    pattern_count = len(PATTERN_NAMES)

    # --- Day 19: WSL ---
    wsl = WeakSupervisionLabelLayer()
    wsl.fit(sessions)
    day19_output = wsl.transform(sessions)

    # --- Narrative-density gate: calibrate against a small labeled sample,
    # then filter the fit set S. For this synthetic set every session was
    # deliberately written to be content-rich, so we expect (and verify)
    # that the gate keeps effectively everyone. ---
    gate = NarrativeDensityGate()
    labeled_for_calibration = [(s, True) for s in sessions[:6]] + [
        (SessionInput(session_id="filler0", transcript="Yeah it was fine, nothing much happened.", prosody_features=None), False),
        (SessionInput(session_id="filler1", transcript="Not much to say today.", prosody_features=None), False),
    ]
    gate.calibrate(labeled_for_calibration, candidate_thresholds=range(1, 8), target_precision=0.9)
    fit_set = gate.filter_fit_set(sessions)
    print(f"\n=== Narrative-density gate ===")
    print(f"min_clauses={gate.min_clauses}; fit set S = {len(fit_set)}/{len(sessions)} sessions passed")

    # Keep planted_labels aligned to whichever sessions actually made it
    # into S (in this synthetic set we expect all of them to pass).
    session_id_to_label = {s.session_id: label for s, label in zip(sessions, planted_labels)}
    fit_set_labels = np.array([session_id_to_label[s.session_id] for s in fit_set])

    # --- Day 20: idiolect normalization + NSSM fit, J fixed at 3 since we
    # KNOW the planted structure for this validation run. (Production
    # fitting always selects J via BIC, as in Day 20 — recovery-accuracy
    # validation is a different exercise: it checks fit QUALITY at a known
    # J, not the model's ability to guess J.) ---
    obs_matrix = np.zeros((len(fit_set), F_DIM))
    sigma_matrix = np.zeros((len(fit_set), F_DIM))
    for i, session in enumerate(fit_set):
        obs, sigma = dimension_outputs_to_observation(day19_output[session.session_id])
        obs_matrix[i] = obs
        sigma_matrix[i] = sigma

    normalizer = IdiolectNormalizer(window_sessions=30, min_sessions_for_baseline=100)  # disabled for this short demo
    normalized_obs = obs_matrix  # too few sessions for a real rolling baseline in this smoke test; see Day 20 for the real path
    emission_var = np.clip(sigma_matrix ** 2, 1e-3, None)

    print("\n=== Fitting NSSM at J=3 (known planted structure) ===")
    # d_max must comfortably cover a planted block's length (6 sessions)
    # or the duration model is forced into a spurious mid-block switch
    # every single block, which corrupts recovery independent of anything
    # else in the pipeline. d_max=6 gives the duration prior room to
    # actually represent "this chapter lasted the whole block."
    fit_result = fit_nssm_for_j(
        normalized_obs, emission_var, j_count=pattern_count,
        d_max=6, n_random_inits=5, optimizer_maxiter=30, seed=11,
    )
    decoded_labels = np.argmax(fit_result.regime_probs, axis=1)

    # --- Evaluate regime recovery against the planted ground truth ---
    accuracy, confusion = evaluate_regime_recovery(fit_set_labels, decoded_labels, pattern_count)
    print(f"\n=== Regime recovery evaluation ===")
    print(f"Planted patterns: {PATTERN_NAMES}")
    print("Confusion matrix (rows=planted, cols=decoded regime index):")
    print(confusion)
    print(f"Best-permutation accuracy: {accuracy:.1%}")

    # This is the actual handoff gate to Sprint 8: a failing run raises
    # here and stops, rather than silently letting Sprint 8 build on an
    # unproven NSSM.
    assert_regime_recovery_accuracy(accuracy, threshold=0.75)
    print("PASSED the 75% regime-recovery bar — eligible for handoff to Sprint 8.")

    # --- Cross-system wiring smoke test (needs a behavioral series; since
    # Sprint 3's real p_t isn't available here, synthesize a placeholder
    # behavioral regime series that's deliberately correlated with the
    # decoded narrative regime, purely to prove the wiring runs end-to-end) ---
    rng = np.random.default_rng(0)
    behavioral_regime_series = np.where(rng.random(len(fit_set)) < 0.8, decoded_labels % 2, 1 - (decoded_labels % 2))
    m_t_placeholder = rng.normal(size=len(fit_set)) + 0.5 * decoded_labels  # [REQUIRES SPRINT 3 HSSM m_t]
    n_t_placeholder = fit_result.filtered_state

    contingency_results, granger_results = prepare_cross_system_inputs(
        behavioral_regime_series, decoded_labels, m_t_placeholder, n_t_placeholder,
        k_count=2, j_count=pattern_count, min_obs=20,
    )
    print(f"\n=== Cross-system wiring smoke test ===")
    print(f"Fisher's-exact pairs tested: {len(contingency_results)}; significant (Bonferroni): "
          f"{sum(r.significant for r in contingency_results)}")
    print(f"Granger tests actually run (power gate passed): {len(granger_results)} "
          f"out of {2 * pattern_count} possible pairs (expected few/none at only "
          f"{len(fit_set)} sessions — this is the 20-observation gate working as intended, "
          f"not a bug: MP-09 is SUPPOSED to block Granger on a session count this small).")
