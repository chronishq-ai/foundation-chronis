"""
synthetic/profiles.py

Sprint 8/9/15, Day 24-27 — synthetic validation suite for the end-to-end
divergence pipeline.

S15.1 / S79.5 FIX: the previous validation suite (`planted_profiles.py`)
hand-planted regime-label arrays (p_t, q_t) and fast-state arrays (m_t, n_t)
directly into `DivergenceInputs`, bypassing state estimation entirely. That
made the suite's >75%-accuracy claim a claim about `compute_divergence_state`
given PERFECT, already-labeled inputs — it said nothing about whether the
pipeline can recover regimes and fast-states from noisy raw observations,
which is what production actually has to do. It also went stale the moment
`granger.py`'s S79.1 rewrite made within-regime Granger a JOINT regime+VAR
estimator (MS-VAR): hand-supplied p_t/q_t regime labels are no longer even
consumed the way they were when Granger ran on pre-sliced windows.

Fixed here: `SyntheticProfileGenerator` plants ground truth at the RAW
observation level only — a discrete latent regime path (semi-Markov, with
controllable behavioral/narrative co-occupancy) driving noisy, higher-
dimensional multivariate observations, with directional causal coupling
(or its planned absence) baked in at the true-latent-signal level, one layer
below what the model ever sees. The harness then runs those raw observations
through the ACTUAL pipeline stages in order:

    raw obs (behavioral, narrative)
      -> fit_hssm(...)                         [state estimation: S15.1]
      -> within_regime_granger_test(m_t, n_t)  [joint MS-VAR Granger: S79.1]
      -> compute_divergence_state(inputs)      [divergence scoring]
      -> evaluate_level2(...)                  [statistical-bypass-fixed gate: S79.3]

and asserts the pipeline recovers the planted dominant divergence type with
>75% accuracy per type across 20+ profiles/type (80+ total) — the actual DoD
from Sprint 8 Day 24 / Sprint 15 Day 44, now measured against something that
resembles what production will actually receive.

ASSUMED INTERFACE — `fit_hssm`:
This suite assumes `divergence_engine.hssm.fit_hssm` exists with the contract
declared in the `HSSMFitter` Protocol below (fit a hidden semi-Markov state-
space model to raw multivariate session observations; return a smoothed
multivariate latent fast-state estimate plus a most-likely discrete regime
path). That module/function is not defined anywhere in this conversation —
if your actual `fit_hssm` has a different signature or return shape, the
import below is the one place to fix; nothing else in this file should need
to change as long as `HSSMFitResult.latent_state` / `.regime_labels` keep
their documented shapes.

ASSUMED TEST DOUBLES — `Domain`, Level-1 `GateEvaluation`:
`evaluate_level2` requires a `Domain` (needs `.confidence`) and a sequence of
already-admissible Level-1 `GateEvaluation`s. This suite is validating the
Level-2 divergence-typing pipeline specifically, not Level-1 attractor
detection or `Domain`'s real constructor (which isn't shown anywhere in this
conversation) — so it uses a minimal local `_StubDomain` duck-typed to just
the `.confidence` attribute `evaluate_level2` reads, and hand-built
already-admissible `GateEvaluation(level=LEVEL_1, ...)` stand-ins, rather
than guessing at `Domain`'s and `AttractorRecord`'s real required fields.
This is a deliberate scope boundary, not an oversight — Level-1/Domain
validation belongs in their own suites.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Protocol, Sequence, Tuple
import numpy as np
import pytest

from divergence_engine.engine import DivergenceInputs, compute_divergence_state
from divergence_engine.granger import within_regime_granger_test, GrangerResult
from divergence_engine.state import DivergenceState
from claims_engine.claim_levels import (
    ClaimLevel,
    GateCheck,
    GateEvaluation,
    evaluate_level2,
)

# See "ASSUMED INTERFACE" note above — fix this import if the real fit_hssm
# lives elsewhere or has a different signature.
from divergence_engine.hssm import fit_hssm  # type: ignore[import-not-found]

RNG_SEED = 42
N_PROFILES_PER_TYPE = 20  # DoD: >=20 planted profiles/type
ACCURACY_TARGET = 0.75


# ---------------------------------------------------------------------------
# Assumed fit_hssm contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HSSMFitResult:
    """Assumed shape of fit_hssm's return value."""
    latent_state: np.ndarray      # (T, d_latent) smoothed multivariate fast-state estimate
    regime_labels: np.ndarray     # (T,) int, most-likely discrete regime path
    n_regimes: int


class HSSMFitter(Protocol):
    def __call__(
        self, raw_obs: np.ndarray, n_regimes: int = 2, **kwargs
    ) -> HSSMFitResult: ...


# ---------------------------------------------------------------------------
# Test doubles for evaluate_level2's non-Granger dependencies
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _StubDomain:
    """Duck-typed stand-in for `upstream_interfaces.Domain` — see module note."""
    confidence: float = 0.9


def _passing_level1_evaluations(n: int = 1) -> List[GateEvaluation]:
    return [
        GateEvaluation(
            level=ClaimLevel.LEVEL_1,
            admissible=True,
            checks=[GateCheck("attractor_declared", True, "Stub: Level-1 assumed pre-passed for this suite.")],
        )
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# Planted regime path generation (ground truth, one layer below the model)
# ---------------------------------------------------------------------------

def _semi_markov_regime_path(
    n_sessions: int,
    occupancy_frac: float,
    mean_dwell: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Planted discrete regime path (0/1) with controllable long-run occupancy
    and controllable dwell time (via a geometric switching probability), so
    the raw observations have genuine regime persistence for fit_hssm to
    recover — not i.i.d. per-session coin flips.
    """
    switch_prob = 1.0 / max(mean_dwell, 1.0)
    path = np.empty(n_sessions, dtype=int)
    path[0] = 1 if rng.random() < occupancy_frac else 0
    for t in range(1, n_sessions):
        if rng.random() < switch_prob:
            path[t] = 1 if rng.random() < occupancy_frac else 0
        else:
            path[t] = path[t - 1]
    return path


def _coupled_regime_paths(
    n_sessions: int,
    p_occupancy: float,
    q_occupancy: float,
    joint_overlap: float,
    mean_dwell: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Planted (behavioral, narrative) regime paths with controllable co-occupancy overlap."""
    p_path = _semi_markov_regime_path(n_sessions, p_occupancy, mean_dwell, rng)
    q_path = np.empty(n_sessions, dtype=int)
    for t in range(n_sessions):
        if p_path[t] == 1 and rng.random() < joint_overlap:
            q_path[t] = 1
        else:
            q_path[t] = 1 if rng.random() < q_occupancy else 0
    return p_path, q_path


def _regime_dependent_offset(regime_path: np.ndarray, d: int, regime1_offset: np.ndarray) -> np.ndarray:
    offsets = np.zeros((len(regime_path), d))
    offsets[regime_path == 1] = regime1_offset
    return offsets


def _raw_observations_from_latent(
    true_latent: np.ndarray,
    regime_path: np.ndarray,
    d_raw: int,
    regime1_offset_scale: float,
    obs_noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Projects a true low-dimensional latent signal into a higher-dimensional,
    regime-offset, noisy raw observation space — so fit_hssm has real
    denoising + regime-recovery work to do, rather than receiving the latent
    state pre-solved.
    """
    d_latent = true_latent.shape[1]
    loading = rng.normal(0, 1, size=(d_latent, d_raw)) / np.sqrt(d_latent)
    projected = true_latent @ loading  # (T, d_raw)

    regime1_offset = rng.normal(0, regime1_offset_scale, size=d_raw)
    offsets = _regime_dependent_offset(regime_path, d_raw, regime1_offset)

    noise = rng.normal(0, obs_noise_std, size=(len(true_latent), d_raw))
    return projected + offsets + noise


# ---------------------------------------------------------------------------
# Synthetic profile generator — plants ground truth at the raw-observation
# level for all four divergence types.
# ---------------------------------------------------------------------------

@dataclass
class PlantedProfile:
    raw_behavioral_obs: np.ndarray  # (T, d_raw_m)
    raw_narrative_obs: np.ndarray   # (T, d_raw_n)
    window_start: datetime
    window_end: datetime
    behavioral_attractor_weakening: bool
    narrative_conformal_confidence: float
    true_type: str


class SyntheticProfileGenerator:
    """Generates raw, multivariate, noisy (behavioral, narrative) observation
    pairs for each of the four divergence types, with directional causal
    structure planted at the TRUE latent signal — never handed to the model
    pre-fit."""

    def __init__(self, d_latent: int = 3, d_raw: int = 6, obs_noise_std: float = 0.6):
        self.d_latent = d_latent
        self.d_raw = d_raw
        self.obs_noise_std = obs_noise_std

    def _base_window(self, n_sessions: int) -> Tuple[datetime, datetime]:
        start = datetime(2026, 1, 1)
        return start, start + timedelta(days=n_sessions)

    def ignorance(self, n_sessions: int = 60, seed: int = 0) -> PlantedProfile:
        """Strong, persistent behavioral regime; near-flat, low-persistence
        narrative signal with ~zero coupling either direction."""
        rng = np.random.default_rng(RNG_SEED + seed)
        p_path, q_path = _coupled_regime_paths(
            n_sessions, p_occupancy=0.85, q_occupancy=0.03, joint_overlap=0.02,
            mean_dwell=8.0, rng=rng,
        )
        true_m = rng.normal(0, 1, size=(n_sessions, self.d_latent))
        true_n = rng.normal(0, 1, size=(n_sessions, self.d_latent)) * 0.1  # near-flat narrative

        raw_m = _raw_observations_from_latent(true_m, p_path, self.d_raw, 0.8, self.obs_noise_std, rng)
        raw_n = _raw_observations_from_latent(true_n, q_path, self.d_raw, 0.2, self.obs_noise_std, rng)

        start, end = self._base_window(n_sessions)
        return PlantedProfile(raw_m, raw_n, start, end, False, 0.7, "ignorance")

    def aspiration(self, n_sessions: int = 100, seed: int = 0) -> PlantedProfile:
        """Weakening behavioral attractor; narrative fast-state LEADS
        behavioral fast-state (n_t predicts m_{t+1}), ~zero reverse coupling."""
        rng = np.random.default_rng(RNG_SEED + 1000 + seed)
        p_path, q_path = _coupled_regime_paths(
            n_sessions, p_occupancy=0.5, q_occupancy=0.6, joint_overlap=0.8,
            mean_dwell=10.0, rng=rng,
        )
        true_n = rng.normal(0, 1, size=(n_sessions, self.d_latent))
        true_m = np.zeros((n_sessions, self.d_latent))
        true_m[0] = rng.normal(0, 1, size=self.d_latent)
        true_m[1:] = 0.85 * true_n[:-1] + rng.normal(0, 0.15, size=(n_sessions - 1, self.d_latent))

        raw_m = _raw_observations_from_latent(true_m, p_path, self.d_raw, 0.5, self.obs_noise_std, rng)
        raw_n = _raw_observations_from_latent(true_n, q_path, self.d_raw, 0.5, self.obs_noise_std, rng)

        start, end = self._base_window(n_sessions)
        return PlantedProfile(raw_m, raw_n, start, end, True, 0.7, "aspiration")

    def self_protection(self, n_sessions: int = 100, seed: int = 0) -> PlantedProfile:
        """Stable, co-occupying behavioral and narrative regimes, but the
        underlying fast-state signals are INDEPENDENT noise — co-occupancy
        without directional Granger coupling either way."""
        rng = np.random.default_rng(RNG_SEED + 2000 + seed)
        p_path, q_path = _coupled_regime_paths(
            n_sessions, p_occupancy=0.85, q_occupancy=0.8, joint_overlap=0.92,
            mean_dwell=12.0, rng=rng,
        )
        true_m = rng.normal(0, 1, size=(n_sessions, self.d_latent))
        true_n = rng.normal(0, 1, size=(n_sessions, self.d_latent))  # independent of true_m

        raw_m = _raw_observations_from_latent(true_m, p_path, self.d_raw, 0.5, self.obs_noise_std, rng)
        raw_n = _raw_observations_from_latent(true_n, q_path, self.d_raw, 0.5, self.obs_noise_std, rng)

        start, end = self._base_window(n_sessions)
        return PlantedProfile(raw_m, raw_n, start, end, False, 0.7, "self_protection")

    def active_transition(self, n_sessions: int = 100, seed: int = 0) -> PlantedProfile:
        """Both systems shifting together: bidirectional lagged coupling,
        co-occupying regimes."""
        rng = np.random.default_rng(RNG_SEED + 3000 + seed)
        p_path, q_path = _coupled_regime_paths(
            n_sessions, p_occupancy=0.6, q_occupancy=0.6, joint_overlap=0.85,
            mean_dwell=9.0, rng=rng,
        )
        base = rng.normal(0, 1, size=(n_sessions, self.d_latent))
        true_m = np.zeros((n_sessions, self.d_latent))
        true_n = np.zeros((n_sessions, self.d_latent))
        true_m[0], true_n[0] = base[0], base[0]
        for t in range(1, n_sessions):
            true_m[t] = 0.7 * true_n[t - 1] + 0.15 * true_m[t - 1] + rng.normal(0, 0.2, size=self.d_latent)
            true_n[t] = 0.7 * true_m[t - 1] + 0.15 * true_n[t - 1] + rng.normal(0, 0.2, size=self.d_latent)

        raw_m = _raw_observations_from_latent(true_m, p_path, self.d_raw, 0.5, self.obs_noise_std, rng)
        raw_n = _raw_observations_from_latent(true_n, q_path, self.d_raw, 0.5, self.obs_noise_std, rng)

        start, end = self._base_window(n_sessions)
        return PlantedProfile(raw_m, raw_n, start, end, True, 0.7, "active_transition")


PROFILE_METHODS: Dict[str, str] = {
    "ignorance": "ignorance",
    "aspiration": "aspiration",
    "self_protection": "self_protection",
    "active_transition": "active_transition",
}


# ---------------------------------------------------------------------------
# End-to-end pipeline runner: raw obs -> fit_hssm -> Granger -> divergence
# state -> Level 2 gate
# ---------------------------------------------------------------------------

@dataclass
class PipelineRunResult:
    profile: PlantedProfile
    hssm_behavioral: HSSMFitResult
    hssm_narrative: HSSMFitResult
    granger_result: GrangerResult
    divergence_state: DivergenceState
    level2_evaluation: GateEvaluation
    predicted_type: str  # dominant divergence type, or "AMBIGUOUS"


def run_profile_through_pipeline(profile: PlantedProfile, n_pairs_tested: int = 1) -> PipelineRunResult:
    n_sessions = profile.raw_behavioral_obs.shape[0]

    # Stage 1 — S15.1: real state estimation, not hand-planted arrays.
    hssm_behavioral = fit_hssm(profile.raw_behavioral_obs, n_regimes=2)
    hssm_narrative = fit_hssm(profile.raw_narrative_obs, n_regimes=2)

    m_t = hssm_behavioral.latent_state
    n_t = hssm_narrative.latent_state
    p_t = hssm_behavioral.regime_labels
    q_t = hssm_narrative.regime_labels

    # Stage 2 — S79.1: joint MS-VAR Granger test, run directly (also feeds
    # into compute_divergence_state's own internal call on the same m_t/n_t,
    # so this doubles as an explicit diagnostic check on the Granger stage
    # in isolation, not just as a black box inside step 3).
    granger_result = within_regime_granger_test(
        m_t=m_t, n_t=n_t, n_pairs_tested=n_pairs_tested,
    )

    # Stage 3 — divergence scoring.
    inputs = DivergenceInputs(
        user_id=f"synthetic-{profile.true_type}",
        domain_id="domX",
        window_start=profile.window_start,
        window_end=profile.window_end,
        p_t=p_t, q_t=q_t, m_t=m_t, n_t=n_t,
        behavioral_regime_id=1, narrative_regime_id=1,
        n_domain_pairs_tested=n_pairs_tested,
        behavioral_attractor_weakening=profile.behavioral_attractor_weakening,
        narrative_conformal_confidence=profile.narrative_conformal_confidence,
    )
    divergence_state = compute_divergence_state(inputs)

    # Stage 4 — S79.3-fixed Level 2 gate.
    level2_evaluation = evaluate_level2(
        level1_evaluations=_passing_level1_evaluations(),
        divergence_state=divergence_state,
        domain=_StubDomain(confidence=0.9),
    )

    predicted = divergence_state.type_scores.dominant()
    predicted_type = predicted if predicted is not None else "AMBIGUOUS"

    return PipelineRunResult(
        profile=profile,
        hssm_behavioral=hssm_behavioral,
        hssm_narrative=hssm_narrative,
        granger_result=granger_result,
        divergence_state=divergence_state,
        level2_evaluation=level2_evaluation,
        predicted_type=predicted_type,
    )


# ---------------------------------------------------------------------------
# pytest suite
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generator() -> SyntheticProfileGenerator:
    return SyntheticProfileGenerator()


@pytest.fixture(scope="module")
def all_pipeline_results(generator: SyntheticProfileGenerator) -> Dict[str, List[PipelineRunResult]]:
    """
    Runs all 4 types x N_PROFILES_PER_TYPE (80+ total) through the real
    pipeline ONCE per test session, so per-type accuracy tests and the
    combined-accuracy test share the same runs rather than re-simulating.
    """
    results: Dict[str, List[PipelineRunResult]] = {t: [] for t in PROFILE_METHODS}
    for true_type, method_name in PROFILE_METHODS.items():
        gen_method = getattr(generator, method_name)
        for i in range(N_PROFILES_PER_TYPE):
            profile = gen_method(seed=i)
            results[true_type].append(run_profile_through_pipeline(profile))
    return results


@pytest.mark.parametrize("true_type", list(PROFILE_METHODS.keys()))
def test_dominant_type_recovery_accuracy(
    all_pipeline_results: Dict[str, List[PipelineRunResult]], true_type: str
):
    """
    DoD (Sprint 8 Day 24 / Sprint 15 Day 44): >75% dominant-divergence-type
    classification accuracy per type, on >=20 planted profiles/type, measured
    against the ACTUAL fit_hssm -> MS-VAR Granger -> divergence-scoring chain
    — not against hand-planted regime/fast-state arrays.
    """
    runs = all_pipeline_results[true_type]
    assert len(runs) >= N_PROFILES_PER_TYPE

    correct = sum(1 for r in runs if r.predicted_type == true_type)
    accuracy = correct / len(runs)

    assert accuracy > ACCURACY_TARGET, (
        f"{true_type}: {accuracy:.1%} accuracy over {len(runs)} profiles, "
        f"target >{ACCURACY_TARGET:.0%}. Predicted-type breakdown: "
        f"{ {t: sum(1 for r in runs if r.predicted_type == t) for t in list(PROFILE_METHODS) + ['AMBIGUOUS']} }"
    )


def test_combined_accuracy_across_all_types(all_pipeline_results: Dict[str, List[PipelineRunResult]]):
    """Sanity check across all 80+ profiles combined, independent of the per-type gate above."""
    all_runs = [r for runs in all_pipeline_results.values() for r in runs]
    assert len(all_runs) >= 4 * N_PROFILES_PER_TYPE

    correct = sum(1 for t, runs in all_pipeline_results.items() for r in runs if r.predicted_type == t)
    overall_accuracy = correct / len(all_runs)
    assert overall_accuracy > ACCURACY_TARGET


@pytest.mark.parametrize(
    "true_type,expect_significant",
    [
        ("ignorance", False),
        ("aspiration", True),
        ("self_protection", False),
        ("active_transition", True),
    ],
)
def test_granger_significance_matches_planted_coupling(
    all_pipeline_results: Dict[str, List[PipelineRunResult]], true_type: str, expect_significant: bool
):
    """
    Diagnostic check on the Granger stage in isolation (S79.1): types with
    planted directional coupling (aspiration, active_transition) should find
    significance in at least one direction more often than not; types
    without planted coupling (ignorance, self_protection) should not, at
    better than chance rates. This isolates Granger-stage recovery from the
    rest of the pipeline (HSSM fitting, divergence scoring) so a failure
    here points specifically at `within_regime_granger_test`/MS-VAR rather
    than at the type-classification chain as a whole.
    """
    runs = all_pipeline_results[true_type]
    ran_runs = [r for r in runs if r.granger_result.ran]
    assert ran_runs, f"{true_type}: power gate blocked Granger on every profile — check MIN_SESSIONS_PER_REGIME."

    sig_frac = sum(
        1 for r in ran_runs
        if r.granger_result.significant_m_causes_n or r.granger_result.significant_n_causes_m
    ) / len(ran_runs)

    if expect_significant:
        assert sig_frac > 0.5, f"{true_type}: only {sig_frac:.1%} of runs found planted Granger coupling."
    else:
        assert sig_frac < 0.5, f"{true_type}: {sig_frac:.1%} of runs falsely found Granger coupling."


@pytest.mark.parametrize(
    "true_type,expect_shared_latent_driver",
    [
        ("ignorance", False),
        ("aspiration", True),
        ("self_protection", False),
        ("active_transition", True),
    ],
)
def test_level2_shared_latent_driver_matches_s79_3_fix(
    all_pipeline_results: Dict[str, List[PipelineRunResult]],
    true_type: str,
    expect_shared_latent_driver: bool,
):
    """
    Validates the S79.3 fix end-to-end: Level 2's `shared_latent_driver_detected`
    check must track actual Granger significance, not just the power gate. Types
    with no planted coupling must NOT pass this check at better-than-chance
    rates even though their power gate passes (same session counts as the
    coupled types) — that's exactly the P0 bypass S79.3 closed.
    """
    runs = all_pipeline_results[true_type]

    def _shared_latent_driver_passed(evaluation: GateEvaluation) -> bool:
        return any(c.name == "shared_latent_driver_detected" and c.passed for c in evaluation.checks)

    pass_frac = sum(1 for r in runs if _shared_latent_driver_passed(r.level2_evaluation)) / len(runs)

    if expect_shared_latent_driver:
        assert pass_frac > 0.5
    else:
        assert pass_frac < 0.5


if __name__ == "__main__":
    generator = SyntheticProfileGenerator()
    results: Dict[str, List[PipelineRunResult]] = {t: [] for t in PROFILE_METHODS}
    for true_type, method_name in PROFILE_METHODS.items():
        gen_method = getattr(generator, method_name)
        for i in range(N_PROFILES_PER_TYPE):
            results[true_type].append(run_profile_through_pipeline(gen_method(seed=i)))

    for true_type, runs in results.items():
        correct = sum(1 for r in runs if r.predicted_type == true_type)
        acc = correct / len(runs)
        status = "PASS" if acc > ACCURACY_TARGET else "FAIL"
        print(f"  [{status}] {true_type}: {acc:.1%} (target >{ACCURACY_TARGET:.0%}, n={len(runs)})")