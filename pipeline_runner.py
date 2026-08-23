# e2e/pipeline_runner.py — Sprint 14 Day 42.
#
# Directive: "Run the complete pipeline on TILES-2018 end-to-end: ingest ->
# decrypt-in-RAM -> transcribe/extract -> align -> HSSM fit -> attractor ->
# domain -> divergence -> claims -> Mirror. Verify the under-20-minutes-
# per-day-of-data processing-time target."
#
# HONEST SCOPE NOTE — read this before trusting any stage's output:
# Only Sprint 3-9 and Sprint 13's code was uploaded to this workspace.
# Sprint 1-2 (ingest/decrypt/transcribe/align) and Sprint 12 (Mirror) have
# NO code in the uploaded zips at all — those stages are STUBBED below,
# clearly marked, not silently faked. Domain emergence's full HDBSCAN/
# BERTopic pipeline (Sprint 6) exists in the uploads but is NOT wired here
# either — reusing it correctly would need real transcript embeddings this
# demo doesn't have; a Domain object is constructed directly from the
# behavioral/narrative regime ids instead, which is the same input shape
# claims_engine already expects (see gated_claims.py's own test fixtures).
#
# What IS real, running actual uploaded code end-to-end:
#   - HSSM fit:        backbone.hssm.fitting.fit_hssm_model (Sprint 3)
#   - regime decode:   model._forward_backward (Sprint 3's own internal
#                       method — no public decode() was exposed; flagged
#                       as a gap for BACKBONE to fix, not papered over here)
#   - attractor:        backbone.attractors.detector (Sprint 4)
#   - narrative regime: nssm_pipeline.cross_system_wiring's own planted
#                       validation-set generator (Sprint 7) — ground-truth
#                       labels, standing in for a full NSSM fit
#   - divergence:       integration.gated_divergence (Day 40, real Fisher +
#                       Granger, POLICY-ENGINE-GATED)
#   - claims:           integration.gated_claims (Day 40, real Sprint 9
#                       gates, POLICY-ENGINE-GATED)
#
# Per Global Standard rule 0 ("a green checkbox that isn't actually true
# is worse than a red one"), do not represent a run of this file as
# validating Sprint 1, 2, 6's clustering/topic code, or Sprint 12's Mirror
# — it does not exercise them.
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np

from backbone.attractors.detector import compute_attractor_stats, is_attractor
from backbone.hssm.fitting import fit_hssm_model
from claims_engine.claim_levels import Claim, ClaimLevel, evaluate_level1
from claims_engine.surfacing_policy import SurfaceDecision
from divergence_engine.engine import DivergenceInputs
from nssm_pipeline.cross_system_wiring import build_synthetic_validation_set
from upstream_interfaces import AttractorRecord, Domain

from e2e.tiles_loader import SurrogateDayOfData, load_surrogate_day
from e2e.timing import Timer, TimingReport

from integration.gated_claims import ClaimAccessInputs, evaluate_claim_access
from integration.gated_divergence import gated_compute_divergence_state
from observer_effect.observer import Observer
from policy_engine.consent import ConsentRecord
from policy_engine.principal import ModelPrincipal

# Placeholder attractor thresholds. Sprint 4 Day 11's own calibration
# harness (grid-searching N/T per user against target precision/recall)
# is NOT run here — these are illustrative constants, explicitly NOT the
# person-calibrated values the Non-Negotiables section requires in
# production. Flagged, not hidden.
_ILLUSTRATIVE_ATTRACTOR_N = 3
_ILLUSTRATIVE_ATTRACTOR_T = 0.5


@dataclass
class PipelineRunResult:
    user_id: str
    timing: TimingReport
    hssm_k_selected: int
    attractor_declared: bool
    claim_level_reached: Optional[ClaimLevel]
    mirror_output: str  # possibly empty — empty is a valid, correct output
    observer: Observer | None = None


def _stub_mirror(claim: Optional[Claim], decision: SurfaceDecision) -> str:
    """
    NOT Sprint 12's real Mirror. Sprint 12's actual insight generator
    (100-200 words, citation-chain grounded, specificity-linted, tone-
    calibrated) has no code in this workspace. This function exists only
    so the pipeline has *something* in the Mirror-output slot to report,
    and to demonstrate the empty-output path is honored end-to-end.
    """
    if decision != SurfaceDecision.SURFACE or claim is None:
        return ""  # silence is a valid, correct output (Non-Negotiables)
    return (
        f"[STUB MIRROR — not Sprint 12] A {claim.level.name.lower().replace('_', ' ')} "
        f"pattern was identified in domain {claim.domain_id!r}."
    )


def run_pipeline_for_user(
    principal: ModelPrincipal,
    consent: ConsentRecord,
    user_id: str,
    *,
    seed: Optional[int] = None,
    observer: Optional[Observer] = None,
    surfaced_on: Optional[date] = None,
) -> PipelineRunResult:
    timer = Timer()
    observer = observer or Observer()

    # --- Stage 1 (STUB): ingest -> decrypt-in-RAM -> transcribe -> align ---
    with timer.stage("ingest_decrypt_align", is_stub=True):
        day_data: SurrogateDayOfData = load_surrogate_day(user_id, seed=seed)

    # --- Stage 2 (REAL): HSSM fit (Sprint 3) ---
    with timer.stage("hssm_fit", is_stub=False):
        model, report = fit_hssm_model(
            day_data.X, candidate_ks=(2, 3), n_initializations=10, random_seed=seed,
        )
        regime_posterior, _, _ = model._forward_backward(day_data.X)
        p_t = regime_posterior.argmax(axis=1)

    # --- Stage 3 (REAL): attractor detection (Sprint 4) ---
    with timer.stage("attractor_detection", is_stub=False):
        target_regime = int(np.bincount(p_t).argmax())  # most-visited regime
        stats = compute_attractor_stats(day_data.X, p_t, target_regime=target_regime)
        declared = is_attractor(stats, N=_ILLUSTRATIVE_ATTRACTOR_N, T=_ILLUSTRATIVE_ATTRACTOR_T)
        attractor = AttractorRecord(
            user_id=user_id, regime_id=target_regime, context_key="e2e-demo",
            revisit_count=stats["revisit_count"], mean_dwell_time=stats["mean_dwell_time"],
            transition_stability=stats["transition_stability"], declared=declared,
        )

    # --- Stage 4 (REAL narrative ground-truth, not a full NSSM fit) ---
    with timer.stage("narrative_regime_standin", is_stub=False):
        _sessions, planted_narrative_labels = build_synthetic_validation_set(sessions_per_pattern=6)
        q_t = planted_narrative_labels  # ground truth used AS the narrative regime signal here

    # --- Stage 5 (construction, not an algorithm stub): Domain object ---
    with timer.stage("domain_construction", is_stub=True):
        domain = Domain(
            domain_id="dom-e2e-demo", user_id=user_id, label="e2e-demo",
            behavioral_regime_ids=[target_regime], narrative_regime_ids=[0],
            confidence=0.5, active=True, high_ignorance_prior=False,
            aspirational_or_hypothetical=False,
        )

    # --- Stage 6 (REAL, policy-engine-gated): divergence (Sprint 8 + Day 40) ---
    with timer.stage("divergence_gated", is_stub=False):
        n = min(len(p_t), len(q_t))
        div_inputs = DivergenceInputs(
            user_id=user_id, domain_id=domain.domain_id,
            window_start=datetime.now(timezone.utc), window_end=datetime.now(timezone.utc),
            p_t=p_t[:n], q_t=q_t[:n],
            m_t=day_data.X[:n, 0], n_t=np.zeros(n),
            behavioral_regime_id=target_regime, narrative_regime_id=0,
            n_domain_pairs_tested=1,
            behavioral_attractor_weakening=not declared,
            narrative_conformal_confidence=0.7,
        )
        divergence_state = gated_compute_divergence_state(principal, consent, div_inputs)

    # --- Stage 7 (REAL, policy-engine-gated): claims (Sprint 9 + Day 40) ---
    with timer.stage("claims_gated", is_stub=False):
        level1_eval = evaluate_level1(attractor)
        claim = Claim.new(
            user_id, domain.domain_id, ClaimLevel.LEVEL_1, level1_eval,
        )
        claim_inputs = ClaimAccessInputs(
            acute_trauma_markers_present=False, has_therapeutic_context=False,
            self_protection_gate_failed=False, contradiction_without_new_evidence=False,
        )
        surfacing_result = evaluate_claim_access(principal, consent, claim, level1_eval, claim_inputs)
        if observer is not None:
            observer.note_shown_claim(
                claim,
                surfacing_result,
                surfaced_on or date.today(),
                consent=consent,
            )

    # --- Stage 8 (STUB): Mirror (Sprint 12 — no code present) ---
    with timer.stage("mirror_stub", is_stub=True):
        mirror_output = _stub_mirror(claim, surfacing_result.decision)

    return PipelineRunResult(
        user_id=user_id,
        timing=timer.report,
        hssm_k_selected=report["k_selected"],
        attractor_declared=declared,
        claim_level_reached=level1_eval.level if level1_eval.admissible else None,
        mirror_output=mirror_output,
        observer=observer,
    )