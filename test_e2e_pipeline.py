# tests/test_e2e_pipeline.py — Sprint 14 Day 42.
#
# DoD (from directive): "A full pipeline run on a day of TILES-2018
# surrogate data completes within the 20-minute target and produces a
# correctly gated (possibly empty) Mirror output."
#
# Scope caveat carried over from e2e/pipeline_runner.py's own docstring:
# this exercises real Sprint 3/4/7/8/9/13 code plus Day 40's policy
# engine, against SYNTHETIC surrogate data, NOT real TILES-2018 (no such
# data or Sprint 1/2 ingest code was ever uploaded to this workspace).
# These tests prove the pipeline's plumbing and timing/gating discipline
# hold — they do not certify the directive's literal TILES-2018 claim.
# See e2e/tiles_loader.py's docstring for the same caveat, stated once
# there and not softened here.
from __future__ import annotations

import pytest

from e2e.pipeline_runner import run_pipeline_for_user
from e2e.timing import TWENTY_MINUTES_SECONDS

from policy_engine.consent import ConsentRecord, ConsentTier, OperationalMode
from policy_engine.errors import ConsentTierError, ModeCBlocked
from policy_engine.principal import ModelPrincipal


@pytest.fixture
def principal() -> ModelPrincipal:
    p = ModelPrincipal()
    p.ensure_default_rule("u1")
    return p


def test_full_run_completes(principal):
    consent = ConsentRecord("u1", ConsentTier.INFERENCE, OperationalMode.MODE_A)
    result = run_pipeline_for_user(principal, consent, "u1", seed=1)
    assert result.user_id == "u1"
    assert result.hssm_k_selected in (2, 3)


def test_full_run_within_twenty_minute_target(principal):
    consent = ConsentRecord("u1", ConsentTier.INFERENCE, OperationalMode.MODE_A)
    result = run_pipeline_for_user(principal, consent, "u1", seed=1)
    assert result.timing.within_target
    assert result.timing.total_seconds < TWENTY_MINUTES_SECONDS


def test_mirror_output_gated_correctly(principal):
    """Mirror output must be empty whenever the claim doesn't surface, and
    non-empty only when it does — never the reverse. This is the concrete
    meaning of 'correctly gated (possibly empty)' in the DoD."""
    consent = ConsentRecord("u1", ConsentTier.INFERENCE, OperationalMode.MODE_A)
    result = run_pipeline_for_user(principal, consent, "u1", seed=1)
    if result.claim_level_reached is None:
        assert result.mirror_output == ""
    else:
        assert result.mirror_output != ""


@pytest.mark.parametrize("seed", range(5))
def test_mirror_gating_consistent_across_seeds(principal, seed):
    """Repeat the same gating-consistency check across several seeds —
    the directive's own emphasis is that silence must be honored whenever
    it's the correct answer, not just in one cherry-picked run."""
    consent = ConsentRecord("u1", ConsentTier.INFERENCE, OperationalMode.MODE_A)
    result = run_pipeline_for_user(principal, consent, "u1", seed=seed)
    if result.claim_level_reached is None:
        assert result.mirror_output == ""
    else:
        assert result.claim_level_reached is not None
        assert result.mirror_output != ""


def test_low_consent_tier_blocks_divergence_stage_before_mirror(principal):
    """A user who never cleared consent must never reach a Mirror output
    at all — the pipeline must raise at the divergence stage, not silently
    degrade to an empty Mirror as if that were a normal gated outcome."""
    low_consent = ConsentRecord("u1", ConsentTier.STORAGE_ONLY, OperationalMode.MODE_A)
    with pytest.raises(ConsentTierError):
        run_pipeline_for_user(principal, low_consent, "u1", seed=1)


def test_KNOWN_GAP_hssm_and_attractor_stages_run_before_any_gate_check(principal):
    """
    Documents a real gap, not a passing assertion to hide it: Stages 2-3
    (HSSM fit, attractor detection) run UNCONDITIONALLY, before the first
    policy_engine.check() call in Stage 6 (divergence). A denied user's
    aligned feature matrix is still fit by the real HSSM and scored for
    attractors before the pipeline ever consults consent/mode.
    Whether this crosses the directive's "every ML data read/write" line
    is a real design question for whoever owns Sprint 3/4's integration —
    HSSM fitting reads a user's abstracted feature matrix and produces a
    model artifact, which arguably needs its own gated_hssm.py wrapper
    parallel to gated_divergence.py, not yet built in Sprint 14. This test
    exists so the gap shows up in CI output, not just in a comment nobody
    reads.
    """
    from e2e.pipeline_runner import _stub_mirror  # noqa: F401 — presence check only
    low_consent = ConsentRecord("u1", ConsentTier.NONE, OperationalMode.MODE_A)
    entries_before = len(principal.audit)
    with pytest.raises(ConsentTierError):
        run_pipeline_for_user(principal, low_consent, "u1", seed=1)
    # the denial IS audited (at the divergence stage) — but note that by
    # the time this entry exists, HSSM fit + attractor detection already
    # ran to completion against this "denied" user's data, unaudited.
    assert len(principal.audit) == entries_before + 1


def test_mode_c_blocks_pipeline_before_mirror(principal):
    modec_consent = ConsentRecord("u1", ConsentTier.FULL, OperationalMode.MODE_C)
    with pytest.raises(ModeCBlocked):
        run_pipeline_for_user(principal, modec_consent, "u1", seed=1)


def test_denied_run_still_produces_audit_trail(principal):
    """Even a run that never reaches Mirror because of a policy denial
    must leave an audit trail — the denial itself is the record that the
    pipeline correctly refused to proceed."""
    low_consent = ConsentRecord("u1", ConsentTier.NONE, OperationalMode.MODE_A)
    with pytest.raises(ConsentTierError):
        run_pipeline_for_user(principal, low_consent, "u1", seed=1)
    assert len(principal.audit) >= 1
    principal.audit.verify()


def test_timing_report_labels_stub_stages_honestly():
    """Structural check on the honesty requirement itself: the timing
    report must report at least one stub stage (ingest/align, domain
    construction, Mirror) — a report claiming zero stubs would mean
    someone quietly deleted the honest labeling this file depends on."""
    principal = ModelPrincipal()
    principal.ensure_default_rule("u1")
    consent = ConsentRecord("u1", ConsentTier.INFERENCE, OperationalMode.MODE_A)
    result = run_pipeline_for_user(principal, consent, "u1", seed=1)
    assert result.timing.stub_stage_count >= 3
    assert result.timing.real_stage_count >= 4


def test_repeated_runs_produce_independent_audit_entries():
    """Two full runs for two different users must not cross-contaminate
    each other's audit trail or gating outcome."""
    principal = ModelPrincipal()
    principal.ensure_default_rule("u1")
    principal.ensure_default_rule("u2")
    c1 = ConsentRecord("u1", ConsentTier.INFERENCE, OperationalMode.MODE_A)
    c2 = ConsentRecord("u2", ConsentTier.INFERENCE, OperationalMode.MODE_A)
    r1 = run_pipeline_for_user(principal, c1, "u1", seed=1)
    r2 = run_pipeline_for_user(principal, c2, "u2", seed=2)
    assert r1.user_id == "u1"
    assert r2.user_id == "u2"
    principal.audit.verify()
    u1_entries = principal.audit.entries_for("u1")
    u2_entries = principal.audit.entries_for("u2")
    assert len(u1_entries) >= 1 and len(u2_entries) >= 1
    assert not set(id(e) for e in u1_entries) & set(id(e) for e in u2_entries)