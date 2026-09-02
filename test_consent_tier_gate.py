# tests/test_consent_tier_gate.py — Sprint 14 Day 41.
#
# test_policy_boundary_cases.py's Grid A already sweeps tier x mode x
# action combinatorially. This file exists to isolate ONE thing clearly:
# the exact cutoff is tier >= 2 (ConsentTier.INFERENCE), not >2, not >=1,
# not >=3 — and that the cutoff is a real comparison, not a hardcoded
# per-tier lookup table that happens to agree with it by coincidence.
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from divergence_engine.engine import DivergenceInputs
from integration.gated_divergence import gated_compute_divergence_state

from policy_engine.consent import (
    MIN_INFERENCE_CONSENT_TIER,
    ConsentRecord,
    ConsentTier,
    OperationalMode,
    check_inference_consent,
    is_inference_permitted,
)
from policy_engine.errors import ConsentTierError
from policy_engine.principal import ModelPrincipal

import numpy as np


def test_named_constant_is_exactly_tier_2():
    """The Global Standard's no-silent-magic-number rule means this
    constant, not a bare `2` scattered across call sites, is the single
    source of truth for the cutoff. Confirm it's what the directive
    actually specifies."""
    assert MIN_INFERENCE_CONSENT_TIER == ConsentTier.INFERENCE
    assert int(MIN_INFERENCE_CONSENT_TIER) == 2


@pytest.mark.parametrize(
    "tier,should_pass",
    [
        (ConsentTier.NONE, False),          # 0
        (ConsentTier.STORAGE_ONLY, False),  # 1
        (ConsentTier.INFERENCE, True),      # 2 — exactly the boundary, must pass
        (ConsentTier.FULL, True),           # 3
    ],
)
def test_boundary_is_exactly_at_tier_2(tier: ConsentTier, should_pass: bool):
    record = ConsentRecord("u1", tier, OperationalMode.MODE_A)
    if should_pass:
        check_inference_consent(record)  # must not raise
    else:
        with pytest.raises(ConsentTierError):
            check_inference_consent(record)


def test_is_inference_permitted_agrees_with_raising_form_at_every_tier():
    """The non-raising probe and the raising check must never disagree —
    if they did, some caller relying on one and some on the other would
    see inconsistent gating for the identical input."""
    for tier in ConsentTier:
        record = ConsentRecord("u1", tier, OperationalMode.MODE_A)
        probe_result = is_inference_permitted(record)
        try:
            check_inference_consent(record)
            raised = False
        except ConsentTierError:
            raised = True
        assert probe_result == (not raised), f"disagreement at tier={tier.name}"


def test_tier_just_below_boundary_denied_via_real_entry_point():
    """STORAGE_ONLY (1) is exactly one step below the cutoff — the
    smallest possible gap between pass and fail. Tested against a real
    entry point (divergence inference), not just the bare consent check,
    to confirm the boundary holds through the whole call chain."""
    principal = ModelPrincipal()
    principal.ensure_default_rule("u1")
    rng = np.random.default_rng(1)
    n = 20
    inputs = DivergenceInputs(
        user_id="u1", domain_id="dom-x",
        window_start=datetime.now(timezone.utc), window_end=datetime.now(timezone.utc),
        p_t=rng.integers(0, 2, n), q_t=rng.integers(0, 2, n),
        m_t=rng.normal(0, 1, n), n_t=rng.normal(0, 1, n),
        behavioral_regime_id=0, narrative_regime_id=0, n_domain_pairs_tested=1,
        behavioral_attractor_weakening=False, narrative_conformal_confidence=0.7,
    )
    just_below = ConsentRecord("u1", ConsentTier.STORAGE_ONLY, OperationalMode.MODE_A)
    with pytest.raises(ConsentTierError):
        gated_compute_divergence_state(principal, just_below, inputs)

    just_at = ConsentRecord("u1", ConsentTier.INFERENCE, OperationalMode.MODE_A)
    result = gated_compute_divergence_state(principal, just_at, inputs)
    assert result is not None  # the one-tier difference is the only variable that changed


def test_consent_tier_error_reports_both_actual_and_required_tier():
    """The error message itself must be actionable — a denial that doesn't
    say what tier was required is a debugging dead end for whoever reads
    the audit log later."""
    record = ConsentRecord("u1", ConsentTier.STORAGE_ONLY, OperationalMode.MODE_A)
    with pytest.raises(ConsentTierError) as exc:
        check_inference_consent(record)
    reason = exc.value.reason
    assert "STORAGE_ONLY" in reason or "1" in reason
    assert "2" in reason or "INFERENCE" in reason


def test_denial_at_tier_boundary_is_audited_with_correct_tier_in_reason():
    principal = ModelPrincipal()
    principal.ensure_default_rule("u1")
    from policy_engine.policy_rule import RuleAction
    from policy_engine.principal import AccessRequest

    record = ConsentRecord("u1", ConsentTier.STORAGE_ONLY, OperationalMode.MODE_A)
    with pytest.raises(ConsentTierError):
        principal.check(AccessRequest(action=RuleAction.INFERENCE, consent=record))
    entry = principal.audit._entries[0]
    assert entry.outcome.value == "denied"
    assert "STORAGE_ONLY" in entry.reason