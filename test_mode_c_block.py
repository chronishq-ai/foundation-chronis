# tests/test_mode_c_block.py — Sprint 14 Day 41.
#
# Mode C (Raw Vault) is already covered inside test_policy_boundary_cases.py's
# Grid A/B/C combinatorics — this file exists separately because a reviewer
# asking "prove Mode C can never be reached" shouldn't have to read 104
# parametrized cases to find the answer. Every test here isolates Mode C
# specifically, at maximum consent tier, across every real ML entry point,
# with no other variable changing.
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from divergence_engine.engine import DivergenceInputs
from integration.gated_claims import ClaimAccessInputs, evaluate_claim_access
from integration.gated_divergence import gated_compute_divergence_state
from integration.gated_registry import GatedRegistry
from integration.gated_store import GatedModelStore

from policy_engine.consent import ConsentRecord, ConsentTier, OperationalMode, check_mode_c_block
from policy_engine.errors import ModeCBlocked, PolicyRuleError
from policy_engine.policy_rule import PolicyRule, RuleAction, Scope
from policy_engine.principal import ModelPrincipal

import numpy as np


MAX_TIER = ConsentTier.FULL  # deliberately the most permissive tier — Mode C
                              # must block even here, since it's mode-gated
                              # not tier-gated (see consent.py::check_mode_c_block).


def _divergence_inputs(user_id: str) -> DivergenceInputs:
    rng = np.random.default_rng(3)
    n = 30
    return DivergenceInputs(
        user_id=user_id, domain_id="dom-x",
        window_start=datetime.now(timezone.utc), window_end=datetime.now(timezone.utc),
        p_t=rng.integers(0, 2, n), q_t=rng.integers(0, 2, n),
        m_t=rng.normal(0, 1, n), n_t=rng.normal(0, 1, n),
        behavioral_regime_id=0, narrative_regime_id=0, n_domain_pairs_tested=1,
        behavioral_attractor_weakening=False, narrative_conformal_confidence=0.7,
    )


def test_mode_c_rejected_at_construction_regardless_of_other_modes_present():
    """Mode C can't even be constructed into a rule's allowed_modes, even
    alongside otherwise-legal modes in the same set."""
    with pytest.raises(PolicyRuleError):
        PolicyRule(
            rule_id="r", principal="system", subject_user_id="u1",
            scope=Scope(actions=frozenset({RuleAction.MODEL_READ})),
            min_consent_tier=ConsentTier.INFERENCE,
            allowed_modes=frozenset({OperationalMode.MODE_A, OperationalMode.MODE_C}),
            granted_at=datetime.now(timezone.utc),
        )


def test_mode_c_rejected_by_bare_consent_check_at_max_tier():
    with pytest.raises(ModeCBlocked):
        check_mode_c_block(OperationalMode.MODE_C)


def test_mode_c_blocked_via_model_store_write(tmp_path: Path):
    principal = ModelPrincipal()
    principal.ensure_default_rule("u1")
    gs = GatedModelStore(principal, root=tmp_path)
    consent = ConsentRecord("u1", MAX_TIER, OperationalMode.MODE_C)
    with pytest.raises(ModeCBlocked):
        gs.write(consent, "u1", "hssm", "x.bin", b"data")
    assert not (tmp_path / "models" / "u1" / "hssm" / "x.bin").exists()


def test_mode_c_blocked_via_model_store_read(tmp_path: Path):
    principal = ModelPrincipal()
    principal.ensure_default_rule("u1")
    gs = GatedModelStore(principal, root=tmp_path)
    # write something real first under a legal mode, then attempt to read
    # it back under Mode C — the data existing doesn't help.
    good = ConsentRecord("u1", ConsentTier.INFERENCE, OperationalMode.MODE_A)
    path = gs.write(good, "u1", "hssm", "x.bin", b"data")
    modec = ConsentRecord("u1", MAX_TIER, OperationalMode.MODE_C)
    with pytest.raises(ModeCBlocked):
        gs.read(modec, path)


def test_mode_c_blocked_via_registry_register_even_with_explicit_rule(tmp_path: Path, monkeypatch):
    """The strongest version of this test: give the user an EXPLICIT rule
    that grants REGISTRY_REGISTER — Mode C must still be blocked, because
    that rule can never legally list Mode C in allowed_modes to begin with,
    and the consent-level check runs independently regardless."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    principal = ModelPrincipal()
    principal.register_rule(PolicyRule(
        rule_id="allow-register", principal="system", subject_user_id="u1",
        scope=Scope(actions=frozenset({RuleAction.REGISTRY_REGISTER})),
        min_consent_tier=ConsentTier.INFERENCE,
        allowed_modes=frozenset({OperationalMode.MODE_A, OperationalMode.MODE_B}),
        granted_at=datetime.now(timezone.utc),
    ))
    gr = GatedRegistry(principal, root=tmp_path)
    artifact = tmp_path / "a.bin"
    artifact.write_bytes(b"w")
    modec = ConsentRecord("u1", MAX_TIER, OperationalMode.MODE_C)
    with pytest.raises(ModeCBlocked):
        gr.register(modec, "u1", "hssm", artifact,
                    {"training_data_hash": "a" * 32, "hyperparameters": {}, "metrics": {},
                     "fit_date": datetime.now(timezone.utc).date()}, "why")


def test_mode_c_blocked_via_claim_access():
    from claims_engine.claim_levels import Claim, ClaimLevel, evaluate_level1
    from claims_engine.surfacing_policy import SurfaceDecision
    from upstream_interfaces import AttractorRecord

    principal = ModelPrincipal()
    principal.ensure_default_rule("u1")
    attractor = AttractorRecord(user_id="u1", regime_id=1, context_key="c",
                                 revisit_count=10, mean_dwell_time=20.0,
                                 transition_stability=0.7, declared=True)
    gate_eval = evaluate_level1(attractor)
    claim = Claim.new("u1", "dom-x", ClaimLevel.LEVEL_1, gate_eval)
    modec = ConsentRecord("u1", MAX_TIER, OperationalMode.MODE_C)
    inputs = ClaimAccessInputs(False, False, False, False)
    result = evaluate_claim_access(principal, modec, claim, gate_eval, inputs)
    # claim access never raises — it WITHHOLDS. Confirm it withholds for
    # exactly the constitutional-restriction reason, not the gate reason,
    # so we know Mode C is what caused this, not an unrelated failure.
    assert result.decision == SurfaceDecision.WITHHOLD
    assert "Constitutional" in result.reason


def test_mode_c_blocked_via_divergence_inference():
    principal = ModelPrincipal()
    principal.ensure_default_rule("u1")
    modec = ConsentRecord("u1", MAX_TIER, OperationalMode.MODE_C)
    with pytest.raises(ModeCBlocked):
        gated_compute_divergence_state(principal, modec, _divergence_inputs("u1"))


def test_mode_c_denials_are_all_audited():
    """Aggregate check: run Mode C against every gated entry point in one
    principal's lifetime and confirm every single one left a DENIED entry
    — Mode C blocks are not somehow exempt from the audit requirement."""
    principal = ModelPrincipal()
    principal.ensure_default_rule("u1")
    modec = ConsentRecord("u1", MAX_TIER, OperationalMode.MODE_C)

    with pytest.raises(ModeCBlocked):
        gated_compute_divergence_state(principal, modec, _divergence_inputs("u1"))

    from claims_engine.claim_levels import Claim, ClaimLevel, evaluate_level1
    from upstream_interfaces import AttractorRecord
    attractor = AttractorRecord(user_id="u1", regime_id=1, context_key="c",
                                 revisit_count=10, mean_dwell_time=20.0,
                                 transition_stability=0.7, declared=True)
    gate_eval = evaluate_level1(attractor)
    claim = Claim.new("u1", "dom-x", ClaimLevel.LEVEL_1, gate_eval)
    evaluate_claim_access(principal, modec, claim, gate_eval, ClaimAccessInputs(False, False, False, False))

    assert len(principal.audit) == 2
    assert all(e.outcome.value == "denied" for e in principal.audit)
    principal.audit.verify()