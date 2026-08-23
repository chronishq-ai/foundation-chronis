from datetime import date, timedelta
from pathlib import Path

from e2e.pipeline_runner import run_pipeline_for_user
from observer_effect.index import INFLUENCE_FLAG, INFLUENCE_WINDOW_DAYS, SurfacedClaim, SurfacingIndex
from observer_effect.observer import Observer, cold_start_silent
from observer_effect.profiles import AT_LAG, log_accuracy_mlflow, plant_profiles, type_accuracy
from observer_effect.regression import cold_start_180, logging_was_on, silence_holds
from observer_effect.safeguard import Change, aspiration_evidence_weight, copy_mentions_internal, product_copy
from policy_engine.consent import ConsentRecord, ConsentTier, OperationalMode
from policy_engine.principal import ModelPrincipal


def test_attractor_change_inside_window_excluded_outside_not():
    index = SurfacingIndex()
    shown = date(2026, 6, 1)
    index.append(SurfacedClaim("c1", "alice", "career", 2, "Aspiration", shown))

    inside = Change("alice", "career", "behavior", shown + timedelta(days=10))
    boundary = Change("alice", "career", "behavior", shown + timedelta(days=INFLUENCE_WINDOW_DAYS))
    outside = Change("alice", "career", "behavior", shown + timedelta(days=31))
    other_domain = Change("alice", "health", "behavior", shown + timedelta(days=5))

    o = Observer(index)
    inside = o.note_change(inside)
    boundary = o.note_change(boundary)
    outside = o.note_change(outside)
    other_domain = o.note_change(other_domain)

    assert inside.potentially_claim_influenced is True
    assert boundary.potentially_claim_influenced is True
    assert outside.potentially_claim_influenced is False
    assert other_domain.potentially_claim_influenced is False
    assert aspiration_evidence_weight(inside, index) == 0.0
    assert aspiration_evidence_weight(outside, index) == 1.0
    assert o.aspiration_weight(inside) == 0.0
    assert o.aspiration_weight(outside) == 1.0


def test_narrative_shift_same_window_rule():
    index = SurfacingIndex()
    shown = date(2026, 6, 1)
    index.append(SurfacedClaim("c1", "alice", "career", 1, "Aspiration", shown))
    o = Observer(index)
    inside = o.note_change(Change("alice", "career", "narrative", shown + timedelta(days=29)))
    outside = o.note_change(Change("alice", "career", "narrative", shown + timedelta(days=31)))
    assert inside.potentially_claim_influenced is True
    assert outside.potentially_claim_influenced is False
    assert aspiration_evidence_weight(inside, index) == 0.0
    assert aspiration_evidence_weight(outside, index) == 1.0


def test_aspiration_exclusion_is_read_time_not_just_a_flag():
    """Even if the caller never set the flag, the index still zeros weight."""
    index = SurfacingIndex()
    shown = date(2026, 6, 1)
    index.append(SurfacedClaim("c1", "alice", "career", 3, "Aspiration", shown))
    naked = Change("alice", "career", "behavior", shown + timedelta(days=2))
    assert naked.potentially_claim_influenced is False
    assert aspiration_evidence_weight(naked, index) == 0.0


def test_level_0_claim_is_not_indexed():
    index = SurfacingIndex()
    rec = index.append(SurfacedClaim("c0", "alice", "career", 0, "Ignorance", date(2026, 6, 1)))
    assert rec is None
    assert len(index) == 0


def test_flag_never_in_product_copy():
    ch = Change("alice", "career", "behavior", date.today(), potentially_claim_influenced=True)
    text = product_copy(ch)
    assert not copy_mentions_internal(text)
    assert INFLUENCE_FLAG not in text.lower()


def test_all_four_types_20plus_each_over_75(tmp_path: Path):
    profiles = plant_profiles(n_per_type=20, seed=7)
    counts = {t: sum(1 for p in profiles if p.label == t) for t in ("Ignorance", "Aspiration", "Self-Protection", "ActiveTransition")}
    assert all(n >= 20 for n in counts.values()), counts
    acc = type_accuracy(profiles)
    for t, v in acc.items():
        assert v > 0.75, (t, v, acc)
    uri = (tmp_path / "mlruns").resolve().as_uri()
    run_id = log_accuracy_mlflow(acc, profiles, tracking_uri=uri)
    assert run_id


def test_at_lag_direction_recovered():
    profiles = [p for p in plant_profiles(n_per_type=20, seed=7) if p.label == "ActiveTransition"]
    recovered = [p.recovered_lag for p in profiles if p.recovered_lag is not None]
    assert recovered
    # planted lag is positive (narrative lags behavior)
    n_correct_sign = sum(1 for lag in recovered if lag > 0)
    assert n_correct_sign / len(recovered) > 0.75
    close = sum(1 for p in profiles if p.recovered_lag is not None and abs(p.recovered_lag - AT_LAG) <= 2)
    assert close / len(profiles) > 0.5


def test_at_score_dominates_on_at_profiles():
    ats = [p for p in plant_profiles(n_per_type=20, seed=7) if p.label == "ActiveTransition" and not p.ambiguous]
    assert ats
    wins = sum(1 for p in ats if p.predicted == "ActiveTransition")
    assert wins / len(ats) > 0.75


def test_cold_start_180_and_mirror_silence_with_logging_on():
    index = SurfacingIndex()
    snaps = cold_start_180(index, sessions_per_day=0.6)
    assert len(snaps) == 180
    assert silence_holds(snaps)
    assert logging_was_on(index, snaps)
    o = Observer(index)
    assert cold_start_silent(0, observer=o) is True
    assert cold_start_silent(1, observer=o) is True
    assert cold_start_silent(2, observer=o) is False


def test_mp13_mitigated_not_closed():
    p = Path(__file__).resolve().parents[1] / "mp_registry.json"
    text = p.read_text(encoding="utf-8")
    assert "MP-13" in text
    assert "permanently open" in text
    readme = Path(__file__).resolve().parents[1] / "observer_effect" / "README.md"
    body = readme.read_text(encoding="utf-8").lower()
    assert "mitigat" in body
    assert "does not solve" in body


def test_pipeline_logs_surfaced_claims_when_mirror_is_shown():
    principal = ModelPrincipal()
    principal.ensure_default_rule("u1")
    o = Observer()
    consent = ConsentRecord("u1", ConsentTier.INFERENCE, OperationalMode.MODE_A)
    result = run_pipeline_for_user(
        principal, consent, "u1", seed=1, observer=o, surfaced_on=date(2026, 6, 1)
    )
    if result.mirror_output:
        assert len(o.surfaced) >= 1
        assert o.surfaced[0].user_id == "u1"
        later = o.note_change(Change("u1", o.surfaced[0].domain, "behavior", date(2026, 6, 10)))
        assert later.potentially_claim_influenced is True
    else:
        assert result.mirror_output == ""
