from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import shutil

import pytest

from chronis_ml.ops import AB, FitReason, HssmQueue, Registry, check_licenses, check_logs, gpu_spec, pins, write_sbom
from chronis_ml.store import IsolatedModelStore, IsolationError, assert_src_isolated, check_path
from chronis_ml.train import PersonalLM


def test_bob_cant_read_alice(tmp_path):
    s = IsolatedModelStore(tmp_path)
    p = s.write("alice", "personal_lm", "adapter.bin", b"ALICE")
    assert s.read("alice", p) == b"ALICE"
    with pytest.raises(IsolationError):
        s.read("bob", p)


def test_outside_models_blocked(tmp_path):
    s = IsolatedModelStore(tmp_path)
    f = tmp_path / "secret.bin"
    f.write_bytes(b"x")
    with pytest.raises(IsolationError):
        check_path("alice", f, tmp_path)


def test_src_is_clean():
    assert_src_isolated(Path(__file__).resolve().parents[1] / "src")


def test_leaky_file_fails_scan(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy(Path(__file__).parent / "leaky.py", src / "leaky.py")
    with pytest.raises(IsolationError):
        assert_src_isolated(src)


def test_finetune_uses_shared_base(tmp_path):
    s = IsolatedModelStore(tmp_path)
    r = PersonalLM(s).fine_tune("alice", ["lab notes"])
    assert r.base_checkpoint_id == "chronis-base-v1"
    assert s.load_base("chronis-base-v1") != s.read("alice", r.adapter_path)


def test_no_promote(tmp_path):
    p = PersonalLM(IsolatedModelStore(tmp_path))
    p.fine_tune("alice", ["hi"])
    with pytest.raises(IsolationError):
        p.promote_to_global("alice")


def test_registry_needs_all_fields():
    with pytest.raises(Exception):
        check_logs({"training_data_hash": "abc"})
    check_logs(
        {
            "training_data_hash": "a" * 32,
            "hyperparameters": {"lr": 0.05},
            "metrics": {"loss": 1},
            "fit_date": date.today().isoformat(),
        }
    )


def test_mlflow_register(tmp_path):
    pytest.importorskip("mlflow")
    a = tmp_path / "a.bin"
    a.write_bytes(b"x")
    rid = Registry((tmp_path / "mlruns").resolve().as_uri(), tmp_path).register(
        "alice",
        "personal_lm",
        a,
        {
            "training_data_hash": "b" * 32,
            "hyperparameters": {"steps": 50},
            "metrics": {"adapter_l2": 1.2},
            "fit_date": date.today().isoformat(),
        },
        why="comparing nightly fits",
    )
    assert rid


def test_queue():
    now = datetime(2026, 8, 19, 12, tzinfo=timezone.utc)
    q = HssmQueue(now=lambda: now)
    cold = q.add("bob", FitReason.COLD, 30)
    assert q.pop() is None
    q.now = lambda: now + timedelta(days=1)
    assert q.pop().uid == cold.uid
    q2 = HssmQueue(now=lambda: now)
    q2.add("slow", FitReason.NORMAL, 40)
    q2.add("fast", FitReason.PHASE, 40)
    assert q2.pop().uid == "fast"


def test_gpu_and_ab():
    assert gpu_spec("alice")["instance_type"] == "g4dn.xlarge"
    ab = AB("v2")
    with pytest.raises(IsolationError):
        ab.arm("alice")
    for i in range(40):
        u = f"p{i:02d}"
        ab.consent(u)
        ab.score(u, 0.9 if ab.arm(u) == "treatment" else 0.1)
    assert ab.maybe_ship()["promote"] is True


def test_licenses():
    pins()
    check_licenses()
    write_sbom()
