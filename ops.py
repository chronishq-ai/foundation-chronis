# mlflow gate, hssm queue, spot box, a/b, pip-audit
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from heapq import heappop, heappush
from pathlib import Path
from typing import Any
from scipy import stats

from chronis_ml.store import IsolationError, ok_user, root_dir

NEED = ("training_data_hash", "hyperparameters", "metrics", "fit_date")
OK_LICENSES = {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "PSF-2.0", "MPL-2.0"}


class RegistryGateError(ValueError):
    pass


def check_logs(p: dict[str, Any]) -> None:
    miss = [k for k in NEED if not p.get(k)]
    if miss:
        raise RegistryGateError(f"missing {miss}")
    if len(str(p["training_data_hash"])) < 16:
        raise RegistryGateError("hash looks too short")
    if not isinstance(p["hyperparameters"], dict) or not isinstance(p["metrics"], dict):
        raise RegistryGateError("hyperparams/metrics should be dicts")
    fit = p["fit_date"]
    p["fit_date"] = fit.isoformat() if isinstance(fit, (date, datetime)) else str(fit)


class Registry:
    def __init__(self, tracking_uri: str | None = None, root: Path | None = None) -> None:
        self.root = Path(root) if root else Path.cwd()
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", (self.root / "mlruns").resolve().as_uri()
        )

    def register(self, uid: str, kind: str, artifact: Path, payload: dict, why: str) -> str:
        if kind not in {"hssm", "personal_lm"}:
            raise IsolationError("bad kind")
        check_logs(payload)
        if not why.strip():
            raise RegistryGateError("say why these settings exist")
        import mlflow

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(f"chronis.user.{ok_user(uid)}")
        with mlflow.start_run(run_name=f"{kind}-{payload['fit_date']}") as run:
            mlflow.set_tags({"user_id": uid, "kind": kind, "rationale": why})
            mlflow.log_param("training_data_hash", payload["training_data_hash"])
            mlflow.log_param("fit_date", payload["fit_date"])
            for k, v in payload["hyperparameters"].items():
                mlflow.log_param(k, v)
            for k, v in payload["metrics"].items():
                try:
                    mlflow.log_metric(k, float(v))
                except (TypeError, ValueError):
                    mlflow.set_tag(k, str(v))
            mlflow.log_artifact(str(artifact))
            return run.info.run_id


class FitReason:
    PHASE = 0
    COLD = 1
    NORMAL = 2


@dataclass(order=True)
class Job:
    key: tuple
    uid: str = field(compare=False)
    reason: int = field(compare=False)
    not_before: datetime = field(compare=False)


class HssmQueue:
    def __init__(self, now=None) -> None:
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.h: list[Job] = []

    def add(self, uid: str, reason: int, sessions: int = 0) -> Job:
        ok_user(uid)
        t = self.now()
        if reason == FitReason.COLD:
            if sessions < 30:
                raise ValueError("cold start is 30 sessions")
            ready = t + timedelta(days=1)
        else:
            ready = t
        job = Job((reason, ready.timestamp()), uid, reason, ready)
        heappush(self.h, job)
        return job

    def pop(self) -> Job | None:
        t = self.now()
        wait = []
        got = None
        while self.h:
            j = heappop(self.h)
            if j.not_before <= t:
                got = j
                break
            wait.append(j)
        for j in wait:
            heappush(self.h, j)
        return got


def gpu_spec(uid: str) -> dict:
    ok_user(uid)
    return {
        "user_id": uid,
        "instance_type": "g4dn.xlarge",
        "market": "spot",
        "model_uri": f"models/{uid}/personal_lm",
    }


class AB:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ok: set[str] = set()
        self.scores = {"control": [], "treatment": []}

    def consent(self, uid: str) -> None:
        self.ok.add(ok_user(uid))

    def arm(self, uid: str) -> str:
        uid = ok_user(uid)
        if uid not in self.ok:
            raise IsolationError("no consent")
        b = hashlib.sha256(f"{self.name}:{uid}".encode()).digest()[0]
        return "treatment" if b % 2 else "control"

    def score(self, uid: str, n: float) -> None:
        self.scores[self.arm(uid)].append(float(n))

    def maybe_ship(self, alpha: float = 0.05) -> dict:
        c, t = self.scores["control"], self.scores["treatment"]
        if len(c) < 8 or len(t) < 8:
            return {"promote": False, "why": "not enough people"}
        _, p = stats.ttest_ind(t, c, equal_var=False, alternative="greater")
        promote = p < alpha and (sum(t) / len(t)) > (sum(c) / len(c))
        return {"promote": bool(promote), "p": float(p)}


def pins(root: Path | None = None) -> dict[str, str]:
    out = {}
    for line in (root or root_dir()).joinpath("requirements.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"not pinned: {line}")
        n, v = line.split("==", 1)
        out[n.lower()] = v
    return out


def check_licenses(root: Path | None = None) -> None:
    root = root or root_dir()
    recs = json.loads((root / "licenses.json").read_text())
    for r in recs:
        if r["license"] not in OK_LICENSES:
            raise ValueError(r)
    have = {r["name"].lower(): r["version"] for r in recs}
    p = pins(root)
    if any(n not in have or have[n] != v for n, v in p.items()):
        raise AssertionError("licenses.json doesn't match requirements.txt")


def write_sbom(root: Path | None = None) -> Path:
    root = root or root_dir()
    recs = json.loads((root / "licenses.json").read_text())
    doc = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "components": [
            {"name": r["name"], "version": r["version"], "licenses": [{"license": {"id": r["license"]}}]}
            for r in recs
        ],
    }
    out = root / "sbom" / "chronis-ml.cdx.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out


def pip_audit() -> int:
    return subprocess.call([sys.executable, "-m", "pip_audit", "-r", "requirements.txt", "--progress-spinner", "off"])
