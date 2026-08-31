# tiny stand-in for a personal lm. real one would be lora; this just proves isolation.
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from chronis_ml.store import IsolatedModelStore, IsolationError, shared_dir, user_dir

BASE = "chronis-base-v1"


def _hash(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode())
    return h.hexdigest()


@dataclass
class FineTuneResult:
    user_id: str
    adapter_path: Path
    base_checkpoint_id: str
    training_data_hash: str
    hyperparameters: dict
    metrics: dict
    fit_date: str


class PersonalLM:
    def __init__(self, store: IsolatedModelStore) -> None:
        self.store = store

    def ensure_base(self) -> str:
        p = shared_dir(BASE, self.store.root) / "weights.bin"
        if not p.exists():
            w = np.random.default_rng(7).normal(0, 0.02, 32).astype(np.float32)
            self.store.put_base(BASE, w.tobytes())
        return BASE

    def fine_tune(self, uid: str, transcripts: list[str], steps: int = 50, lr: float = 0.05) -> FineTuneResult:
        if not transcripts:
            raise ValueError("need transcripts")
        self.ensure_base()
        base = np.frombuffer(self.store.load_base(BASE), dtype=np.float32).copy()
        vec = np.zeros_like(base)
        for t in transcripts:
            raw = np.frombuffer(hashlib.sha256(t.encode()).digest(), dtype=np.uint8)
            vec += np.resize(raw.astype(np.float32) / 255.0, base.shape)
        vec /= len(transcripts)
        adapter = base + lr * vec
        for _ in range(steps):
            adapter = 0.99 * adapter + 0.01 * (base + vec)
        path = self.store.write(uid, "personal_lm", "adapter.bin", adapter.astype(np.float32).tobytes())
        today = datetime.now(timezone.utc).date().isoformat()
        meta = {
            "user_id": uid,
            "base_checkpoint_id": BASE,
            "training_data_hash": _hash(transcripts),
            "hyperparameters": {"steps": steps, "lr": lr, "init": "shared_pretrained"},
            "metrics": {"adapter_l2": float(np.linalg.norm(adapter - base)), "n_transcripts": len(transcripts)},
            "fit_date": today,
        }
        (user_dir(uid, "personal_lm", self.store.root) / "META.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        if self.store.load_base(BASE) == adapter.astype(np.float32).tobytes():
            raise IsolationError("adapter overwrote the shared base")
        return FineTuneResult(uid, path, BASE, meta["training_data_hash"], meta["hyperparameters"], meta["metrics"], today)

    def promote_to_global(self, uid: str) -> None:
        raise IsolationError(f"can't dump {uid}'s model into the shared one")
