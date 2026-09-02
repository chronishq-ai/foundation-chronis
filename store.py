# model files live under models/<user_id>/  — that's the whole isolation story
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

USER_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
BAD_FNS = {"load_other_user", "merge_user_weights", "average_personal_lms"}


class IsolationError(PermissionError):
    pass


def ok_user(uid: str) -> str:
    if not USER_RE.match(uid) or uid == "_shared":
        raise IsolationError(f"bad id {uid!r}")
    return uid


def root_dir(start: Path | None = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for c in [p, *p.parents]:
        if (c / "pyproject.toml").exists():
            return c
    return Path.cwd()


def models_dir(root: Path | None = None) -> Path:
    return (root or root_dir()) / "models"


def user_dir(uid: str, kind: str, root: Path | None = None) -> Path:
    ok_user(uid)
    if kind not in {"hssm", "personal_lm"}:
        raise IsolationError("kind should be hssm or personal_lm")
    return models_dir(root) / uid / kind


def shared_dir(ckpt: str, root: Path | None = None) -> Path:
    if not USER_RE.match(ckpt):
        raise IsolationError("bad checkpoint name")
    return models_dir(root) / "_shared" / "pretrained" / ckpt


def check_path(uid: str, path: Path, root: Path | None = None) -> Path:
    ok_user(uid)
    base = models_dir(root).resolve()
    path = path.resolve()
    try:
        rel = path.relative_to(base)
    except ValueError as e:
        raise IsolationError("file is not in models/") from e
    owner = rel.parts[0]
    if owner == "_shared":
        if rel.parts[1] != "pretrained":
            raise IsolationError("shared folder is only for the base model")
        return path
    if owner != uid:
        raise IsolationError(f"{uid} can't read {owner}'s stuff")
    return path


class IsolatedModelStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else root_dir()
        self.models = models_dir(self.root)
        self.models.mkdir(parents=True, exist_ok=True)

    def write(self, uid: str, kind: str, name: str, data: bytes) -> Path:
        d = user_dir(uid, kind, self.root)
        d.mkdir(parents=True, exist_ok=True)
        p = check_path(uid, d / name, self.root)
        p.write_bytes(data)
        return p

    def read(self, uid: str, path: Path) -> bytes:
        return check_path(uid, path, self.root).read_bytes()

    def put_base(self, ckpt: str, data: bytes) -> Path:
        d = shared_dir(ckpt, self.root)
        d.mkdir(parents=True, exist_ok=True)
        w = d / "weights.bin"
        w.write_bytes(data)
        (d / "MANIFEST.json").write_text(
            json.dumps({"class": "A", "checkpoint_id": ckpt}), encoding="utf-8"
        )
        return w

    def load_base(self, ckpt: str) -> bytes:
        p = shared_dir(ckpt, self.root) / "weights.bin"
        if not p.exists():
            raise FileNotFoundError(ckpt)
        return p.read_bytes()


def scan(src: Path) -> list[str]:
    hits = []
    for py in src.rglob("*.py"):
        if "tests" in py.parts and py.name != "leaky.py":
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in BAD_FNS:
                hits.append(f"{py}: {node.name}")
    return hits


def assert_src_isolated(src: Path) -> None:
    hits = scan(src)
    if hits:
        raise IsolationError("isolation check failed:\n" + "\n".join(hits))
