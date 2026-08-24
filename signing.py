"""Small Ed25519 signing boundary for Sprint 11 portable artifacts.

The module deliberately does not own or persist a user's private/device key.
Callers supply the device private key (or a DeviceSigner wrapping it). Only
canonical payload bytes are signed; verification uses the corresponding
public key.
"""
from __future__ import annotations

import base64
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


class SigningError(ValueError):
    pass


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"unsupported value for canonical signing: {type(value)!r}")


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, default=_json_default, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class DeviceSigner:
    """Ed25519 signer backed by a caller-owned device private key."""

    def __init__(self, private_key: Ed25519PrivateKey):
        if not isinstance(private_key, Ed25519PrivateKey):
            raise SigningError("private_key must be an Ed25519PrivateKey")
        self._private_key = private_key

    @classmethod
    def generate(cls) -> "DeviceSigner":
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_private_bytes(cls, raw: bytes) -> "DeviceSigner":
        return cls(Ed25519PrivateKey.from_private_bytes(raw))

    def sign(self, payload: Any) -> str:
        return base64.b64encode(self._private_key.sign(canonical_json(payload))).decode("ascii")

    def public_key_bytes(self) -> bytes:
        return self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def verify(self, payload: Any, signature: str) -> bool:
        try:
            self._private_key.public_key().verify(base64.b64decode(signature), canonical_json(payload))
            return True
        except Exception:
            return False


def verify_signature(payload: Any, signature: str, public_key_bytes: bytes) -> bool:
    try:
        public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
        public_key.verify(base64.b64decode(signature), canonical_json(payload))
        return True
    except Exception:
        return False
