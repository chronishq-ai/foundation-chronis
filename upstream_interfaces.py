from dataclasses import dataclass
from typing import Any

@dataclass
class AttractorRecord:
    declared: bool

@dataclass
class Domain:
    confidence: float

@dataclass
class SessionExcerpt:
    text: str
    session_id: str
    timestamp: Any
