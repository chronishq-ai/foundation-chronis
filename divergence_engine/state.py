from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class TypeScores:
    def dominant(self) -> Optional[str]:
        return "some_type"

@dataclass
class Provenance:
    power_gate_passed: bool

@dataclass
class DivergenceState:
    type_scores: TypeScores
    provenance: Provenance
