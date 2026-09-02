from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DivergenceState:
    user_id: str
    domain_id: str
    state_id: str
    previous_state_id: str | None = None
    # Bible Part 5.5–5.7 type scores — produced by compute_divergence_state.
    type_scores: dict[str, float] = field(default_factory=dict)
    dominant_type: Optional[str] = None
    ambiguous: bool = False

    def dominant(self) -> Optional[str]:
        return self.dominant_type
