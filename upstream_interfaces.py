from dataclasses import dataclass

@dataclass
class AttractorRecord:
    user_id: str
    regime_id: int
    context_key: str
    revisit_count: int
    mean_dwell_time: float
    transition_stability: float
    declared: bool

@dataclass
class Domain:
    domain_id: str
    user_id: str
    label: str
    behavioral_regime_ids: list[int]
    narrative_regime_ids: list[int]
    confidence: float
    active: bool
    high_ignorance_prior: bool
    aspirational_or_hypothetical: bool
