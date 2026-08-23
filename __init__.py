# Mock claims_engine if not present in the environment
try:
    import claims_engine
except ImportError:
    import sys
    from types import ModuleType
    claims_engine = ModuleType("claims_engine")
    claims_engine.__path__ = []
    
    claim_levels = ModuleType("claims_engine.claim_levels")
    claim_levels.Claim = object
    claim_levels.ClaimLevel = object
    
    surfacing_policy = ModuleType("claims_engine.surfacing_policy")
    surfacing_policy.SurfaceDecision = object
    surfacing_policy.SurfacingResult = object
    
    sys.modules["claims_engine"] = claims_engine
    sys.modules["claims_engine.claim_levels"] = claim_levels
    sys.modules["claims_engine.surfacing_policy"] = surfacing_policy

from .index import INFLUENCE_FLAG, INFLUENCE_WINDOW_DAYS, SurfacedClaim, SurfacingIndex
from .observer import Observer, classify, cold_start_silent
from .profiles import TYPES, log_accuracy_mlflow, plant_profiles, type_accuracy
from .safeguard import Change, aspiration_evidence_weight, product_copy

__all__ = [
    "INFLUENCE_FLAG",
    "INFLUENCE_WINDOW_DAYS",
    "TYPES",
    "Change",
    "Observer",
    "SurfacedClaim",
    "SurfacingIndex",
    "aspiration_evidence_weight",
    "classify",
    "cold_start_silent",
    "log_accuracy_mlflow",
    "plant_profiles",
    "product_copy",
    "type_accuracy",
]
