import sys
from types import ModuleType

# Mock missing claims_engine package and its submodules
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

# Initialize empty modules for chronis_ml to satisfy early imports
chronis_ml = ModuleType("chronis_ml")
chronis_ml_ops = ModuleType("chronis_ml.ops")
chronis_ml_store = ModuleType("chronis_ml.store")
chronis_ml_train = ModuleType("chronis_ml.train")

sys.modules["chronis_ml"] = chronis_ml
sys.modules["chronis_ml.ops"] = chronis_ml_ops
sys.modules["chronis_ml.store"] = chronis_ml_store
sys.modules["chronis_ml.train"] = chronis_ml_train

# Now, setup system path
import os
root = os.path.abspath(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

# Import the actual local flat modules in correct dependency order:
# 1. store
import store as real_store
for name, val in vars(real_store).items():
    setattr(chronis_ml_store, name, val)

# 2. train
import train as real_train
for name, val in vars(real_train).items():
    setattr(chronis_ml_train, name, val)

# 3. ops
import ops as real_ops
for name, val in vars(real_ops).items():
    setattr(chronis_ml_ops, name, val)

# Bind package properties
chronis_ml.ops = chronis_ml_ops
chronis_ml.store = chronis_ml_store
chronis_ml.train = chronis_ml_train
