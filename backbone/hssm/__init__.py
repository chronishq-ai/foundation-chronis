from backbone.hssm.model import (
    GaussianHSMM,
    KimHSSMModel,
    FitState,
    HSSMError,
    NotFittedError,
    FittingConvergenceError,
    InternalStateError,
)
from backbone.hssm.fitting import fit_with_random_restarts, fit_dual_with_random_restarts, select_k_by_bic, fit_hssm_model, fit_hssm, HSSMResult
from backbone.hssm.label_switching import canonicalize_labels, activity_levels, canonicalize_regime_order
from backbone.hssm.gating import fit_hssm_gated, ColdStartError, count_present_sessions
from backbone.hssm.dual_model import DualTimescaleHSSM
from backbone.hssm.neural_residual import NeuralResidualAdapter, ResidualMode
from backbone.hssm.config import DEFAULT_FIT_CONFIG, DEFAULT_COLD_START_CONFIG

__all__ = [
    "GaussianHSMM", "KimHSSMModel", "DualTimescaleHSSM", "NeuralResidualAdapter", "ResidualMode", "FitState", "HSSMError", "NotFittedError", "FittingConvergenceError", "InternalStateError",
    "fit_with_random_restarts", "fit_dual_with_random_restarts", "select_k_by_bic", "fit_hssm_model",
    "fit_hssm", "HSSMResult",
    "HSSMFitDataset", "make_dataset", "compute_eligibility_mask",
    "canonicalize_labels", "activity_levels", "canonicalize_regime_order",
    "fit_hssm_gated", "ColdStartError", "count_present_sessions",
    "DEFAULT_FIT_CONFIG", "DEFAULT_COLD_START_CONFIG",
]

