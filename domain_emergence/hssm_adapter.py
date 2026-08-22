"""
S56.6 -- real Sprint 3-4 HSSM adapter, replacing the synthetic_hssm.py
stand-in on the default runtime path.

This is the ONLY module in domain_emergence/ that should import HSSM
output. context_signature.py / context_clustering.py / everything
downstream consumes `regime_sequence` (np.ndarray, shape (T,)) and
`observations` (np.ndarray, shape (T, F)) -- this adapter is
responsible for producing those two arrays from a real fit, so
downstream logic never has to know or care whether the source was
backbone.hssm or (in tests only) the relocated synthetic fixture.

Per S34.7, the canonical upstream entry point is:

    from backbone.hssm import fit_hssm
    result = fit_hssm(matrix)   # -> HSSMResult

`backbone` (Sprint 3-4's package) is not part of THIS delivered zip
(sprint 5-6 only) -- it ships separately. Importing it here is
therefore deferred (done lazily inside get_hssm_output, not at module
import time) so this module can still be imported/tested in isolation
before backbone.hssm is available in the environment, without masking
a real missing-dependency error with a fallback that silently swaps in
fake data.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


class BackboneHSSMUnavailableError(ImportError):
    """Raised when backbone.hssm.fit_hssm cannot be imported. Distinct
    exception type (not a bare ImportError) so callers can catch this
    specifically rather than accidentally swallowing unrelated import
    errors."""


@dataclass
class HSSMAdapterOutput:
    """Minimal shape context_signature.py actually consumes, extracted
    from the real HSSMResult. Kept as its own type (rather than reusing
    the retired SyntheticHSSMOutput) so nothing in production code can
    accidentally import the test-fixture dataclass."""
    regime_sequence: np.ndarray   # (T,) int
    observations: np.ndarray      # (T, F) float, NaN = missing session


def _import_fit_hssm():
    try:
        from backbone.hssm import fit_hssm  # canonical export, S34.7
    except ImportError as e:
        raise BackboneHSSMUnavailableError(
            "backbone.hssm.fit_hssm is not importable. This adapter "
            "requires the real Sprint 3-4 HSSM package (backbone/) to "
            "be installed/on PYTHONPATH -- it is not part of this "
            "sprint-5-6 package. Test code should use "
            "tests.fixtures.synthetic_hssm_fixture instead of calling "
            "this adapter."
        ) from e
    return fit_hssm


def get_hssm_output(matrix: np.ndarray) -> HSSMAdapterOutput:
    """Real call to backbone.hssm.fit_hssm(matrix) -> HSSMResult, no
    local re-implementation of HSSM logic. Extracts the two fields
    context_signature.py needs.

    Raises BackboneHSSMUnavailableError if backbone.hssm isn't
    installed, and AttributeError (with a clear message) if a
    real HSSMResult is missing an expected field -- never silently
    substitutes synthetic data for a missing/broken real result."""
    fit_hssm = _import_fit_hssm()
    result = fit_hssm(matrix)

    for field in ("regime_sequence", "observations"):
        if not hasattr(result, field):
            raise AttributeError(
                f"backbone.hssm.fit_hssm's HSSMResult is missing expected "
                f"field '{field}' -- adapter contract mismatch, escalate "
                f"to Research ML (S34.7 owns the canonical export shape)."
            )

    return HSSMAdapterOutput(
        regime_sequence=np.asarray(result.regime_sequence),
        observations=np.asarray(result.observations),
    )
