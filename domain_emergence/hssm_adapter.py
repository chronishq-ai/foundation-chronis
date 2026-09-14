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

--- R2-S56.1 field-mapping fix (P0, item 1 of SPRINT5-6_Changes_Final) ---

The real `backbone.hssm.HSSMResult` (S34.7 canonical contract) does NOT
have `regime_sequence` or `observations` fields -- it has `p_t` (MAP
regime trajectory) and no field at all that echoes the input matrix
back. This adapter used to hard-require the old field names and raise
AttributeError on every real (non-mocked) call, which is the bug this
pass fixes:

  - `regime_sequence` <- `result.p_t`. `p_t` IS the MAP discrete
    regime trajectory (see backbone/hssm/fitting.py HSSMResult
    docstring) -- this is a rename, not a semantic change.
  - `observations` <- the adapter's own `matrix` input, echoed back.
    `HSSMResult` never carries the input matrix, so there is no
    "extract from result" option; the adapter already receives
    `matrix` as its own argument, so echoing it is the only source
    that exists.

FLAG (per S34.7 ownership -- Research ML owns the canonical export
shape, not this adapter): this mapping is the only mapping that fits
the data available, and is applied here so the pipeline is unblocked,
but it has NOT received a formal Research ML sign-off. If a future
`HSSMResult` revision adds a real `observations`-equivalent field (e.g.
because `p_t` alone is judged insufficient for context-signature
purposes) this mapping must be revisited then, not assumed permanent.
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
    regime_sequence: np.ndarray   # (T,) int -- sourced from HSSMResult.p_t
    observations: np.ndarray      # (T, F) float, NaN = missing session -- echoed input matrix


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
    local re-implementation of HSSM logic. Extracts/derives the two
    fields context_signature.py needs.

    Raises BackboneHSSMUnavailableError if backbone.hssm isn't
    installed, and AttributeError (with a clear message) if a real
    HSSMResult is missing the field this mapping depends on -- never
    silently substitutes synthetic data.

    Field mapping (see module docstring's R2-S56.1 section for why):
      regime_sequence <- result.p_t
      observations    <- echoed input `matrix` (HSSMResult carries no
                          input-echo field of its own)
    """
    fit_hssm = _import_fit_hssm()

    # Echo the input up front, in whatever numeric form context_signature.py
    # expects (float array, NaN = missing session) -- this is the adapter's
    # own argument, not something read off the result.
    input_matrix = np.asarray(matrix, dtype=float)

    result = fit_hssm(input_matrix)

    if not hasattr(result, "p_t") or result.p_t is None:
        raise AttributeError(
            "backbone.hssm.fit_hssm's HSSMResult is missing expected "
            "field 'p_t' (MAP regime trajectory) -- adapter contract "
            "mismatch, escalate to Research ML (S34.7 owns the "
            "canonical export shape)."
        )

    return HSSMAdapterOutput(
        regime_sequence=np.asarray(result.p_t),
        observations=input_matrix,
    )