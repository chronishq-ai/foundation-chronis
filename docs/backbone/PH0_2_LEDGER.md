# PH0.2 Exception and Invariant Ledger

## Closure decision

PH0.2 preserves the approved exception taxonomy and keeps Site #24 removed.
The implementation changes below are limited to the fitting and label
canonicalization boundaries.

## Audited sites

- **Site #9:** `bic()` first enforces the fitted lifecycle with
  `NotFittedError`. A fitted model whose `log_likelihood_` is missing raises
  `InternalStateError`, because the model contradicts its successful-fit
  contract.
- **Site #23:** `fit_with_random_restarts()` raises
  `FittingConvergenceError` only after valid restart attempts produce no
  converged winner.
- **Site #24:** Removed as structurally redundant. Site #23 establishes that
  `best_model` is non-`None`, and no operation currently rebinds it before
  canonicalization. Any future mutation boundary must establish its own
  invariant rather than adding a duplicate guard here.
- **Site #25:** Canonicalization is an in-place operation. The fitting boundary
  now rejects both `None` and a replacement object with `InternalStateError`.
  The test monkeypatches canonicalization after a genuine successful model
  selection and proves that the boundary is reached.
- **Site #26:** `select_k_by_bic()` uses the real production path and rejects a
  supposedly converged model with missing or non-finite log likelihood as
  `InternalStateError`. The error is not translated into convergence failure.
- **Site #27:** Empty candidates are rejected as `ValueError` before the
  fitting loop. The test proves that no fitting call occurs.

## Taxonomy

- `NotFittedError`: fitted-only operation requested before fitting.
- `ValueError`: invalid caller input or configuration.
- `FittingConvergenceError`: fitting was attempted but no acceptable model
  converged.
- `InternalStateError`: an internal contract or invariant was violated.

## Verification

The focused tests include representatives for all four exception classes,
real-path checks for Sites #25 and #26, a zero-side-effect check for Site #27,
and positive canonicalization checks covering the full regime permutation.