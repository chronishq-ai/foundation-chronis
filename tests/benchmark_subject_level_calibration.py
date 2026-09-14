"""
R2-S56.2 outstanding follow-up (item 2a's B2 benchmark): the module tests
prove correct DIRECTION (positive control finds the effect, null control
finds nothing) at reduced N for runtime. This script is the full-scale
empirical check the doc separately calls for:

  - >=2,000-dataset Type-I error calibration under the null (no true
    within-subject association): at alpha=0.05, the false-positive rate
    across independently generated null datasets should be close to 0.05
    -- not just "usually not significant on one dataset."
  - >=1,000-dataset power curve under a range of true effect sizes: at a
    given effect size, what fraction of independently generated datasets
    correctly reject the null.

This is a verification/calibration run, not a statistical-method-design
decision -- the method itself (restricted within-subject permutation,
Davison & Hinkley +1/+1 correction) was already implemented and is
Statistics/Research-lead-owned per the module's Ownership Model notes;
this script only measures how that already-chosen method actually
behaves at the scale the doc asked for, which is intern-safe.

Not wired into the pytest default run (2,000 + 4,000 permutation-test
calls at n_permutations=200 takes tens of seconds -- appropriate for an
explicit benchmark invocation, not every `pytest tests/`). Run directly:

    python tests/benchmark_subject_level_calibration.py

Reduced n_permutations (200, vs. the module's own default of 1,000) is
used here deliberately to keep >=3,000 total datasets tractable in this
sandbox; per-dataset permutation count and per-effect-size dataset count
are both parameters below and can be raised for a slower, tighter-CI re-run.
"""
from __future__ import annotations
import time
import numpy as np

from domain_emergence.domain_alignment import _subject_level_pvalue

ALPHA = 0.05
N_SUBJECTS = 20
WINDOWS_PER_SUBJECT = 10
N_PERMUTATIONS = 200


def _make_dataset(rng: np.random.Generator, effect_size: float):
    """effect_size in [0, 1]: probability that, within a subject whose
    behavioral label is 1, the narrative label is FORCED to match it
    (rather than drawn independently). effect_size=0 reproduces the
    null (fully independent draws); effect_size>0 injects a genuine
    within-subject behavioral<->narrative association without breaking
    subject structure or marginal label rates, so the permutation
    test's own null-preserving structure (subject membership, per-
    subject episode counts) is respected by the generator too."""
    n = N_SUBJECTS * WINDOWS_PER_SUBJECT
    subject_ids = np.repeat(np.arange(N_SUBJECTS), WINDOWS_PER_SUBJECT)
    behavioral = rng.integers(0, 2, size=n)
    narrative = rng.integers(0, 2, size=n)
    if effect_size > 0:
        force_match = rng.random(n) < effect_size
        narrative = np.where(force_match, behavioral, narrative)
    return behavioral, narrative, subject_ids


def run_type1_calibration(n_datasets: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n_rejections = 0
    for i in range(n_datasets):
        behavioral, narrative, subject_ids = _make_dataset(rng, effect_size=0.0)
        p = _subject_level_pvalue(
            behavioral, narrative, subject_ids, b_id=1, n_id=1,
            n_permutations=N_PERMUTATIONS, seed=int(rng.integers(0, 2**31 - 1)),
        )
        if p < ALPHA:
            n_rejections += 1
    rate = n_rejections / n_datasets
    # Wald 95% CI on the observed false-positive rate.
    se = (rate * (1 - rate) / n_datasets) ** 0.5
    return {
        "n_datasets": n_datasets,
        "false_positive_rate": rate,
        "ci_95": (max(0.0, rate - 1.96 * se), min(1.0, rate + 1.96 * se)),
        "nominal_alpha": ALPHA,
    }


def run_power_curve(n_datasets: int, effect_sizes: list, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    curve = {}
    for es in effect_sizes:
        n_rejections = 0
        for i in range(n_datasets):
            behavioral, narrative, subject_ids = _make_dataset(rng, effect_size=es)
            p = _subject_level_pvalue(
                behavioral, narrative, subject_ids, b_id=1, n_id=1,
                n_permutations=N_PERMUTATIONS, seed=int(rng.integers(0, 2**31 - 1)),
            )
            if p < ALPHA:
                n_rejections += 1
        curve[es] = n_rejections / n_datasets
    return {"n_datasets_per_effect_size": n_datasets, "power_curve": curve}


def main():
    t0 = time.time()
    calibration = run_type1_calibration(n_datasets=2000)
    t1 = time.time()
    power = run_power_curve(n_datasets=1000, effect_sizes=[0.0, 0.05, 0.10, 0.20, 0.40])
    t2 = time.time()

    print("=== Type-I error calibration (null, no true association) ===")
    print(f"  datasets: {calibration['n_datasets']}, nominal alpha: {calibration['nominal_alpha']}")
    print(f"  observed false-positive rate: {calibration['false_positive_rate']:.4f}")
    print(f"  95% CI: ({calibration['ci_95'][0]:.4f}, {calibration['ci_95'][1]:.4f})")
    print(f"  wall time: {t1 - t0:.1f}s")
    print()
    print("=== Power curve (true within-subject association at effect_size) ===")
    for es, power_rate in power["power_curve"].items():
        print(f"  effect_size={es:.2f} -> power={power_rate:.4f}  (n={power['n_datasets_per_effect_size']})")
    print(f"  wall time: {t2 - t1:.1f}s")


if __name__ == "__main__":
    main()